use crate::{
    preprocess,
    runtime::{RuntimeConfig, RuntimeInfo},
    wheel,
};
use anyhow::{Context, Result, ensure};
use image::RgbImage;
use ort::{
    environment::GlobalThreadPoolOptions,
    session::{Session, builder::GraphOptimizationLevel},
    value::{Outlet, TensorElementType, TensorRef},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, path::Path, time::Instant};

#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    pub xyxy: [f32; 4],
    pub score: f32,
    pub class_id: usize,
    pub class_name: String,
}

#[derive(Serialize)]
pub struct Slot {
    pub index: usize,
    pub xyxy_in_area: [f32; 4],
    pub xyxy_in_image: [f32; 4],
    pub crop_xyxy_in_image: [u32; 4],
    pub score: f32,
    pub independent_state: f64,
}

#[derive(Serialize)]
pub struct TypeScores {
    pub digital: f64,
    pub wheel: f64,
}

#[derive(Serialize)]
pub struct InferenceResult {
    pub width: u32,
    pub height: u32,
    pub source_sha256: String,
    pub meter_type: String,
    pub type_scores: TypeScores,
    pub area_candidates: Vec<Detection>,
    pub area: Option<Detection>,
    pub area_best_raw_score: f32,
    pub area_crop_xyxy: Option<[u32; 4]>,
    pub raw_digits: Option<String>,
    pub reading: Option<String>,
    pub status: String,
    pub reasons: Vec<String>,
    pub digit_detections: Vec<Detection>,
    pub all_symbols: Vec<Detection>,
    pub slots: Vec<Slot>,
    pub slot_best_raw_score: Option<f32>,
    pub prediction: Option<wheel::Prediction>,
    pub elapsed_ms: f64,
    pub timings_ms: HashMap<String, f64>,
    pub model_timings_ms: HashMap<String, f64>,
}

#[derive(Deserialize)]
struct TensorSpec {
    name: String,
    dtype: String,
    shape: Vec<serde_json::Value>,
}
#[derive(Deserialize)]
struct ModelSpec {
    file: String,
    bytes: u64,
    sha256: String,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}
#[derive(Deserialize)]
struct Margin {
    width_fraction_per_side: f32,
    height_fraction_per_side: f32,
}
#[derive(Deserialize)]
struct AreaContext {
    horizontal_fraction_per_side: f32,
    vertical_fraction_per_side: f32,
}
#[derive(Deserialize)]
struct Config {
    confidence_threshold: f32,
    nms_iou_threshold: f32,
    wheel_area_context: AreaContext,
    wheel_crop_margin: Margin,
    lcd_decimal_places: usize,
    wheel_decimal_places: usize,
}
#[derive(Deserialize)]
struct Manifest {
    bundle_version: String,
    models: HashMap<String, ModelSpec>,
    pipeline: Config,
}

pub struct Pipeline {
    pub bundle_version: String,
    pub runtime: RuntimeInfo,
    profiling: bool,
    sessions: HashMap<String, Session>,
    config: Config,
    preprocessor: preprocess::Preprocessor,
    classifier_preprocessor: preprocess::Preprocessor,
    model_timings: HashMap<String, f64>,
}

struct InitialStages {
    logits: Vec<f32>,
    candidates: Vec<Detection>,
    best_score: f32,
    source_sha256: String,
    classification_ms: f64,
    area_ms: f64,
}

fn validate_contract(actual: &[Outlet], specs: &[TensorSpec]) -> Result<()> {
    ensure!(actual.len() == specs.len(), "tensor count mismatch");
    for (tensor, spec) in actual.iter().zip(specs) {
        ensure!(
            tensor.name() == spec.name
                && spec.dtype == "float32"
                && tensor.dtype().tensor_type() == Some(TensorElementType::Float32),
            "tensor name/type mismatch: {}",
            spec.name
        );
        let shape = tensor.dtype().tensor_shape().context("expected tensor")?;
        ensure!(
            shape.len() == spec.shape.len(),
            "tensor rank mismatch: {}",
            spec.name
        );
        for (dimension, expected) in shape.iter().zip(&spec.shape) {
            ensure!(
                *dimension == expected.as_i64().unwrap_or(-1),
                "tensor shape mismatch: {}",
                spec.name
            );
        }
    }
    Ok(())
}

impl Pipeline {
    pub fn load(root: &Path, runtime: &RuntimeConfig) -> Result<Self> {
        runtime.validate()?;
        if let Some(directory) = &runtime.profile_dir {
            fs::create_dir_all(directory)?;
        }
        let manifest: Manifest = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
        let pool = GlobalThreadPoolOptions::default()
            .with_intra_threads(runtime.threads)?
            .with_inter_threads(1)?
            .with_spin_control(runtime.spinning)?;
        let env = ort::init()
            .with_name("meter-reader")
            .with_global_thread_pool(pool)
            .build()?;
        let mut sessions = HashMap::new();
        for role in [
            "classifier",
            "area",
            "lcd_reader",
            "wheel_slots",
            "wheel_reader",
        ] {
            let spec = manifest
                .models
                .get(role)
                .with_context(|| format!("missing model role {role}"))?;
            let path = root.join(&spec.file);
            let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            ensure!(
                bytes.len() as u64 == spec.bytes
                    && format!("{:x}", Sha256::digest(&bytes)) == spec.sha256,
                "model size/SHA256 mismatch: {}",
                path.display()
            );
            let builder = Session::builder(&env)?
                .with_optimization_level(GraphOptimizationLevel::All)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let session = runtime
                .configure_session(builder, role)?
                .commit_from_memory(&bytes)
                .with_context(|| format!("load {}", path.display()))?;
            validate_contract(session.inputs(), &spec.inputs)
                .with_context(|| format!("{role} inputs"))?;
            validate_contract(session.outputs(), &spec.outputs)
                .with_context(|| format!("{role} outputs"))?;
            tracing::info!(role, file = %path.display(), "model loaded and verified");
            sessions.insert(role.to_string(), session);
        }
        Ok(Self {
            bundle_version: manifest.bundle_version,
            runtime: runtime.info(),
            profiling: runtime.profile_dir.is_some(),
            sessions,
            config: manifest.pipeline,
            preprocessor: preprocess::Preprocessor::default(),
            classifier_preprocessor: preprocess::Preprocessor::default(),
            model_timings: HashMap::new(),
        })
    }

    /// Finalize profiling only after the last inference. Kernel events show the
    /// actual execution provider, including any CPU nodes in a DirectML session.
    pub fn finish_profiles(&mut self) -> Result<Vec<serde_json::Value>> {
        if !self.profiling {
            return Ok(Vec::new());
        }
        self.profiling = false;
        let mut profiles = Vec::new();
        for (role, session) in &mut self.sessions {
            let filename = session.end_profiling()?;
            let events: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(&filename)?)?;
            let mut providers = std::collections::BTreeMap::<String, usize>::new();
            for event in events {
                if let Some(provider) = event["args"]["provider"].as_str() {
                    *providers.entry(provider.into()).or_default() += 1;
                }
            }
            profiles.push(serde_json::json!({"role":role,"file":filename,"provider_execution_events":providers}));
        }
        profiles.sort_by(|a, b| a["role"].as_str().cmp(&b["role"].as_str()));
        Ok(profiles)
    }

    fn run(
        sessions: &mut HashMap<String, Session>,
        timings: &mut HashMap<String, f64>,
        role: &str,
        data: &[f32],
        shape: [usize; 4],
        expected: &[usize],
    ) -> Result<Vec<f32>> {
        let start = Instant::now();
        let input = TensorRef::from_array_view((shape, data))?;
        let session = sessions.get_mut(role).context("missing session")?;
        let output = session.run(ort::inputs![input])?;
        let (shape, values) = output[0].try_extract_tensor::<f32>()?;
        ensure!(
            shape.iter().copied().eq(expected.iter().map(|&n| n as i64))
                && values.iter().all(|v| v.is_finite()),
            "invalid {role} output: {shape:?}"
        );
        let values = values.to_vec();
        timings.insert(role.into(), ms(start));
        Ok(values)
    }

    fn detect(
        &mut self,
        role: &str,
        image: &RgbImage,
        classes: &[&str],
    ) -> Result<(Vec<Detection>, f32)> {
        let (tensor, transform) = self.preprocessor.detector(image);
        let output = Self::run(
            &mut self.sessions,
            &mut self.model_timings,
            role,
            tensor,
            [1, 3, 640, 640],
            &[1, 4 + classes.len(), 8400],
        )?;
        Ok(decode_detections(
            &output,
            transform,
            image.dimensions(),
            classes,
            self.config.confidence_threshold,
            self.config.nms_iou_threshold,
        ))
    }

    // Hashing and classifier preprocessing do not depend on the area detector.
    // Overlap this CPU work with area inference while keeping ORT calls serial,
    // so the shared inference pool never competes with another model's pool.
    fn initial_stages(&mut self, image: &RgbImage, source: &[u8]) -> Result<InitialStages> {
        let classifier_prep = &mut self.classifier_preprocessor;
        let area_prep = &mut self.preprocessor;
        let sessions = &mut self.sessions;
        let model_timings = &mut self.model_timings;
        let config = &self.config;
        std::thread::scope(|scope| {
            let prepared = scope.spawn(|| {
                let start = Instant::now();
                let tensor = classifier_prep.classifier(image);
                let elapsed = ms(start);
                let sha256 = format!("{:x}", Sha256::digest(source));
                (tensor, elapsed, sha256)
            });
            let start = Instant::now();
            let (tensor, transform) = area_prep.detector(image);
            let output = Self::run(
                sessions,
                model_timings,
                "area",
                tensor,
                [1, 3, 640, 640],
                &[1, 5, 8400],
            )?;
            let (candidates, best_score) = decode_detections(
                &output,
                transform,
                image.dimensions(),
                &["area"],
                config.confidence_threshold,
                config.nms_iou_threshold,
            );
            let area_ms = ms(start);
            let (tensor, preprocess_ms, source_sha256) = prepared
                .join()
                .map_err(|_| anyhow::anyhow!("classifier preprocessing thread panicked"))?;
            let start = Instant::now();
            let logits = Self::run(
                sessions,
                model_timings,
                "classifier",
                tensor,
                [1, 3, 256, 256],
                &[1, 2],
            )?;
            Ok(InitialStages {
                logits,
                candidates,
                best_score,
                source_sha256,
                classification_ms: preprocess_ms + ms(start),
                area_ms,
            })
        })
    }

    pub fn predict(&mut self, image: &RgbImage, source: &[u8]) -> Result<InferenceResult> {
        let start = Instant::now();
        self.model_timings.clear();
        let mut timings = HashMap::new();
        let initial = self.initial_stages(image, source)?;
        let scores = wheel::softmax(&initial.logits);
        let is_wheel = scores[1] > scores[0];
        timings.insert("classification".into(), initial.classification_ms);
        timings.insert("area_detection".into(), initial.area_ms);
        let candidates = initial.candidates;
        let best = initial.best_score;
        let selected = candidates
            .iter()
            .max_by(|a, b| {
                a.score
                    .total_cmp(&b.score)
                    .then_with(|| area(a.xyxy).total_cmp(&area(b.xyxy)))
            })
            .cloned();
        let (w, h) = image.dimensions();
        let mut result = InferenceResult {
            width: w,
            height: h,
            source_sha256: initial.source_sha256,
            meter_type: if is_wheel { "wheel" } else { "digital" }.into(),
            type_scores: TypeScores {
                digital: scores[0],
                wheel: scores[1],
            },
            area_candidates: candidates,
            area: selected.clone(),
            area_best_raw_score: best,
            area_crop_xyxy: None,
            raw_digits: None,
            reading: None,
            status: "no_area".into(),
            reasons: Vec::new(),
            digit_detections: Vec::new(),
            all_symbols: Vec::new(),
            slots: Vec::new(),
            slot_best_raw_score: None,
            prediction: None,
            elapsed_ms: 0.0,
            timings_ms: HashMap::new(),
            model_timings_ms: HashMap::new(),
        };
        if let Some(selected) = selected {
            let ctx = &self.config.wheel_area_context;
            let bounds = preprocess::crop_bounds(
                selected.xyxy,
                w,
                h,
                if is_wheel {
                    ctx.horizontal_fraction_per_side
                } else {
                    0.0
                },
                if is_wheel {
                    ctx.vertical_fraction_per_side
                } else {
                    0.0
                },
            );
            let cropped = preprocess::crop(image, bounds);
            result.area_crop_xyxy = Some(bounds);
            if is_wheel {
                self.wheel(image, &cropped, bounds, &mut result, &mut timings)?;
            } else {
                self.digital(&cropped, bounds, &mut result, &mut timings)?;
            }
        }
        result.timings_ms = timings;
        result.model_timings_ms = self.model_timings.clone();
        result.elapsed_ms = ms(start);
        Ok(result)
    }

    fn digital(
        &mut self,
        image: &RgbImage,
        b: [u32; 4],
        r: &mut InferenceResult,
        timings: &mut HashMap<String, f64>,
    ) -> Result<()> {
        let start = Instant::now();
        let (symbols, _) = self.detect(
            "lcd_reader",
            image,
            &[
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "dot", "minus",
            ],
        )?;
        let mut digits: Vec<_> = symbols
            .iter()
            .filter(|d| d.class_id < 10)
            .cloned()
            .collect();
        digits.sort_by(|a, b| (a.xyxy[0] + a.xyxy[2]).total_cmp(&(b.xyxy[0] + b.xyxy[2])));
        let raw = digits
            .iter()
            .map(|d| d.class_name.as_str())
            .collect::<String>();
        r.reading = wheel::format_reading(&raw, self.config.lcd_decimal_places);
        if b[0] == 0 || b[1] == 0 || b[2] == r.width || b[3] == r.height {
            r.reasons.push("window_touches_image_edge".into());
        }
        if r.reading.is_none() {
            r.reasons.push("too_few_digits".into());
        }
        r.raw_digits = Some(raw);
        r.digit_detections = digits;
        r.all_symbols = symbols;
        r.status = if r.reasons.is_empty() {
            "candidate"
        } else {
            "review"
        }
        .into();
        timings.insert("digit_recognition".into(), ms(start));
        Ok(())
    }

    fn wheel(
        &mut self,
        original: &RgbImage,
        image: &RgbImage,
        b: [u32; 4],
        r: &mut InferenceResult,
        timings: &mut HashMap<String, f64>,
    ) -> Result<()> {
        let start = Instant::now();
        let (mut slots, best) = self.detect("wheel_slots", image, &["wheel_slot"])?;
        slots.sort_by(|a, b| {
            (a.xyxy[0] + a.xyxy[2])
                .total_cmp(&(b.xyxy[0] + b.xyxy[2]))
                .then_with(|| (a.xyxy[1] + a.xyxy[3]).total_cmp(&(b.xyxy[1] + b.xyxy[3])))
        });
        r.slot_best_raw_score = Some(best);
        timings.insert("slot_detection".into(), ms(start));
        if slots.is_empty() {
            r.status = "no_slots".into();
            return Ok(());
        }
        // Dynamic wheel batches are deliberately bounded to avoid pathological allocations
        // and f64 counter overflow on false-positive detections.
        ensure!(slots.len() <= 32, "too many wheel detections (maximum 32)");
        let start = Instant::now();
        let mut cells = Vec::with_capacity(slots.len() * 32 * 48);
        for (index, slot) in slots.iter().enumerate() {
            let global = [
                slot.xyxy[0] + b[0] as f32,
                slot.xyxy[1] + b[1] as f32,
                slot.xyxy[2] + b[0] as f32,
                slot.xyxy[3] + b[1] as f32,
            ];
            let margin = &self.config.wheel_crop_margin;
            let crop = preprocess::crop_bounds(
                global,
                original.width(),
                original.height(),
                margin.width_fraction_per_side,
                margin.height_fraction_per_side,
            );
            cells.extend(preprocess::cell(&preprocess::crop(original, crop)));
            r.slots.push(Slot {
                index,
                xyxy_in_area: slot.xyxy,
                xyxy_in_image: global,
                crop_xyxy_in_image: crop,
                score: slot.score,
                independent_state: 0.0,
            });
        }
        let logits = Self::run(
            &mut self.sessions,
            &mut self.model_timings,
            "wheel_reader",
            &cells,
            [slots.len(), 1, 48, 32],
            &[slots.len(), 100],
        )?;
        let prediction = wheel::decode(&logits)?;
        r.raw_digits = Some(prediction.raw_digits.clone());
        r.reading = wheel::format_reading(&prediction.raw_digits, self.config.wheel_decimal_places);
        r.status = "review".into();
        if r.reading.is_none() {
            r.reasons.push("too_few_slots_for_decimal_format".into());
        }
        if ![5, 6].contains(&slots.len()) {
            r.reasons.push("unusual_slot_count_review_required".into());
        }
        for (slot, &state) in r.slots.iter_mut().zip(&prediction.independent_states) {
            slot.independent_state = state;
        }
        r.prediction = Some(prediction);
        timings.insert("wheel_recognition".into(), ms(start));
        Ok(())
    }
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}
fn area(b: [f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}
fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let intersection = area([
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]);
    intersection / (area(a) + area(b) - intersection).max(f32::EPSILON)
}

fn decode_detections(
    output: &[f32],
    t: preprocess::Letterbox,
    (w, h): (u32, u32),
    classes: &[&str],
    confidence: f32,
    threshold: f32,
) -> (Vec<Detection>, f32) {
    let n = output.len() / (4 + classes.len());
    let mut candidates = Vec::new();
    let mut best = 0.0f32;
    for i in 0..n {
        let mut class = 0;
        for c in 1..classes.len() {
            if output[(4 + c) * n + i] > output[(4 + class) * n + i] {
                class = c;
            }
        }
        let score = output[(4 + class) * n + i];
        best = best.max(score);
        if score <= confidence {
            continue;
        }
        let x = output[i] - output[2 * n + i] / 2.0;
        let y = output[n + i] - output[3 * n + i] / 2.0;
        candidates.push(Detection {
            xyxy: [x, y, x + output[2 * n + i], y + output[3 * n + i]],
            score,
            class_id: class,
            class_name: classes[class].into(),
        });
    }
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for candidate in candidates {
        if kept
            .iter()
            .all(|d| iou(candidate.xyxy, d.xyxy) <= threshold)
        {
            kept.push(candidate);
        }
    }
    for d in &mut kept {
        d.xyxy = [
            (d.xyxy[0] - t.left) / t.sx,
            (d.xyxy[1] - t.top) / t.sy,
            (d.xyxy[2] - t.left) / t.sx,
            (d.xyxy[3] - t.top) / t.sy,
        ];
        for (i, v) in d.xyxy.iter_mut().enumerate() {
            *v = v.clamp(0.0, if i % 2 == 0 { w as f32 } else { h as f32 });
        }
    }
    kept.retain(|d| area(d.xyxy) > 0.0);
    (kept, best)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nms_is_class_agnostic_and_threshold_is_strict() {
        let output = [
            10., 10., 30., 10., 10., 30., 10., 10., 4., 10., 10., 4., 0.9, 0.1, 0.25, 0.1, 0.8, 0.1,
        ];
        let (d, best) = decode_detections(
            &output,
            preprocess::Letterbox {
                sx: 1.,
                sy: 1.,
                left: 0.,
                top: 0.,
            },
            (100, 100),
            &["0", "1"],
            0.25,
            0.45,
        );
        assert_eq!(best, 0.9);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].class_name, "0");
        assert_eq!(d[0].xyxy, [5., 5., 15., 15.]);
    }
}
