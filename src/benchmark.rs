//! Serial full-pipeline benchmark. Every measured iteration decodes the image anew.
use crate::{pipeline::Pipeline, preprocess};
use anyhow::Result;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path, time::Instant};

fn stats(samples: &mut [f64]) -> Value {
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    json!({
        "mean_ms": mean,
        "p50_ms": samples[(samples.len().div_ceil(2)) - 1],
        "p95_ms": samples[(samples.len() * 95).div_ceil(100) - 1],
        "min_ms": samples[0],
        "max_ms": samples[samples.len() - 1],
        "fps": 1000.0 / mean,
    })
}

pub fn run(pipeline: &mut Pipeline, path: &Path, iterations: usize) -> Result<Value> {
    let bytes = fs::read(path)?;
    let image = preprocess::decode(&bytes)?;
    for _ in 0..5 {
        pipeline.predict(&image, &bytes)?;
    }
    let (width, height) = image.dimensions();
    drop(image);
    let mut total = Vec::with_capacity(iterations);
    let mut inference = Vec::with_capacity(iterations);
    let mut decoding = Vec::with_capacity(iterations);
    let mut stages = BTreeMap::<String, f64>::new();
    let mut models = BTreeMap::<String, f64>::new();
    let mut reading = None;
    let mut status = String::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let image = preprocess::decode(&bytes)?;
        decoding.push(start.elapsed().as_secs_f64() * 1000.0);
        let result = pipeline.predict(&image, &bytes)?;
        total.push(start.elapsed().as_secs_f64() * 1000.0);
        inference.push(result.elapsed_ms);
        for (key, time) in result.timings_ms {
            *stages.entry(key).or_default() += time / iterations as f64;
        }
        for (key, time) in result.model_timings_ms {
            *models.entry(key).or_default() += time / iterations as f64;
        }
        reading = result.reading;
        status = result.status;
    }
    Ok(json!({
        "filename": path.display().to_string(), "iterations": iterations, "warmup_iterations": 5,
        "threads": pipeline.runtime.threads, "spinning": pipeline.runtime.spinning,
        "runtime": pipeline.runtime, "bundle_version": pipeline.bundle_version,
        "width": width, "height": height, "reading": reading, "status": status,
        "inference": stats(&mut inference), "decode": stats(&mut decoding),
        "decode_and_inference": stats(&mut total), "stage_mean_ms": stages, "model_mean_ms": models,
    }))
}
