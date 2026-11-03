use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Prediction {
    pub independent_raw_digits: String,
    pub independent_states: Vec<f64>,
    pub joint_value_in_last_wheel_units: f64,
    pub joint_log_score: f64,
    pub common_shift_bins: i32,
    pub raw_digits: String,
    pub author_rounded_raw_digits: String,
    pub logits: Vec<Vec<f32>>,
    pub probabilities: Vec<Vec<f64>>,
}

pub fn softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut p: Vec<_> = row.iter().map(|&v| (f64::from(v) - max).exp()).collect();
    let sum: f64 = p.iter().sum();
    for v in &mut p {
        *v /= sum;
    }
    p
}

fn argmax(row: &[f64]) -> usize {
    (1..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

fn mechanical(logp: &[[f64; 100]], shift: i32) -> (f64, f64) {
    let n = logp.len();
    let at = |k: usize, b: usize| logp[k][(b as i32 + shift).rem_euclid(100) as usize];
    let mut best: Vec<f64> = (0..100).map(|b| at(n - 1, b)).collect();
    let mut back = Vec::new();
    for k in (0..n - 1).rev() {
        let mut next = vec![f64::NEG_INFINITY; 100];
        let mut arg = [0; 100];
        for (right, &right_score) in best.iter().enumerate() {
            for digit in 0..10 {
                let b = digit * 10 + right.saturating_sub(90);
                let score = right_score + at(k, b);
                if score > next[b] {
                    next[b] = score;
                    arg[b] = right;
                }
            }
        }
        back.push(arg);
        best = next;
    }
    let mut b = argmax(&best);
    let score = best[b];
    let mut bins = vec![b];
    for arg in back.iter().rev() {
        b = arg[b];
        bins.push(b);
    }
    let mut value = 0.0;
    for b in &bins[..n - 1] {
        value = value * 10.0 + (b / 10) as f64;
    }
    (value * 10.0 + bins[n - 1] as f64 / 10.0, score)
}

pub fn decode(logits: &[f32]) -> Result<Prediction> {
    ensure!(
        !logits.is_empty()
            && logits.len().is_multiple_of(100)
            && logits.iter().all(|v| v.is_finite()),
        "wheel_reader must return finite [N,100] logits"
    );
    let rows: Vec<Vec<f32>> = logits
        .as_chunks::<100>()
        .0
        .iter()
        .map(|r| r.to_vec())
        .collect();
    let probabilities: Vec<_> = rows.iter().map(|r| softmax(r)).collect();
    let logp: Vec<[f64; 100]> = probabilities
        .iter()
        .map(|p| {
            std::array::from_fn(|b| {
                (p[(b + 99) % 100] + p[b] + p[(b + 1) % 100])
                    .max(1e-12)
                    .ln()
            })
        })
        .collect();
    let mut selected = (0.0, f64::NEG_INFINITY, 0);
    for shift in -2i32..=2 {
        let (value, score) = mechanical(&logp, shift);
        let score = score - 0.3 * f64::from(shift.abs());
        if score > selected.1 {
            selected = (value, score, shift);
        }
    }
    let bins: Vec<_> = rows
        .iter()
        .map(|r| argmax(&r.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()))
        .collect();
    let n = rows.len();
    // Format without an integer power or u64 conversion: arbitrary detected slot counts
    // cannot overflow; floating point precision for unusually many slots is flagged by caller.
    let format = |v: f64| {
        let value = format!("{:.0}", v.floor());
        let value = if value.len() > n {
            &value[value.len() - n..]
        } else {
            &value
        };
        format!("{:0>width$}", value, width = n)
    };
    Ok(Prediction {
        independent_raw_digits: bins
            .iter()
            .map(|b| char::from(b'0' + (b / 10) as u8))
            .collect(),
        independent_states: bins.iter().map(|&b| b as f64 / 10.0).collect(),
        joint_value_in_last_wheel_units: selected.0,
        joint_log_score: selected.1,
        common_shift_bins: selected.2,
        raw_digits: format(selected.0),
        author_rounded_raw_digits: format(selected.0 + 0.5),
        logits: rows,
        probabilities,
    })
}

pub fn format_reading(raw: &str, decimals: usize) -> Option<String> {
    (raw.len() > decimals).then(|| {
        if decimals == 0 {
            raw.to_owned()
        } else {
            format!(
                "{}.{}",
                &raw[..raw.len() - decimals],
                &raw[raw.len() - decimals..]
            )
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joint_decoder_preserves_leading_zeros_and_floor_policy() {
        let mut logits = vec![-20.0; 600];
        for (k, bin) in [0, 0, 30, 70, 30, 30].into_iter().enumerate() {
            logits[k * 100 + bin] = 20.0;
        }
        let p = decode(&logits).unwrap();
        assert_eq!(p.independent_raw_digits, "003733");
        assert_eq!(p.raw_digits, "003732");
        assert_eq!(p.author_rounded_raw_digits, "003733");
        assert_eq!(format_reading(&p.raw_digits, 1).as_deref(), Some("00373.2"));
    }
    #[test]
    fn invalid_logits_and_short_readings() {
        assert!(decode(&[]).is_err());
        assert!(decode(&[f32::NAN; 100]).is_err());
        assert_eq!(format_reading("01", 2), None);
        assert_eq!(format_reading("00123", 2).as_deref(), Some("001.23"));
    }
}
