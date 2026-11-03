use anyhow::{Context, Result, bail, ensure};
use ort::session::builder::SessionBuilder;
use serde::Serialize;
use std::{collections::BTreeMap, env, path::PathBuf, str::FromStr};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    Cpu,
    Directml,
    DirectmlHybrid,
}

impl FromStr for Backend {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "directml" => Ok(Self::Directml),
            "directml-hybrid" => Ok(Self::DirectmlHybrid),
            _ => bail!("ORT_BACKEND must be cpu, directml or directml-hybrid"),
        }
    }
}

impl Backend {
    pub fn uses_directml(self, role: &str) -> bool {
        match self {
            Self::Cpu => false,
            Self::Directml => true,
            Self::DirectmlHybrid => matches!(role, "area" | "lcd_reader" | "wheel_slots"),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeInfo {
    pub backend: Backend,
    pub threads: usize,
    pub spinning: bool,
    pub requested_device_id: Option<i32>,
    pub device_selection: String,
    // Registration policy, not proof that every node executes on the GPU.
    pub model_providers: BTreeMap<String, String>,
    pub cpu_node_fallback_allowed: bool,
}

pub struct RuntimeConfig {
    pub backend: Backend,
    pub threads: usize,
    pub spinning: bool,
    pub device_id: Option<i32>,
    pub profile_dir: Option<PathBuf>,
}

impl RuntimeConfig {
    pub fn from_env() -> Result<Self> {
        let backend: Backend = env::var("ORT_BACKEND")
            .unwrap_or_else(|_| "cpu".into())
            .parse()?;
        let default_threads = if backend == Backend::Cpu {
            std::thread::available_parallelism().map_or(4, |n| n.get().min(12))
        } else {
            1
        };
        let threads = env::var("ORT_THREADS")
            .unwrap_or_else(|_| default_threads.to_string())
            .parse()
            .context("ORT_THREADS must be a positive integer")?;
        let spinning = env::var("ORT_SPINNING")
            .unwrap_or_else(|_| (backend == Backend::Cpu).to_string())
            .parse()
            .context("ORT_SPINNING must be true or false")?;
        let device_id = env::var("ORT_DML_DEVICE_ID")
            .ok()
            .map(|value| value.parse::<i32>())
            .transpose()
            .context("ORT_DML_DEVICE_ID must be a nonnegative adapter index")?;
        let config = Self {
            backend,
            threads,
            spinning,
            device_id,
            profile_dir: env::var_os("ORT_PROFILE_DIR")
                .filter(|s| !s.is_empty())
                .map(PathBuf::from),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.threads > 0, "ORT_THREADS must be positive");
        ensure!(
            self.device_id.is_none_or(|id| id >= 0),
            "ORT_DML_DEVICE_ID must be nonnegative"
        );
        if self.backend != Backend::Cpu {
            ensure!(
                cfg!(all(feature = "directml", target_os = "windows")),
                "DirectML requires Windows and a build with --features directml"
            );
        }
        Ok(())
    }

    pub fn configure_session(
        &self,
        mut builder: SessionBuilder,
        role: &str,
    ) -> Result<SessionBuilder> {
        if self.backend.uses_directml(role) {
            #[cfg(all(feature = "directml", target_os = "windows"))]
            {
                use ort::ep::{DirectML, directml::PerformancePreference};
                let provider = if let Some(id) = self.device_id {
                    DirectML::default().with_device_id(id)
                } else {
                    DirectML::default()
                        .with_performance_preference(PerformancePreference::HighPerformance)
                };
                builder = builder
                    .with_memory_pattern(false)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .with_parallel_execution(false)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .with_execution_providers([provider.build().error_on_failure()])
                    .map_err(|e| anyhow::anyhow!("register DirectML for {role}: {e}"))?;
            }
            #[cfg(not(all(feature = "directml", target_os = "windows")))]
            bail!("DirectML requires Windows and --features directml");
        }
        if let Some(directory) = &self.profile_dir {
            builder = builder
                .with_profiling(directory.join(role))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        Ok(builder)
    }

    pub fn info(&self) -> RuntimeInfo {
        RuntimeInfo {
            backend: self.backend,
            threads: self.threads,
            spinning: self.spinning,
            requested_device_id: self.device_id,
            device_selection: if self.backend == Backend::Cpu {
                "cpu"
            } else if self.device_id.is_some() {
                "explicit-adapter-index"
            } else {
                "high-performance-auto"
            }
            .into(),
            model_providers: [
                "classifier",
                "area",
                "lcd_reader",
                "wheel_slots",
                "wheel_reader",
            ]
            .into_iter()
            .map(|role| {
                (
                    role.into(),
                    if self.backend.uses_directml(role) {
                        "DmlExecutionProvider"
                    } else {
                        "CPUExecutionProvider"
                    }
                    .into(),
                )
            })
            .collect(),
            cpu_node_fallback_allowed: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backend_selection_and_hybrid_policy() {
        assert_eq!(
            "directml-hybrid".parse::<Backend>().unwrap(),
            Backend::DirectmlHybrid
        );
        assert!("cuda".parse::<Backend>().is_err());
        assert!(Backend::DirectmlHybrid.uses_directml("area"));
        assert!(!Backend::DirectmlHybrid.uses_directml("wheel_reader"));
        assert!(Backend::Directml.uses_directml("wheel_reader"));
        assert!(!Backend::Cpu.uses_directml("area"));
    }
    #[test]
    fn invalid_thread_and_device_settings_are_rejected() {
        let mut config = RuntimeConfig {
            backend: Backend::Cpu,
            threads: 0,
            spinning: false,
            device_id: None,
            profile_dir: None,
        };
        assert!(config.validate().is_err());
        config.threads = 1;
        config.device_id = Some(-1);
        assert!(config.validate().is_err());
    }
}
