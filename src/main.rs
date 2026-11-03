mod benchmark;
mod pipeline;
mod preprocess;
mod runtime;
mod server;
mod wheel;

use actix_web::{App, HttpServer, middleware, web};
use anyhow::{Context, Result};
use pipeline::Pipeline;
use std::{
    env,
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[actix_web::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<String> = env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "--help" || a == "-h") {
        println!(
            "电表推理服务器\n\ncargo run --release\ncargo run --release -- --check-models\ncargo run --release -- --predict <image>\ncargo run --release -- --benchmark <image> [iterations=30]\n\n环境变量: BIND_ADDR=127.0.0.1:8080 MODEL_DIR=models ORT_BACKEND=cpu|directml|directml-hybrid ORT_THREADS=<n> ORT_SPINNING=true|false ORT_DML_DEVICE_ID=<index> ORT_PROFILE_DIR=<dir>"
        );
        return Ok(());
    }
    anyhow::ensure!(
        args.is_empty()
            || args == ["--check-models"]
            || (args.len() == 2 && args[0] == "--predict")
            || ((args.len() == 2 || args.len() == 3) && args[0] == "--benchmark"),
        "参数无效，使用 --help 查看用法"
    );
    let models = PathBuf::from(env::var_os("MODEL_DIR").unwrap_or_else(|| "models".into()));
    let runtime = runtime::RuntimeConfig::from_env()?;
    anyhow::ensure!(
        !args.is_empty() || runtime.profile_dir.is_none(),
        "ORT_PROFILE_DIR is for --predict, --benchmark or --check-models; omit it when serving HTTP"
    );
    let mut pipeline = Pipeline::load(&models, &runtime).context("模型初始化失败")?;
    log::info!(
        "inference runtime: {}",
        serde_json::to_string(&pipeline.runtime)?
    );
    log::info!("已验证并加载 5 个模型，版本 {}", pipeline.bundle_version);
    if args == ["--check-models"] {
        for profile in pipeline.finish_profiles()? {
            log::info!("execution profile: {profile}");
        }
        println!("OK: 5 model hashes and tensor contracts verified");
        return Ok(());
    }
    if args.first().is_some_and(|a| a == "--predict") {
        let bytes = std::fs::read(&args[1])?;
        let image = preprocess::decode(&bytes)?;
        let result = pipeline.predict(&image, &bytes)?;
        for profile in pipeline.finish_profiles()? {
            log::info!("execution profile: {profile}");
        }
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    if args.first().is_some_and(|a| a == "--benchmark") {
        let iterations = args
            .get(2)
            .map_or(Ok(30), |value| value.parse::<usize>())
            .context("benchmark iterations must be an integer")?;
        anyhow::ensure!(
            (1..=10_000).contains(&iterations),
            "benchmark iterations must be 1..10000"
        );
        let mut report = benchmark::run(&mut pipeline, std::path::Path::new(&args[1]), iterations)?;
        report["execution_profiles"] = serde_json::to_value(pipeline.finish_profiles()?)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    let bind = env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let state = web::Data::new(server::State {
        bundle_version: pipeline.bundle_version.clone(),
        runtime: pipeline.runtime.clone(),
        pipeline: Arc::new(Mutex::new(pipeline)),
        admission: Arc::new(tokio::sync::Semaphore::new(4)),
    });
    log::info!("打开 http://{bind} · 推理服务 · 每文件上限 16 MiB");
    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .wrap(middleware::Logger::default())
            .wrap(
                middleware::DefaultHeaders::new()
                    .add(("X-Content-Type-Options", "nosniff"))
                    .add(("Referrer-Policy", "no-referrer")),
            )
            .configure(server::configure)
    })
    .workers(2)
    .bind(&bind)
    .with_context(|| format!("无法监听 {bind}"))?
    .run()
    .await?;
    Ok(())
}
