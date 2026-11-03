use crate::{
    pipeline::{InferenceResult, Pipeline},
    preprocess,
};
use actix_multipart::Multipart;
use actix_web::{HttpResponse, ResponseError, http::StatusCode, web};
use futures_util::StreamExt;
use serde::Serialize;
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};
use tokio::sync::Semaphore;

pub const MAX_FILE_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_REQUEST_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_FILES: usize = 20;

pub struct State {
    pub bundle_version: String,
    pub runtime: crate::runtime::RuntimeInfo,
    pub pipeline: Arc<Mutex<Pipeline>>,
    pub admission: Arc<Semaphore>,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ApiError {
    status: StatusCode,
    message: String,
}
impl ApiError {
    fn bad(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
    fn large(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: message.into(),
        }
    }
    fn internal(error: impl std::fmt::Display) -> Self {
        log::error!("inference failed: {error}");
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "推理失败，请检查服务器日志".into(),
        }
    }
}
impl ResponseError for ApiError {
    fn status_code(&self) -> StatusCode {
        self.status
    }
    fn error_response(&self) -> HttpResponse {
        HttpResponse::build(self.status).json(
            serde_json::json!({"error": {"message": self.message, "status": self.status.as_u16()}}),
        )
    }
}

struct Upload {
    filename: String,
    bytes: Vec<u8>,
}

#[derive(Serialize)]
struct FileResult {
    filename: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<InferenceResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn read_uploads(mut payload: Multipart, batch: bool) -> Result<Vec<Upload>, ApiError> {
    let mut uploads = Vec::new();
    let mut total = 0usize;
    while let Some(field) = payload.next().await {
        let mut field = field.map_err(|e| ApiError::bad(format!("无效的 multipart 请求: {e}")))?;
        let disposition = field
            .content_disposition()
            .ok_or_else(|| ApiError::bad("缺少 Content-Disposition"))?;
        let name = disposition.get_name().unwrap_or("");
        if !(name == "image" || batch && (name == "images" || name == "files")) {
            return Err(ApiError::bad(if batch {
                "文件字段应为 images（或 files/image）"
            } else {
                "单图文件字段应为 image"
            }));
        }
        let filename = disposition
            .get_filename()
            .ok_or_else(|| ApiError::bad("缺少上传文件名"))?
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or("image")
            .chars()
            .take(200)
            .collect::<String>();
        if uploads.len() >= if batch { MAX_FILES } else { 1 } {
            return Err(ApiError::bad(if batch {
                "每批最多 20 张图片"
            } else {
                "单图接口只能上传一张图片，请使用 /api/batch"
            }));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = field.next().await {
            let chunk = chunk.map_err(|e| ApiError::bad(format!("读取上传失败: {e}")))?;
            total = total.saturating_add(chunk.len());
            if bytes.len().saturating_add(chunk.len()) > MAX_FILE_BYTES {
                return Err(ApiError::large("单张图片上限 16 MiB"));
            }
            if total > MAX_REQUEST_BYTES {
                return Err(ApiError::large("每批上传总大小上限 64 MiB"));
            }
            bytes.extend_from_slice(&chunk);
        }
        uploads.push(Upload { filename, bytes });
    }
    if uploads.is_empty() {
        return Err(ApiError::bad("请至少上传一张图片"));
    }
    Ok(uploads)
}

async fn infer(
    state: web::Data<State>,
    payload: Multipart,
    batch: bool,
) -> Result<HttpResponse, ApiError> {
    // The permit remains with the blocking task even if the HTTP client disconnects.
    let permit = state
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "服务器繁忙，请稍后重试".into(),
        })?;
    let uploads = read_uploads(payload, batch).await?;
    let pipeline = state.pipeline.clone();
    let start = Instant::now();
    let results = web::block(move || -> Result<Vec<FileResult>, ApiError> {
        let _permit = permit;
        let mut pipeline = pipeline.lock().map_err(ApiError::internal)?;
        let mut results = Vec::with_capacity(uploads.len());
        for upload in uploads {
            let image = match preprocess::decode(&upload.bytes) {
                Ok(image) => image,
                Err(error) => {
                    if !batch {
                        return Err(ApiError::bad(format!("图片无效: {error:#}")));
                    }
                    results.push(FileResult {
                        filename: upload.filename,
                        result: None,
                        error: Some(format!("图片无效: {error:#}")),
                    });
                    continue;
                }
            };
            match pipeline.predict(&image, &upload.bytes) {
                Ok(result) => results.push(FileResult {
                    filename: upload.filename,
                    result: Some(result),
                    error: None,
                }),
                Err(error) => {
                    let error = ApiError::internal(error);
                    if !batch {
                        return Err(error);
                    }
                    results.push(FileResult {
                        filename: upload.filename,
                        result: None,
                        error: Some(error.message),
                    });
                }
            }
        }
        Ok(results)
    })
    .await
    .map_err(ApiError::internal)??;
    if batch {
        let succeeded = results.iter().filter(|r| r.result.is_some()).count();
        Ok(HttpResponse::Ok().json(serde_json::json!({
            "bundle_version": state.bundle_version, "total": results.len(), "succeeded": succeeded,
            "failed": results.len() - succeeded, "elapsed_ms": start.elapsed().as_secs_f64() * 1000.0, "results": results,
        })))
    } else {
        let result = results
            .into_iter()
            .next()
            .ok_or_else(|| ApiError::internal("empty result"))?;
        Ok(
            HttpResponse::Ok().json(serde_json::json!({"bundle_version": state.bundle_version,
            "filename": result.filename, "result": result.result})),
        )
    }
}

async fn predict(state: web::Data<State>, payload: Multipart) -> Result<HttpResponse, ApiError> {
    infer(state, payload, false).await
}
async fn batch(state: web::Data<State>, payload: Multipart) -> Result<HttpResponse, ApiError> {
    infer(state, payload, true).await
}
async fn health(state: web::Data<State>) -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({"status":"ready", "bundle_version":state.bundle_version,
        "models_loaded":5, "backend":state.runtime.backend,"runtime":state.runtime, "limits":{"file_bytes":MAX_FILE_BYTES,
        "batch_bytes":MAX_REQUEST_BYTES,"batch_files":MAX_FILES,"image_pixels":preprocess::MAX_PIXELS}}))
}
async fn index() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(include_str!("../web/index.html"))
}
async fn script() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/javascript; charset=utf-8")
        .body(include_str!("../web/app.js"))
}
async fn styles() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/css; charset=utf-8")
        .body(include_str!("../web/style.css"))
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.route("/", web::get().to(index))
        .route("/app.js", web::get().to(script))
        .route("/style.css", web::get().to(styles))
        .route("/api/health", web::get().to(health))
        .route("/api/predict", web::post().to(predict))
        .route("/api/batch", web::post().to(batch));
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{App, test};
    async fn parse(payload: Multipart) -> Result<HttpResponse, ApiError> {
        let files = read_uploads(payload, false).await?;
        Ok(HttpResponse::Ok()
            .json(serde_json::json!({"filename":files[0].filename,"size":files[0].bytes.len()})))
    }
    #[actix_web::test]
    async fn multipart_validation() {
        let app = test::init_service(App::new().route("/", web::post().to(parse))).await;
        let body = "--test\r\nContent-Disposition: form-data; name=\"image\"; filename=\"../../meter.jpg\"\r\n\r\nabc\r\n--test--\r\n";
        let request = test::TestRequest::post()
            .uri("/")
            .insert_header(("Content-Type", "multipart/form-data; boundary=test"))
            .set_payload(body)
            .to_request();
        let response: serde_json::Value = test::call_and_read_body_json(&app, request).await;
        assert_eq!(response["filename"], "meter.jpg");
        assert_eq!(response["size"], 3);
        for body in [
            "--test--\r\n",
            "--test\r\nContent-Disposition: form-data; name=\"wrong\"; filename=\"a.jpg\"\r\n\r\nabc\r\n--test--\r\n",
        ] {
            let request = test::TestRequest::post()
                .uri("/")
                .insert_header(("Content-Type", "multipart/form-data; boundary=test"))
                .set_payload(body)
                .to_request();
            assert_eq!(
                test::call_service(&app, request).await.status(),
                StatusCode::BAD_REQUEST
            );
        }
    }
}
