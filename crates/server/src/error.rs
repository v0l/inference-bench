use axum::Json;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use inference_bench_core::result::{BackendError, ErrorKind};
use serde_json::json;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub message: String,
    pub retry_after: Option<u64>,
    pub code: Option<&'static str>,
    pub param: Option<&'static str>,
}

impl ApiError {
    pub fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self { status, kind, message: message.into(), retry_after: None, code: None, param: None }
    }

    pub fn with_code(mut self, code: &'static str, param: &'static str) -> Self {
        self.code = Some(code);
        self.param = Some(param);
        self
    }

    pub fn model_not_found(id: &str) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found_error", format!("the model {id} does not exist"))
            .with_code("model_not_found", "model")
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found_error", message)
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "authentication_error", message)
    }

    pub fn busy(retry_after: u64) -> Self {
        let mut e = Self::new(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", "the request queue is full, retry shortly");
        e.retry_after = Some(retry_after);
        e
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, "server_error", message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", message)
    }
}

impl From<BackendError> for ApiError {
    fn from(e: BackendError) -> Self {
        let (status, kind) = match e.kind {
            ErrorKind::ModelNotFound => (StatusCode::NOT_FOUND, "not_found_error"),
            ErrorKind::Unreachable => (StatusCode::BAD_GATEWAY, "upstream_error"),
            ErrorKind::UnsupportedArchitecture => (StatusCode::UNPROCESSABLE_ENTITY, "invalid_request_error"),
            ErrorKind::InvalidRequest => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            ErrorKind::Cancelled => (StatusCode::from_u16(499).unwrap(), "cancelled"),
            ErrorKind::Failed if e.message.contains("exceeds the context") || e.message.contains("can never fit") => {
                (StatusCode::BAD_REQUEST, "invalid_request_error")
            }
            ErrorKind::Failed => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        };
        Self::new(status, kind, e.message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({"error": {
            "message": self.message,
            "type": self.kind,
            "param": self.param,
            "code": self.code.map(String::from).unwrap_or_else(|| self.status.as_u16().to_string()),
        }});
        let mut response = (self.status, Json(body)).into_response();
        if let Some(secs) = self.retry_after {
            response.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(secs));
        }
        if self.status == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response
    }
}
