use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Read;

pub use specrt::vision::{MediaItem, MediaKind};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MediaLimits {
    pub max_images: usize,
    pub max_videos: usize,
    pub max_image_bytes: usize,
    pub max_video_bytes: usize,
    pub allow_url_fetch: bool,
    pub fetch_timeout_secs: u64,
    pub max_pixels: Option<usize>,
    pub max_video_secs: Option<f64>,
}

impl MediaLimits {
    pub fn request_body_limit(&self) -> usize {
        let images = self.max_images * self.max_image_bytes;
        let videos = self.max_videos * self.max_video_bytes;
        ((images + videos) / 3 * 4 + (1 << 20)).max(2 << 20)
    }
}

impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            max_images: 8,
            max_videos: 1,
            max_image_bytes: 20 << 20,
            max_video_bytes: 200 << 20,
            allow_url_fetch: false,
            fetch_timeout_secs: 10,
            max_pixels: None,
            max_video_secs: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaError(pub String);

impl std::fmt::Display for MediaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn err<T>(m: impl Into<String>) -> Result<T, MediaError> {
    Err(MediaError(m.into()))
}

fn part_url<'a>(part: &'a Value, key: &str) -> Option<&'a str> {
    match &part[key] {
        Value::String(s) => Some(s),
        Value::Object(o) => o.get("url").and_then(Value::as_str),
        _ => None,
    }
}

pub fn decode_data_url(url: &str, max_bytes: usize) -> Result<Vec<u8>, MediaError> {
    let rest = url.strip_prefix("data:").ok_or_else(|| MediaError("not a data url".into()))?;
    let (head, body) = rest.split_once(',').ok_or_else(|| MediaError("malformed data url".into()))?;
    if !head.split(';').any(|p| p == "base64") {
        return err("data urls must be base64 encoded");
    }
    if body.len() / 4 * 3 > max_bytes + 3 {
        return err(format!("media is larger than the {max_bytes} byte limit"));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(body.trim())
        .map_err(|e| MediaError(format!("invalid base64 in data url: {e}")))?;
    if bytes.len() > max_bytes {
        return err(format!("media is larger than the {max_bytes} byte limit"));
    }
    Ok(bytes)
}

fn fetch(url: &str, limits: &MediaLimits, max_bytes: usize) -> Result<Vec<u8>, MediaError> {
    if !limits.allow_url_fetch {
        return err("fetching http(s) media urls is disabled: send a base64 data url, or set media.allow_url_fetch in the server config");
    }
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(limits.fetch_timeout_secs)))
        .build()
        .into();
    let resp = agent.get(url).call().map_err(|e| MediaError(format!("could not fetch {url}: {e}")))?;
    let mut bytes = Vec::new();
    resp.into_body()
        .into_reader()
        .take(max_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| MediaError(format!("could not read {url}: {e}")))?;
    if bytes.len() > max_bytes {
        return err(format!("media is larger than the {max_bytes} byte limit"));
    }
    Ok(bytes)
}

fn load(url: &str, limits: &MediaLimits, max_bytes: usize) -> Result<Vec<u8>, MediaError> {
    if url.starts_with("data:") {
        decode_data_url(url, max_bytes)
    } else if url.starts_with("http://") || url.starts_with("https://") {
        fetch(url, limits, max_bytes)
    } else {
        err("media urls must be data: or http(s):")
    }
}

pub fn has_media(messages: &[Value]) -> bool {
    messages.iter().any(|m| {
        m["content"]
            .as_array()
            .is_some_and(|parts| parts.iter().any(|p| matches!(p["type"].as_str(), Some("image_url" | "video_url" | "image" | "video"))))
    })
}

pub fn extract(messages: &[Value], limits: &MediaLimits) -> Result<Vec<MediaItem>, MediaError> {
    let mut out = Vec::new();
    let (mut images, mut videos) = (0, 0);
    for m in messages {
        let Some(parts) = m["content"].as_array() else { continue };
        for part in parts {
            let (kind, key) = match part["type"].as_str() {
                Some("image_url") => (MediaKind::Image, "image_url"),
                Some("video_url") => (MediaKind::Video, "video_url"),
                _ => continue,
            };
            let url = part_url(part, key).ok_or_else(|| MediaError(format!("{key} needs a url")))?;
            let (count, max, max_bytes, label) = match kind {
                MediaKind::Image => (&mut images, limits.max_images, limits.max_image_bytes, "images"),
                MediaKind::Video => (&mut videos, limits.max_videos, limits.max_video_bytes, "videos"),
            };
            *count += 1;
            if *count > max {
                return err(format!("at most {max} {label} per request are allowed"));
            }
            out.push(MediaItem::new(kind, load(url, limits, max_bytes)?));
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attachment {
    pub kind: MediaKind,
    pub mime: String,
    pub name: String,
    pub data: String,
}

impl Attachment {
    pub fn from_bytes(name: &str, bytes: &[u8]) -> Option<Self> {
        let (kind, mime) = sniff(name, bytes)?;
        Some(Self { kind, mime: mime.into(), name: name.into(), data: base64::engine::general_purpose::STANDARD.encode(bytes) })
    }

    pub fn from_path(path: &std::path::Path) -> Result<Self, MediaError> {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let bytes = std::fs::read(path).map_err(|e| MediaError(format!("{}: {e}", path.display())))?;
        Self::from_bytes(&name, &bytes).ok_or_else(|| MediaError(format!("{name} is not an image or a video the app can send")))
    }

    pub fn data_url(&self) -> String {
        format!("data:{};base64,{}", self.mime, self.data)
    }

    pub fn bytes(&self) -> Vec<u8> {
        base64::engine::general_purpose::STANDARD.decode(&self.data).unwrap_or_default()
    }

    pub fn part(&self) -> Value {
        match self.kind {
            MediaKind::Image => serde_json::json!({"type": "image_url", "image_url": {"url": self.data_url()}}),
            MediaKind::Video => serde_json::json!({"type": "video_url", "video_url": {"url": self.data_url()}}),
        }
    }
}

fn sniff(name: &str, bytes: &[u8]) -> Option<(MediaKind, &'static str)> {
    let image = |m| Some((MediaKind::Image, m));
    let video = |m| Some((MediaKind::Video, m));
    match bytes {
        [0x89, b'P', b'N', b'G', ..] => return image("image/png"),
        [0xff, 0xd8, 0xff, ..] => return image("image/jpeg"),
        [b'G', b'I', b'F', b'8', ..] => return image("image/gif"),
        [b'B', b'M', ..] => return image("image/bmp"),
        [b'R', b'I', b'F', b'F', _, _, _, _, b'W', b'E', b'B', b'P', ..] => return image("image/webp"),
        [0x1a, 0x45, 0xdf, 0xa3, ..] => return video("video/webm"),
        [_, _, _, _, b'f', b't', b'y', b'p', ..] => return video("video/mp4"),
        _ => {}
    }
    match name.rsplit('.').next().map(str::to_lowercase).as_deref() {
        Some("mov") => video("video/quicktime"),
        Some("mkv") => video("video/x-matroska"),
        _ => None,
    }
}

pub fn content_with_attachments(text: &str, attachments: &[Attachment]) -> Value {
    if attachments.is_empty() {
        return Value::String(text.to_string());
    }
    let mut parts: Vec<Value> = attachments.iter().map(Attachment::part).collect();
    parts.push(serde_json::json!({"type": "text", "text": text}));
    Value::Array(parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn data(bytes: &[u8]) -> String {
        format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes))
    }

    fn msg(parts: Value) -> Vec<Value> {
        vec![json!({"role": "user", "content": parts})]
    }

    #[test]
    fn data_urls_become_media_items_in_order() {
        let m = msg(json!([
            {"type": "text", "text": "what"},
            {"type": "image_url", "image_url": {"url": data(b"abc")}},
            {"type": "video_url", "video_url": {"url": data(b"vid")}},
            {"type": "image_url", "image_url": data(b"def")},
        ]));
        let limits = MediaLimits { max_videos: 1, ..Default::default() };
        let items = extract(&m, &limits).unwrap();
        let kinds: Vec<_> = items.iter().map(|i| i.kind).collect();
        assert_eq!(kinds, [MediaKind::Image, MediaKind::Video, MediaKind::Image]);
        assert_eq!(items[0].bytes.as_slice(), b"abc");
        assert_ne!(items[0].hash, items[2].hash);
        assert_eq!(items[0].hash, MediaItem::new(MediaKind::Image, b"abc".to_vec()).hash);
    }

    #[test]
    fn same_bytes_as_image_and_video_hash_differently() {
        assert_ne!(MediaItem::new(MediaKind::Image, b"x".to_vec()).hash, MediaItem::new(MediaKind::Video, b"x".to_vec()).hash);
    }

    #[test]
    fn urls_are_not_fetched_by_default() {
        let m = msg(json!([{"type": "image_url", "image_url": {"url": "http://127.0.0.1:1/x.png"}}]));
        let e = extract(&m, &MediaLimits::default()).unwrap_err();
        assert!(e.0.contains("disabled"), "{e}");
    }

    #[test]
    fn limits_count_images_and_bytes() {
        let two = msg(json!([
            {"type": "image_url", "image_url": {"url": data(b"a")}},
            {"type": "image_url", "image_url": {"url": data(b"b")}},
        ]));
        let e = extract(&two, &MediaLimits { max_images: 1, ..Default::default() }).unwrap_err();
        assert!(e.0.contains("at most 1 images"), "{e}");
        let big = msg(json!([{"type": "image_url", "image_url": {"url": data(&[0u8; 64])}}]));
        let e = extract(&big, &MediaLimits { max_image_bytes: 16, ..Default::default() }).unwrap_err();
        assert!(e.0.contains("byte limit"), "{e}");
    }

    #[test]
    fn malformed_urls_are_rejected() {
        for url in ["file:///etc/passwd", "data:image/png,abc", "data:image/png;base64,@@@@"] {
            let m = msg(json!([{"type": "image_url", "image_url": {"url": url}}]));
            assert!(extract(&m, &MediaLimits::default()).is_err(), "{url}");
        }
    }

    #[test]
    fn attachments_sniff_their_kind_and_round_trip_into_content_parts() {
        let png = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let a = Attachment::from_bytes("cat.png", &png).unwrap();
        assert_eq!((a.kind, a.mime.as_str()), (MediaKind::Image, "image/png"));
        let mp4 = [0, 0, 0, 0x18, b'f', b't', b'y', b'p', b'm', b'p', b'4', b'2'];
        assert_eq!(Attachment::from_bytes("clip.mp4", &mp4).unwrap().kind, MediaKind::Video);
        assert!(Attachment::from_bytes("notes.txt", b"hello").is_none());
        let content = content_with_attachments("what is it", &[a.clone()]);
        let items = extract(&[serde_json::json!({"role": "user", "content": content})], &MediaLimits::default()).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].bytes.as_slice(), png);
        assert_eq!(a.bytes(), png);
        assert_eq!(content_with_attachments("hi", &[]), Value::String("hi".into()));
    }

    #[test]
    fn the_body_limit_fits_the_configured_media_as_base64() {
        let l = MediaLimits { max_images: 2, max_image_bytes: 3 << 20, max_videos: 0, ..Default::default() };
        assert!(l.request_body_limit() >= 8 << 20);
        assert!(MediaLimits::default().request_body_limit() > 200 << 20);
    }

    #[test]
    fn plain_text_has_no_media() {
        let m = msg(json!([{"type": "text", "text": "hi"}]));
        assert!(!has_media(&m));
        assert!(extract(&m, &MediaLimits::default()).unwrap().is_empty());
        assert!(has_media(&msg(json!([{"type": "image_url", "image_url": {"url": "x"}}]))));
    }
}
