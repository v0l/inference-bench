use serde_json::{Value, json};

pub struct ApiRoute {
    pub method: &'static str,
    pub path: &'static str,
    pub purpose: &'static str,
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

fn uptime(secs: u64) -> String {
    format!("{}h {:02}m {:02}s", secs / 3600, secs % 3600 / 60, secs % 60)
}

pub fn render(info: &Value) -> String {
    let text = |v: &Value| escape(v.as_str().unwrap_or(""));
    let mut rows = String::new();
    for r in info["routes"].as_array().into_iter().flatten() {
        rows.push_str(&format!(
            "<tr><td class=m>{}</td><td class=v>{}</td><td>{}</td></tr>",
            text(&r["method"]),
            text(&r["path"]),
            text(&r["purpose"])
        ));
    }
    let mut models = String::new();
    for m in info["models"].as_array().into_iter().flatten() {
        let ctx = m["max_ctx"].as_u64().map(|c| c.to_string()).unwrap_or_else(|| "default".into());
        models.push_str(&format!(
            "<tr><td class=v>{}</td><td>{}</td><td class=v>{}</td><td class=v>{}</td><td class=v>{}</td></tr>",
            text(&m["id"]),
            text(&m["state"]),
            ctx,
            m["parallel_requests"],
            gib(m["vram_bytes"].as_u64().unwrap_or(0)),
        ));
    }
    if models.is_empty() {
        models.push_str("<tr><td colspan=5>no models configured</td></tr>");
    }
    let gpu = match (info["gpu"]["used_bytes"].as_u64(), info["gpu"]["total_bytes"].as_u64()) {
        (Some(u), Some(t)) => format!("{} / {}", gib(u), gib(t)),
        (Some(u), None) => format!("{} used", gib(u)),
        _ => "not detected".into(),
    };
    let ttft = info["stats"]["ttft_p50_ms"].as_f64().map(|v| format!("{v:.0} ms")).unwrap_or_else(|| "n/a".into());
    let git = info["git_hash"].as_str().filter(|g| !g.is_empty()).map(|g| format!(" ({})", escape(g))).unwrap_or_default();
    let key = if info["auth_required"].as_bool().unwrap_or(false) {
        "required: send <span class=v>Authorization: Bearer &lt;key&gt;</span>"
    } else {
        "not required"
    };
    format!(
        "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\"><title>inference-bench</title><style>\
body{{background:#0d1117;color:#c9d1d9;font:15px/1.5 system-ui,sans-serif;margin:0;padding:2rem;max-width:62rem;margin:auto}}\
h1{{color:#f5a623;margin:0}}h2{{color:#f5a623;font-size:1rem;text-transform:uppercase;letter-spacing:.08em;margin-top:2rem}}\
.v,.m,code{{font-family:ui-monospace,Menlo,Consolas,monospace;color:#39d0e0}}.m{{color:#f5a623}}\
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:.3rem .8rem .3rem 0;border-bottom:1px solid #21262d}}th{{color:#8b949e;font-weight:normal}}\
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(12rem,1fr));gap:1rem}}.card{{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:.8rem 1rem}}\
.card b{{display:block;font:600 1.4rem ui-monospace,Menlo,Consolas,monospace;color:#39d0e0}}.card span{{color:#8b949e;font-size:.8rem}}\
</style></head><body>\
<h1>inference-bench</h1><div><span class=v>v{version}{git}</span> &middot; up <span class=v>{up}</span></div>\
<h2>Connect</h2><p>Base URL for an OpenAI client: <code>{base}</code><br>API key: {key}</p>\
<h2>Activity</h2><div class=grid>\
<div class=card><b>{requests}</b><span>requests served</span></div>\
<div class=card><b>{tokens}</b><span>tokens generated</span></div>\
<div class=card><b>{ttft}</b><span>TTFT p50</span></div>\
<div class=card><b>{gpu}</b><span>GPU memory used / total</span></div></div>\
<h2>Models</h2><table><tr><th>id</th><th>state</th><th>context</th><th>parallel</th><th>VRAM</th></tr>{models}</table>\
<h2>API</h2><table><tr><th>method</th><th>path</th><th>purpose</th></tr>{rows}</table>\
</body></html>",
        version = escape(info["version"].as_str().unwrap_or("")),
        git = git,
        up = uptime(info["uptime_secs"].as_u64().unwrap_or(0)),
        base = escape(info["base_url"].as_str().unwrap_or("")),
        key = key,
        requests = info["stats"]["requests"],
        tokens = info["stats"]["completion_tokens"],
        ttft = ttft,
        gpu = gpu,
        models = models,
        rows = rows,
    )
}

pub fn routes_json(routes: &[ApiRoute]) -> Value {
    Value::Array(routes.iter().map(|r| json!({"method": r.method, "path": r.path, "purpose": r.purpose})).collect())
}
