use crate::App;
use eframe::egui;
use egui_bench::prelude::*;
use inference_bench_core::specrt_sessions::now_seconds;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub id: String,
    pub tokens: usize,
    pub bytes: u64,
    pub age_secs: u64,
    pub expires_in_secs: u64,
    pub stripped: bool,
}

pub fn rows(shared: &inference_bench_core::specrt_sessions::Shared, now: u64) -> Vec<Row> {
    let ttl = shared.options.ttl_seconds();
    shared
        .store
        .list()
        .into_iter()
        .map(|e| Row {
            tokens: e.tokens,
            bytes: e.bytes,
            age_secs: now.saturating_sub(e.last_used),
            expires_in_secs: e.expires(ttl).saturating_sub(now),
            stripped: e.stripped,
            id: e.id,
        })
        .collect()
}

pub fn size(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{:.1} GB", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{:.0} MB", b as f64 / (1u64 << 20) as f64),
        b => format!("{:.0} KB", b as f64 / 1024.0),
    }
}

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    let Some(shared) = app.engine.handle().sessions() else { return };
    let list = rows(&shared, now_seconds());
    let stats = shared.stats();
    section(ui, "sessions", "idle conversations saved to disk, restored by the next request that continues them", |ui| {
        note(ui, format!("{} saved, {} on disk, {} restores, {} prompt tokens not prefilled", list.len(), size(stats.stored_bytes), stats.restores_ok, stats.prefill_tokens_saved), LEGEND);
        if list.is_empty() {
            note(ui, "no saved sessions", LEGEND);
        }
        for (i, r) in list.iter().enumerate() {
            ui.horizontal(|ui| {
                let state = if r.stripped { "stripped" } else { "saved" };
                note(ui, format!("{} {} tokens {} idle {} expires in {}", &r.id[..8.min(r.id.len())], r.tokens, size(r.bytes), crate::fmt::duration(r.age_secs as f64), crate::fmt::duration(r.expires_in_secs as f64)), VALUE);
                note(ui, state, LEGEND);
                if ui.add(Key::new("restore").outline(READOUT).enabled(!r.stripped).id_salt(("session-restore", i))).clicked() {
                    if let Err(e) = shared.store.verify(&r.id) {
                        app.notice = Some((false, format!("{e:#}")));
                    } else {
                        app.notice = Some((true, "session verified, the next request that continues it skips the prefill".into()));
                    }
                }
                if ui.add(Key::new("strip").outline(VALUE).enabled(!r.stripped).id_salt(("session-strip", i))).clicked() {
                    if let Err(e) = shared.store.strip(&r.id, now_seconds()) {
                        app.notice = Some((false, format!("{e:#}")));
                    }
                }
                if ui.add(Key::new("delete").outline(FAULT).id_salt(("session-delete", i))).clicked() {
                    let _ = shared.store.delete(&r.id);
                }
            });
        }
    });
}
