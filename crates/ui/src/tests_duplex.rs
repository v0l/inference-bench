use crate::tests::{harness_at, pump_until, shots_dir, with_gpu};
use crate::*;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use inference_bench_core::fake::{echo_duplex_loader, tiny_haru};
use inference_bench_core::library::LocalModel;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn shows(h: &Harness<'static, App>, text: &str) -> bool {
    h.query_all(egui_kittest::kittest::by().predicate(move |n| n.value().is_some_and(|v| v.contains(text)) || n.label().is_some_and(|l| l.contains(text)))).next().is_some()
}

fn haru() -> LocalModel {
    inference_bench_core::library::local_model_at(&tiny_haru()).expect("the tiny Haru fixture scans as a model")
}

fn duplex_app(name: &str, size: egui::Vec2) -> (Harness<'static, App>, Arc<AtomicUsize>) {
    let mut h = harness_at(name, Tab::Models, size);
    let mk = |i: usize, used: u64| inference_bench_core::gpu::GpuStats { name: format!("card{i}"), used_bytes: used, total_bytes: 96 << 30, util_percent: None, uuid: format!("GPU-{i}"), own_bytes: 0, display: false, index: Some(i as u32) };
    with_gpu(&mut h, 0, 96 << 30);
    h.state_mut().gpu = vec![mk(0, 90 << 30), mk(1, 1 << 30)];
    let loads = Arc::new(AtomicUsize::new(0));
    h.state().engine.set_duplex_loader(echo_duplex_loader(loads.clone()));
    h.state_mut().library.push(haru());
    (h, loads)
}

#[test]
fn the_library_takes_haru_support_from_the_engine() {
    let m = haru();
    let s = m.support_for("specrt").unwrap();
    assert!(s.supported && s.reason.contains("duplex"), "{}", s.reason);
    assert!(crate::duplex::is_duplex(&m));
}

#[test]
fn a_duplex_model_opens_its_own_load_page_and_loads_on_the_freest_gpu() {
    let (mut h, loads) = duplex_app("duplex-load", egui::vec2(1280.0, 800.0));
    let m = haru();
    h.state_mut().open_load_dialog(&m);
    h.run();
    assert!(h.state().load.is_none() && h.state().image_load.is_none(), "not the text or image dialog");
    let d = h.state().duplex_load.as_ref().unwrap();
    assert_eq!((d.device, d.sessions), (Some(1), 4));
    for label in ["SESSIONS", "CONTEXT", "PER SESSION", "fits"] {
        assert!(shows(&h, label), "{label}");
    }
    h.get_by_label("load model").click();
    h.run();
    pump_until(&mut h, 10, "duplex load", |a| a.loaded().is_some());
    h.run();
    let l = h.state().loaded().unwrap();
    assert_eq!(l.kind, specrt::media::ModelKind::Duplex);
    assert_eq!(l.engine.overrides.get(specrt::media::DEVICE_KEY), Some(&serde_json::json!(1)));
    assert_eq!(l.max_num_seqs, 4);
    assert_eq!(loads.load(Ordering::SeqCst), 1);
    assert_eq!(h.state().tab, Tab::Chat);
    assert!(h.state().speaking() && !h.state().generating());
    assert!(shows(&h, "8192 frames"));
    assert!(shows(&h, "start it on the Server tab"));
}

#[test]
fn the_duplex_pages_fit_a_phone_without_spilling_sideways() {
    let (mut h, _) = duplex_app("duplex-phone", egui::vec2(390.0, 844.0));
    h.state_mut().open_load_dialog(&haru());
    h.run();
    for label in ["load model", "cancel"] {
        let r = h.get_by_label(label).rect();
        assert!(r.right() <= 390.0 + 0.5 && r.left() >= 0.0, "{label} at {r:?}");
    }
    h.get_by_label("load model").click();
    h.run();
    pump_until(&mut h, 10, "duplex load", |a| a.loaded().is_some());
    h.run();
    let r = h.get_by_label("server tab").rect();
    assert!(r.right() <= 390.0 + 0.5, "server tab at {r:?}");
}

#[test]
#[ignore = "renders with wgpu; run with SHOTS=dir"]
fn screenshots_of_the_duplex_load_page_and_voice_page_at_desktop_and_phone_width() {
    let out = shots_dir();
    std::fs::create_dir_all(&out).unwrap();
    for (name, size) in [("desktop-1280x800", egui::vec2(1280.0, 800.0)), ("phone-390x844", egui::vec2(390.0, 844.0))] {
        let (mut h, _) = duplex_app(&format!("duplex-shot-{name}"), size);
        h.state_mut().open_load_dialog(&haru());
        h.run();
        h.render().unwrap().save(out.join(format!("duplex-load-{name}.png"))).unwrap();
        h.get_by_label("load model").click();
        h.run();
        pump_until(&mut h, 10, "load", |a| a.loaded().is_some());
        h.run();
        h.render().unwrap().save(out.join(format!("duplex-voice-{name}.png"))).unwrap();
    }
}
