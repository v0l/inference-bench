use specrt::nvrtc::{Source, download, is_installed};

#[test]
#[ignore = "talks to developer.download.nvidia.com and downloads about 95 MB"]
fn the_pinned_nvidia_package_downloads_two_dlls_with_the_pinned_checksums() {
    let cache = std::env::temp_dir().join(format!("nvrtc-live-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    let source = Source::nvidia_cuda_13();
    let dir = download(&cache, &source).unwrap();
    assert!(is_installed(&dir, &source));
    let _ = std::fs::remove_dir_all(cache);
}
