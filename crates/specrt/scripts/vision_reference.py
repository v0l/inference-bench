import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from transformers.video_utils import VideoMetadata

SMALL_SIZE = {"shortest_edge": 4096, "longest_edge": 1048576}
COMMON = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}


def make_image(path, h, w, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x + y) * 255 // (h + w - 2)], axis=-1)
    noise = rng.integers(0, 24, size=(h, w, 3))
    Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8)).save(path)


def make_video(path, w, h, seconds, fps):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}:duration={seconds}",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "10", str(path)],
        check=True,
    )


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), int(s["nb_read_frames"])


def decode_all(path, w, h):
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)


def save(out, name, patches, grid, extra=None):
    patches.numpy().astype("<f4").tofile(out / f"{name}.f32")
    meta = {"grid": [int(v) for v in grid], "patch_dim": int(patches.shape[1])}
    meta.update(extra or {})
    (out / f"{name}.json").write_text(json.dumps(meta))


def image_case(out, name, h, w, seed, size):
    path = out / f"{name}.png"
    make_image(path, h, w, seed)
    proc = Qwen2VLImageProcessor(size=size, **COMMON)
    res = proc(images=[Image.open(path).convert("RGB")], return_tensors="pt")
    save(out, name, res["pixel_values"], res["image_grid_thw"][0])


def video_case(out, name, w, h, seconds, fps, size):
    path = out / f"{name}.mp4"
    make_video(path, w, h, seconds, fps)
    vw, vh, vfps, total = probe(path)
    frames = torch.from_numpy(decode_all(path, vw, vh)).permute(0, 3, 1, 2)
    proc = Qwen3VLVideoProcessor(size=size, **COMMON)
    meta = VideoMetadata(total_num_frames=total, fps=vfps, width=vw, height=vh, duration=total / vfps)
    res = proc(videos=[frames], video_metadata=[meta], return_tensors="pt", return_metadata=True)
    md = res["video_metadata"][0] if "video_metadata" in res else meta
    indices = [int(i) for i in (md.frames_indices if md.frames_indices is not None else [])]
    save(out, name, res["pixel_values_videos"], res["video_grid_thw"][0],
         {"frames": total, "fps": vfps, "indices": indices})


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    image_case(out, "image_exact", 64, 64, 1, SMALL_SIZE)
    image_case(out, "image_resize", 70, 90, 2, SMALL_SIZE)
    image_case(out, "image_big", 200, 150, 3, {"shortest_edge": 65536, "longest_edge": 16777216})
    video_case(out, "video_exact", 64, 64, 2, 8, SMALL_SIZE)
    video_case(out, "video_resize", 90, 70, 3, 10, SMALL_SIZE)


main()
