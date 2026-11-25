use anyhow::{Context, Result, ensure};
use openh264::encoder::{BitRate, Encoder, EncoderConfig, FrameRate, RateControlMode, UsageType};
use openh264::formats::{RgbSliceU8, YUVBuffer};

pub const TIMESCALE: u32 = 90000;

pub struct Sample {
    pub data: Vec<u8>,
    pub key: bool,
}

pub struct Stream {
    pub width: usize,
    pub height: usize,
    pub fps: u32,
    pub sps: Vec<u8>,
    pub pps: Vec<u8>,
    pub samples: Vec<Sample>,
}

fn nal_units(annex_b: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= annex_b.len() {
        if annex_b[i] == 0 && annex_b[i + 1] == 0 && annex_b[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::new();
    for (j, &s) in starts.iter().enumerate() {
        let mut end = starts.get(j + 1).map_or(annex_b.len(), |n| n - 3);
        while end > s && annex_b[end - 1] == 0 && starts.get(j + 1).is_some() {
            end -= 1;
        }
        out.push(&annex_b[s..end]);
    }
    out
}

pub fn even(v: usize) -> usize {
    v & !1
}

pub fn encode(frames: &[Vec<u8>], width: usize, height: usize, fps: u32) -> Result<Stream> {
    ensure!(!frames.is_empty(), "a video needs at least one frame");
    ensure!(fps > 0, "fps must be at least 1");
    ensure!(width >= 2 && height >= 2 && width.is_multiple_of(2) && height.is_multiple_of(2), "H.264 needs an even frame size, not {width}x{height}");
    let bps = ((width * height) as u64 * u64::from(fps) * 4).min(u64::from(u32::MAX)) as u32;
    let config = EncoderConfig::new().bitrate(BitRate::from_bps(bps)).max_frame_rate(FrameRate::from_hz(fps as f32)).usage_type(UsageType::CameraVideoRealTime).rate_control_mode(RateControlMode::Quality).skip_frames(false);
    let mut enc = Encoder::with_api_config(openh264::OpenH264API::from_source(), config).context("open the H.264 encoder")?;
    let mut stream = Stream { width, height, fps, sps: Vec::new(), pps: Vec::new(), samples: Vec::new() };
    for (i, rgb) in frames.iter().enumerate() {
        ensure!(rgb.len() == width * height * 3, "frame {i} holds {} bytes, not {width}x{height} RGB", rgb.len());
        let yuv = YUVBuffer::from_rgb8_source(RgbSliceU8::new(rgb, (width, height)));
        let bits = enc.encode(&yuv).with_context(|| format!("encode frame {i}"))?.to_vec();
        let mut sample = Vec::new();
        let mut key = false;
        for nal in nal_units(&bits) {
            let Some(&head) = nal.first() else { continue };
            match head & 0x1f {
                7 => stream.sps = nal.to_vec(),
                8 => stream.pps = nal.to_vec(),
                t => {
                    key |= t == 5;
                    sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                    sample.extend_from_slice(nal);
                }
            }
        }
        ensure!(!sample.is_empty(), "the encoder dropped frame {i}");
        stream.samples.push(Sample { data: sample, key });
    }
    ensure!(!stream.sps.is_empty() && !stream.pps.is_empty(), "the encoder wrote no SPS or PPS");
    Ok(stream)
}

fn put(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&((body.len() + 8) as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
}

fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 8);
    put(&mut out, kind, body);
    out
}

fn full(kind: &[u8; 4], version_flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = version_flags.to_be_bytes().to_vec();
    b.extend_from_slice(body);
    boxed(kind, &b)
}

fn be32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_be_bytes()).collect()
}

const MATRIX: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

impl Stream {
    pub fn frame_ticks(&self) -> u32 {
        TIMESCALE / self.fps
    }

    pub fn mp4(&self) -> Vec<u8> {
        let ftyp = boxed(b"ftyp", &[b"isom".as_slice(), &0x200u32.to_be_bytes(), b"isom", b"iso2", b"avc1", b"mp41"].concat());
        let mdat_body: Vec<u8> = self.samples.iter().flat_map(|s| s.data.iter().copied()).collect();
        let mdat_start = (ftyp.len() + 8) as u32;
        let n = self.samples.len() as u32;
        let tick = self.frame_ticks();
        let duration = n * tick;
        let (w, h) = (self.width as u32, self.height as u32);
        let mut mvhd = be32(&[0, 0, TIMESCALE, duration, 0x0001_0000]);
        mvhd.extend_from_slice(&[0x01, 0x00, 0, 0]);
        mvhd.extend_from_slice(&[0; 8]);
        mvhd.extend(be32(&MATRIX));
        mvhd.extend_from_slice(&[0; 24]);
        mvhd.extend(be32(&[2]));
        let mut tkhd = be32(&[0, 0, 1, 0, duration, 0, 0]);
        tkhd.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
        tkhd.extend(be32(&MATRIX));
        tkhd.extend(be32(&[w << 16, h << 16]));
        let mdhd = [be32(&[0, 0, TIMESCALE, duration]), vec![0x55, 0xc4, 0, 0]].concat();
        let hdlr = [be32(&[0]), b"vide".to_vec(), vec![0; 12], b"VideoHandler\0".to_vec()].concat();
        let vmhd = full(b"vmhd", 1, &[0; 8]);
        let dref = full(b"dref", 0, &[be32(&[1]), full(b"url ", 1, &[])].concat());
        let dinf = boxed(b"dinf", &dref);
        let mut avcc = vec![1, self.sps[1], self.sps[2], self.sps[3], 0xff, 0xe1];
        avcc.extend_from_slice(&(self.sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&self.sps);
        avcc.push(1);
        avcc.extend_from_slice(&(self.pps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&self.pps);
        let mut avc1 = vec![0; 6];
        avc1.extend_from_slice(&[0, 1]);
        avc1.extend_from_slice(&[0; 16]);
        avc1.extend_from_slice(&(w as u16).to_be_bytes());
        avc1.extend_from_slice(&(h as u16).to_be_bytes());
        avc1.extend(be32(&[0x0048_0000, 0x0048_0000, 0]));
        avc1.extend_from_slice(&[0, 1]);
        avc1.extend_from_slice(&[0; 32]);
        avc1.extend_from_slice(&[0, 0x18, 0xff, 0xff]);
        avc1.extend(boxed(b"avcC", &avcc));
        let stsd = full(b"stsd", 0, &[be32(&[1]), boxed(b"avc1", &avc1)].concat());
        let stts = full(b"stts", 0, &be32(&[1, n, tick]));
        let keys: Vec<u32> = self.samples.iter().enumerate().filter(|(_, s)| s.key).map(|(i, _)| i as u32 + 1).collect();
        let stss = full(b"stss", 0, &[be32(&[keys.len() as u32]), be32(&keys)].concat());
        let stsc = full(b"stsc", 0, &be32(&[1, 1, n, 1]));
        let sizes: Vec<u32> = self.samples.iter().map(|s| s.data.len() as u32).collect();
        let stsz = full(b"stsz", 0, &[be32(&[0, n]), be32(&sizes)].concat());
        let stco = full(b"stco", 0, &be32(&[1, mdat_start]));
        let stbl = boxed(b"stbl", &[stsd, stts, stss, stsc, stsz, stco].concat());
        let minf = boxed(b"minf", &[vmhd, dinf, stbl].concat());
        let mdia = boxed(b"mdia", &[full(b"mdhd", 0, &mdhd), full(b"hdlr", 0, &hdlr), minf].concat());
        let trak = boxed(b"trak", &[full(b"tkhd", 3, &tkhd), mdia].concat());
        let moov = boxed(b"moov", &[full(b"mvhd", 0, &mvhd), trak].concat());
        [ftyp, boxed(b"mdat", &mdat_body), moov].concat()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub timescale: u32,
    pub duration: u32,
    pub codec: String,
}

impl Header {
    pub fn fps(&self) -> f64 {
        f64::from(self.frames) * f64::from(self.timescale) / f64::from(self.duration.max(1))
    }
}

fn children(data: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 8 <= data.len() {
        let size = u32::from_be_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        if size < 8 || i + size > data.len() {
            break;
        }
        out.push((data[i + 4..i + 8].try_into().unwrap(), &data[i + 8..i + size]));
        i += size;
    }
    out
}

fn find<'a>(data: &'a [u8], path: &[&[u8; 4]]) -> Option<&'a [u8]> {
    let (first, rest) = path.split_first()?;
    let (_, body) = children(data).into_iter().find(|(k, _)| k == *first)?;
    if rest.is_empty() { Some(body) } else { find(body, rest) }
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

pub fn header(mp4: &[u8]) -> Result<Header> {
    let mdhd = find(mp4, &[b"moov", b"trak", b"mdia", b"mdhd"]).context("no mdhd box")?;
    let stbl = find(mp4, &[b"moov", b"trak", b"mdia", b"minf", b"stbl"]).context("no stbl box")?;
    let stsz = find(stbl, &[b"stsz"]).context("no stsz box")?;
    let stsd = find(stbl, &[b"stsd"]).context("no stsd box")?;
    let entry = &stsd[8..];
    let codec = String::from_utf8_lossy(entry.get(4..8).context("short stsd")?).to_string();
    let sample = &entry[8..];
    Ok(Header {
        width: u32::from(u16::from_be_bytes(sample.get(24..26).context("short avc1")?.try_into()?)),
        height: u32::from(u16::from_be_bytes(sample.get(26..28).context("short avc1")?.try_into()?)),
        frames: u32_at(stsz, 8).context("short stsz")?,
        timescale: u32_at(mdhd, 12).context("short mdhd")?,
        duration: u32_at(mdhd, 16).context("short mdhd")?,
        codec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(n: usize, w: usize, h: usize) -> Vec<Vec<u8>> {
        (0..n).map(|f| (0..w * h * 3).map(|i| ((i / 3 % w) * 255 / w + f * 20) as u8).collect()).collect()
    }

    #[test]
    fn the_header_reads_back_frame_count_size_and_fps() {
        let s = encode(&frames(5, 32, 32), 32, 32, 16).unwrap();
        assert!(s.samples[0].key);
        let h = header(&s.mp4()).unwrap();
        assert_eq!((h.width, h.height, h.frames, h.codec.as_str()), (32, 32, 5, "avc1"));
        assert!((h.fps() - 16.0).abs() < 1e-9, "{}", h.fps());
    }

    #[test]
    fn ffprobe_reads_the_frame_count_fps_and_size_and_ffmpeg_decodes_every_frame() {
        if std::process::Command::new("ffprobe").arg("-version").output().is_err() {
            eprintln!("ffprobe is not installed, nothing to compare");
            return;
        }
        let file = std::env::temp_dir().join(format!("specrt-mp4-{}.mp4", std::process::id()));
        std::fs::write(&file, encode(&frames(5, 32, 32), 32, 32, 16).unwrap().mp4()).unwrap();
        let out = std::process::Command::new("ffprobe").args(["-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames,nb_frames", "-of", "csv=p=0"]).arg(&file).output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
        eprintln!("ffprobe: {text} {}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(text, "h264,32,32,16/1,5,5");
        let raw = std::process::Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&file).args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).output().unwrap();
        let _ = std::fs::remove_file(&file);
        assert!(raw.stderr.is_empty(), "{}", String::from_utf8_lossy(&raw.stderr));
        assert_eq!(raw.stdout.len(), 5 * 32 * 32 * 3);
        let want = frames(5, 32, 32).concat();
        let err = raw.stdout.iter().zip(&want).map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs()).max().unwrap();
        assert!(err < 40, "decoded frames differ by {err} levels");
    }

    #[test]
    fn an_odd_size_or_a_short_frame_is_refused() {
        assert!(encode(&frames(1, 33, 32), 33, 32, 16).is_err());
        assert!(encode(&[vec![0; 10]], 32, 32, 16).is_err());
        assert!(encode(&[], 32, 32, 16).is_err());
    }
}
