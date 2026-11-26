use anyhow::{Result, bail, ensure};

pub fn pcm16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect()
}

pub fn f32_to_pcm16(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|v| ((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()).collect()
}

pub fn pcm16_round(samples: &[f32]) -> Vec<f32> {
    pcm16_to_f32(&f32_to_pcm16(samples))
}

pub fn read_wav(b: &[u8]) -> Result<(Vec<f32>, u32)> {
    ensure!(b.len() > 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WAVE", "not a WAV file");
    let mut at = 12;
    let (mut format, mut channels, mut rate, mut bits) = (0u16, 0u16, 0u32, 0u16);
    while at + 8 <= b.len() {
        let id = &b[at..at + 4];
        let len = u32::from_le_bytes(b[at + 4..at + 8].try_into()?) as usize;
        let body = &b[at + 8..(at + 8 + len).min(b.len())];
        if id == b"fmt " {
            ensure!(body.len() >= 16, "a short fmt chunk");
            format = u16::from_le_bytes(body[0..2].try_into()?);
            channels = u16::from_le_bytes(body[2..4].try_into()?);
            rate = u32::from_le_bytes(body[4..8].try_into()?);
            bits = u16::from_le_bytes(body[14..16].try_into()?);
        } else if id == b"data" {
            let samples: Vec<f32> = match (format, bits) {
                (1, 16) => pcm16_to_f32(body),
                (3, 32) | (0xfffe, 32) => body.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
                _ => bail!("WAV format {format} with {bits} bits is not read; use PCM16 or float32"),
            };
            let ch = channels.max(1) as usize;
            return Ok((samples.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect(), rate));
        }
        at += 8 + len + (len & 1);
    }
    bail!("the WAV file has no data chunk")
}

pub fn wav_bytes(samples: &[f32], rate: u32) -> Vec<u8> {
    let data = f32_to_pcm16(samples);
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend(b"RIFF");
    out.extend(((36 + data.len()) as u32).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(rate.to_le_bytes());
    out.extend((rate * 2).to_le_bytes());
    out.extend(2u16.to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend((data.len() as u32).to_le_bytes());
    out.extend(data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wav_round_trips_through_pcm16() {
        let s = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.123];
        let (back, rate) = read_wav(&wav_bytes(&s, 24000)).unwrap();
        assert_eq!(rate, 24000);
        assert_eq!(back, pcm16_round(&s));
        assert!(back.iter().zip(&s).all(|(a, b)| (a - b).abs() < 1e-4));
        assert!(read_wav(b"nope").is_err());
    }
}
