use super::engine::{Engine, Live, SessionId};
use crate::sessions::{Session, SessionStore};
use anyhow::{Context, Result, ensure};

pub const SECTION_CODEC: usize = 4;
pub const SECTION_SAMPLER: usize = 5;
pub const SECTION_FRAME: usize = 6;
pub const SECTIONS: usize = 7;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect()
}

pub fn scope(engine: &Engine) -> [u8; 32] {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(b"duplex");
    h.update(engine.spec.name.as_bytes());
    h.finalize().into()
}

impl Engine {
    pub fn snapshot(&mut self, id: SessionId) -> Result<Session> {
        let live = self.live(id)?.clone();
        self.rt.synchronize()?;
        let page = crate::codegen::PAGE;
        let used = live.kept.div_ceil(page);
        let blocks: Vec<u32> = self.rt.block_table_of(live.slot)[..used].to_vec();
        let (k, v) = self.rt.save_kv_blocks(&blocks)?;
        let state = self.rt.snapshot_state_blocking(live.slot)?;
        let mut sections = crate::sessions::pack_sections(k, v, Some(&state));
        let (enc, dec) = self.slot_codec(live.slot)?;
        let mut codec = (enc.len() as u64).to_le_bytes().to_vec();
        codec.extend(f32_bytes(&enc));
        codec.extend(f32_bytes(&dec));
        sections.push(codec);
        sections.push(serde_json::to_vec(&live)?);
        let mut frame = live.frame.to_le_bytes().to_vec();
        frame.extend((live.kept as u64).to_le_bytes());
        frame.extend((used as u64).to_le_bytes());
        sections.push(frame);
        let mut tokens = live.text.history.clone();
        tokens.extend([live.token, live.frame as u32, live.kept as u32]);
        Ok(Session { scope: scope(self), tokens, sections })
    }

    pub fn restore(&mut self, s: &Session) -> Result<SessionId> {
        ensure!(s.sections.len() == SECTIONS, "a duplex snapshot has {SECTIONS} sections, this one {}", s.sections.len());
        ensure!(s.scope == scope(self), "the snapshot is for another model");
        let mut live: Live = serde_json::from_slice(&s.sections[SECTION_SAMPLER]).context("sampler section")?;
        let f = &s.sections[SECTION_FRAME];
        ensure!(f.len() == 24, "frame section of {} bytes", f.len());
        let word = |i: usize| u64::from_le_bytes(f[i * 8..(i + 1) * 8].try_into().unwrap());
        ensure!(word(0) == live.frame && word(1) as usize == live.kept, "the frame counter disagrees with the sampler state");
        let used = word(2) as usize;
        let state = crate::sessions::unpack_state(&s.sections[..4])?.context("a duplex snapshot carries the recurrent state")?;
        let c = &s.sections[SECTION_CODEC];
        ensure!(c.len() >= 8, "codec section too short");
        let enc_len = u64::from_le_bytes(c[..8].try_into().unwrap()) as usize;
        let all = bytes_f32(&c[8..]);
        ensure!(all.len() >= enc_len, "codec section too short");
        let slot = self.take_slot()?;
        let restored = (|| -> Result<()> {
            let pages = self.rt.kv_pages();
            let blocks: Vec<u32> = (0..pages as u32).map(|i| (slot * pages) as u32 + i).collect();
            self.rt.set_block_table(slot, &blocks)?;
            self.rt.load_kv_blocks(&blocks[..used], &s.sections[1], &s.sections[2])?;
            self.rt.restore_state(slot, &state)?;
            self.set_slot_codec(slot, &all[..enc_len], &all[enc_len..])?;
            let delta = (live.prompt_len as i64 + live.frame as i64 - 1) - (live.kept as i64 - 1);
            self.rt.set_position_offset(slot, delta as i32)?;
            Ok(())
        })();
        if let Err(e) = restored {
            self.give_slot(slot);
            return Err(e);
        }
        live.slot = slot;
        Ok(self.insert(live))
    }

    pub fn park(&mut self, id: SessionId, store: &SessionStore, now: u64) -> Result<String> {
        let s = self.snapshot(id)?;
        let entry = store.write(&s, now)?;
        self.end(id)?;
        Ok(entry.id)
    }

    pub fn resume(&mut self, store: &SessionStore, key: &str, now: u64) -> Result<SessionId> {
        let s = store.load(key, now)?;
        self.restore(&s)
    }
}
