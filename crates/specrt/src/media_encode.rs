use crate::sched::MediaInput;
use crate::vision::{MediaPlan, Patches, VisionTokens};
use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

const BATCH_PATCHES: usize = crate::vision_encoder::WORKSPACE_PATCHES;

pub trait EncodeImages: Send + Sync {
    fn encode_images(&self, items: &[&Patches]) -> Result<Vec<Vec<f32>>>;
}

impl EncodeImages for crate::vision_encoder::VisionEncoder {
    fn encode_images(&self, items: &[&Patches]) -> Result<Vec<Vec<f32>>> {
        self.encode_batch(items)
    }
}

pub struct EmbedCache {
    limit_bytes: usize,
    bytes: usize,
    map: HashMap<[u8; 32], Arc<Vec<f32>>>,
    order: VecDeque<[u8; 32]>,
}

impl EmbedCache {
    pub fn new(limit_mb: usize) -> Self {
        EmbedCache { limit_bytes: limit_mb << 20, bytes: 0, map: HashMap::new(), order: VecDeque::new() }
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn get(&mut self, hash: &[u8; 32]) -> Option<Arc<Vec<f32>>> {
        let hit = self.map.get(hash)?.clone();
        if let Some(i) = self.order.iter().position(|h| h == hash) {
            self.order.remove(i);
        }
        self.order.push_back(*hash);
        Some(hit)
    }

    pub fn insert(&mut self, hash: [u8; 32], embeds: Arc<Vec<f32>>) {
        let size = embeds.len() * 4;
        if size > self.limit_bytes || self.map.contains_key(&hash) {
            return;
        }
        while self.bytes + size > self.limit_bytes {
            let Some(old) = self.order.pop_front() else { break };
            if let Some(v) = self.map.remove(&old) {
                self.bytes -= v.len() * 4;
            }
        }
        self.bytes += size;
        self.map.insert(hash, embeds);
        self.order.push_back(hash);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncodeStats {
    pub encoded: u64,
    pub cache_hits: u64,
    pub batches: u64,
}

pub struct MediaEncoder {
    encoder: Box<dyn EncodeImages>,
    tokens: VisionTokens,
    merge: usize,
    max_batch: usize,
    max_patches: usize,
    cache: Mutex<EmbedCache>,
    stats: Mutex<EncodeStats>,
    gpu: Mutex<()>,
}

pub struct MediaRequest<'a> {
    pub prompt: &'a [u32],
    pub media: &'a [MediaInput],
}

impl MediaEncoder {
    pub fn new(encoder: Box<dyn EncodeImages>, tokens: VisionTokens, merge: usize, opts: crate::spec::VisionOptions) -> Self {
        MediaEncoder {
            encoder,
            tokens,
            merge,
            max_batch: opts.max_batch.max(1),
            max_patches: BATCH_PATCHES,
            cache: Mutex::new(EmbedCache::new(opts.cache_mb)),
            stats: Mutex::new(EncodeStats::default()),
            gpu: Mutex::new(()),
        }
    }

    pub fn stats(&self) -> EncodeStats {
        *self.stats.lock().unwrap()
    }

    pub fn cached_bytes(&self) -> usize {
        self.cache.lock().unwrap().bytes()
    }

    pub fn plan(&self, prompt: &[u32], media: &[MediaInput]) -> Result<MediaPlan> {
        self.plan_batch(&[MediaRequest { prompt, media }]).pop().expect("one plan per request")
    }

    pub fn plan_batch(&self, requests: &[MediaRequest<'_>]) -> Vec<Result<MediaPlan>> {
        let layouts: Vec<Result<MediaPlan>> = requests
            .iter()
            .map(|r| {
                let processed: Vec<crate::vision::Processed> = r.media.iter().map(|m| m.processed.clone()).collect();
                let hashes: Vec<[u8; 32]> = r.media.iter().map(|m| m.hash).collect();
                crate::vision::plan_media(r.prompt, &self.tokens, self.merge, &processed, &hashes)
            })
            .collect();
        let mut found: HashMap<[u8; 32], Arc<Vec<f32>>> = HashMap::new();
        let mut wanted: Vec<&MediaInput> = Vec::new();
        {
            let mut cache = self.cache.lock().unwrap();
            let mut stats = self.stats.lock().unwrap();
            for (r, l) in requests.iter().zip(&layouts) {
                if l.is_err() {
                    continue;
                }
                for m in r.media {
                    if found.contains_key(&m.hash) || wanted.iter().any(|w| w.hash == m.hash) {
                        continue;
                    }
                    match cache.get(&m.hash) {
                        Some(e) => {
                            stats.cache_hits += 1;
                            found.insert(m.hash, e);
                        }
                        None => wanted.push(m),
                    }
                }
            }
        }
        let mut failed: HashMap<[u8; 32], String> = HashMap::new();
        for group in self.groups(&wanted) {
            let patches: Vec<&Patches> = group.iter().map(|m| m.processed.patches()).collect();
            let started = std::time::Instant::now();
            let out = {
                let _gpu = self.gpu.lock().unwrap();
                self.encoder.encode_images(&patches)
            };
            let ms = started.elapsed().as_secs_f64() * 1e3;
            match out {
                Ok(embeds) => {
                    tracing::info!(images = group.len(), patches = patches.iter().map(|p| p.data.len()).sum::<usize>(), encode_ms = ms, "vision encode");
                    let mut cache = self.cache.lock().unwrap();
                    let mut stats = self.stats.lock().unwrap();
                    stats.batches += 1;
                    stats.encoded += group.len() as u64;
                    for (m, e) in group.iter().zip(embeds) {
                        let e = Arc::new(e);
                        cache.insert(m.hash, e.clone());
                        found.insert(m.hash, e);
                    }
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    for m in &group {
                        failed.insert(m.hash, msg.clone());
                    }
                }
            }
        }
        layouts
            .into_iter()
            .zip(requests)
            .map(|(layout, r)| {
                let mut plan = layout?;
                let mut embeds = Vec::new();
                for m in r.media {
                    if let Some(e) = failed.get(&m.hash) {
                        anyhow::bail!("{e}");
                    }
                    embeds.extend_from_slice(found.get(&m.hash).context("media item was not encoded")?);
                }
                plan.embeds = Arc::new(embeds);
                Ok(plan)
            })
            .collect()
    }

    fn groups<'a>(&self, items: &[&'a MediaInput]) -> Vec<Vec<&'a MediaInput>> {
        let mut out: Vec<Vec<&MediaInput>> = Vec::new();
        let mut patches = 0;
        for &m in items {
            let n = m.processed.patches().grid.iter().product::<usize>();
            let full = out.last().is_none_or(|g| g.len() >= self.max_batch || patches + n > self.max_patches);
            if full {
                out.push(Vec::new());
                patches = 0;
            }
            patches += n;
            out.last_mut().expect("a group was pushed").push(m);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::Processed;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const WIDTH: usize = 4;

    struct Counting(Arc<AtomicUsize>);

    impl EncodeImages for Counting {
        fn encode_images(&self, items: &[&Patches]) -> Result<Vec<Vec<f32>>> {
            self.0.fetch_add(items.len(), Ordering::SeqCst);
            Ok(items.iter().map(|p| vec![p.data[0]; p.grid.iter().product::<usize>() / 4 * WIDTH]).collect())
        }
    }

    fn tokens() -> VisionTokens {
        VisionTokens { image: 9, video: 10, start: 11, end: 12 }
    }

    fn image(seed: f32, hash: u8) -> MediaInput {
        MediaInput { processed: Processed::Image(Patches { grid: [1, 4, 4], data: vec![seed; 16] }), hash: [hash; 32] }
    }

    fn encoder(count: Arc<AtomicUsize>, cache_mb: usize) -> MediaEncoder {
        MediaEncoder::new(Box::new(Counting(count)), tokens(), 2, crate::spec::VisionOptions { cache_mb, ..Default::default() })
    }

    const PROMPT: [u32; 6] = [1, 9, 9, 9, 9, 2];

    #[test]
    fn a_retried_image_is_served_from_the_cache_without_encoding() {
        let count = Arc::new(AtomicUsize::new(0));
        let enc = encoder(count.clone(), 16);
        let first = enc.plan(&PROMPT, &[image(1.0, 7)]).unwrap();
        let again = enc.plan(&PROMPT, &[image(1.0, 7)]).unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1, "the retry must not reach the encoder");
        assert_eq!(first, again);
        assert_eq!(enc.stats(), EncodeStats { encoded: 1, cache_hits: 1, batches: 1 });
        let other = enc.plan(&PROMPT, &[image(2.0, 8)]).unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 2, "a different image is encoded");
        assert_ne!(other.embeds, first.embeds);
    }

    #[test]
    fn requests_planned_together_share_one_encoder_call_and_one_copy_of_a_repeated_image() {
        let count = Arc::new(AtomicUsize::new(0));
        let enc = encoder(count.clone(), 0);
        let a = [image(1.0, 1)];
        let b = [image(2.0, 2)];
        let plans = enc.plan_batch(&[
            MediaRequest { prompt: &PROMPT, media: &a },
            MediaRequest { prompt: &PROMPT, media: &b },
            MediaRequest { prompt: &PROMPT, media: &a },
        ]);
        assert_eq!(count.load(Ordering::SeqCst), 2);
        assert_eq!(enc.stats().batches, 1);
        let plans: Vec<MediaPlan> = plans.into_iter().map(Result::unwrap).collect();
        assert_eq!(plans[0], plans[2]);
        assert_eq!(plans[1].embeds[0], 2.0);
        assert_eq!(enc.cached_bytes(), 0, "cache_mb = 0 keeps nothing");
    }

    #[test]
    fn a_bad_request_fails_alone() {
        let count = Arc::new(AtomicUsize::new(0));
        let enc = encoder(count, 16);
        let a = [image(1.0, 1)];
        let plans = enc.plan_batch(&[MediaRequest { prompt: &[1, 9, 2], media: &a }, MediaRequest { prompt: &PROMPT, media: &a }]);
        assert!(plans[0].is_err());
        assert!(plans[1].is_ok());
    }

    #[test]
    fn the_cache_evicts_the_least_recently_used_image_past_its_bound() {
        let mut c = EmbedCache::new(1);
        let quarter = || Arc::new(vec![0f32; (1 << 20) / 16]);
        for h in 0..4u8 {
            c.insert([h; 32], quarter());
        }
        assert_eq!(c.len(), 4);
        assert!(c.get(&[0; 32]).is_some());
        c.insert([9; 32], quarter());
        assert_eq!(c.len(), 4);
        assert!(c.get(&[1; 32]).is_none(), "the oldest unused entry goes first");
        assert!(c.get(&[0; 32]).is_some(), "a recent hit is kept");
        assert!(c.bytes() <= 1 << 20);
        c.insert([10; 32], Arc::new(vec![0f32; 1 << 20]));
        assert!(c.get(&[10; 32]).is_none(), "an item larger than the whole cache is not kept");
    }
}
