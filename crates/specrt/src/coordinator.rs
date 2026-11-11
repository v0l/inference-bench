use crate::cluster::{Event, Membership, Replan, diff};
use crate::discovery::Browser;
use crate::placement::{Placement, PlacementConfig};
use crate::spec::Spec;
use crate::stage::{Generated, generate};
use crate::wire::Chain;
use anyhow::{Context, Result, anyhow, bail};
use std::time::{Duration, Instant};

pub struct Coordinator {
    spec: Spec,
    base: PlacementConfig,
    key: Vec<u8>,
    timeout: Duration,
    members: Membership,
    browser: Option<Browser>,
    epoch: Instant,
    placement: Option<Placement>,
    chain: Option<Chain>,
    last_error: Option<String>,
    replans: u64,
}

impl Coordinator {
    pub fn new(spec: &Spec, base: PlacementConfig, key: Vec<u8>, timeout: Duration, member_ttl: Duration) -> Result<Coordinator> {
        let browser = match base.discovery {
            crate::placement::Discovery::Mdns => Some(Browser::start(base.mdns_port)?),
            crate::placement::Discovery::Static => None,
        };
        let members = Membership::from_static(&base.devices, member_ttl.as_millis() as u64);
        let mut c = Coordinator {
            spec: spec.clone(),
            base,
            key,
            timeout,
            members,
            browser,
            epoch: Instant::now(),
            placement: None,
            chain: None,
            last_error: None,
            replans: 0,
        };
        c.replan();
        Ok(c)
    }

    pub fn members(&self) -> &Membership {
        &self.members
    }

    pub fn placement(&self) -> Option<&Placement> {
        self.placement.as_ref()
    }

    pub fn replans(&self) -> u64 {
        self.replans
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn connected(&self) -> bool {
        self.chain.is_some()
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    pub fn apply(&mut self, event: Event) -> Option<Replan> {
        let before = self.placement.clone();
        if !self.members.apply(event) {
            return None;
        }
        self.chain = None;
        self.replan();
        Some(match (&before, &self.placement) {
            (Some(old), Some(new)) => diff(old, new),
            _ => Replan { moved_stages: self.placement.as_ref().map_or(0, |p| p.stages.len()), unchanged: false },
        })
    }

    fn replan(&mut self) {
        let previous = self.placement.take();
        self.replans += 1;
        match self.members.replan(&self.spec, &self.base) {
            Ok(new) => {
                if previous.as_ref().is_none_or(|old| !diff(old, &new).unchanged) {
                    self.chain = None;
                }
                self.placement = Some(new);
                self.last_error = None;
            }
            Err(e) => {
                self.chain = None;
                self.last_error = Some(format!("{e:#}"));
            }
        }
    }

    pub fn poll(&mut self) -> usize {
        let mut handled = 0;
        let mut events = Vec::new();
        if let Some(b) = &self.browser {
            while let Ok(e) = b.events().try_recv() {
                events.push(e);
            }
        }
        for name in self.members.expire(self.now_ms()) {
            events.push(Event::Leave { name });
        }
        for e in events {
            let e = match e {
                Event::Join { device, address, .. } => Event::Join { device, address, now_ms: self.now_ms() },
                other => other,
            };
            if let Event::Leave { name } = &e {
                if !self.members.names().contains(&name.as_str()) {
                    continue;
                }
            }
            if self.apply(e).is_some() {
                handled += 1;
            }
        }
        handled
    }

    fn connect(&mut self) -> Result<()> {
        let placement = self.placement.as_ref().ok_or_else(|| anyhow!("no placement: {}", self.last_error.as_deref().unwrap_or("no stages known")))?;
        let mut addrs = Vec::new();
        let mut ranges = Vec::new();
        for s in &placement.stages {
            let member = self.members.members().iter().find(|m| m.device.name == s.device_name).context("a placed device left the cluster")?;
            let Some(addr) = member.address.clone() else { bail!("device {} has no stage address", s.device_name) };
            addrs.push(addr);
            ranges.push((s.start, s.end));
        }
        self.chain = Some(Chain::connect_assigned(&addrs, &ranges, &self.key, self.timeout)?);
        Ok(())
    }

    pub fn wait_ready(&mut self, wait: Duration) -> Result<()> {
        let deadline = Instant::now() + wait;
        loop {
            self.poll();
            if self.chain.is_some() {
                return Ok(());
            }
            let outcome = if self.placement.is_some() { self.connect() } else { Err(anyhow!("{}", self.last_error.clone().unwrap_or_default())) };
            match outcome {
                Ok(()) => return Ok(()),
                Err(e) if Instant::now() >= deadline => return Err(e.context("the stage chain is not ready")),
                Err(e) => self.last_error = Some(format!("{e:#}")),
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    pub fn generate(&mut self, slot: u32, prompt: &[u32], new_tokens: usize, chunk: usize, wait: Duration) -> Result<Generated> {
        self.wait_ready(wait)?;
        let chain = self.chain.as_mut().expect("chain is ready");
        match generate(chain, slot, prompt, new_tokens, chunk) {
            Ok(g) => Ok(g),
            Err(e) => {
                self.chain = None;
                Err(e.context("request failed, the stage chain was lost"))
            }
        }
    }
}
