use crate::spec::{DryOptions, DuplexSampling};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

pub fn dry_penalties(history: &[u32], d: &DryOptions) -> BTreeMap<u32, f32> {
    let n = history.len();
    let mut longest: BTreeMap<u32, usize> = BTreeMap::new();
    if n < d.allowed_length.max(1) {
        return BTreeMap::new();
    }
    let last = history[n - 1];
    let lo = (n as isize - 1 - d.lookback as isize).max(0) as usize;
    for j in (lo..n.saturating_sub(1)).rev() {
        if history[j] != last {
            continue;
        }
        let mut length = 1;
        while length < d.max_match && j >= length && history[j - length] == history[n - 1 - length] {
            length += 1;
        }
        if length >= d.allowed_length {
            let e = longest.entry(history[j + 1]).or_insert(0);
            *e = (*e).max(length);
        }
    }
    longest.into_iter().map(|(t, l)| (t, (d.max_penalty).min(d.multiplier * d.base.powi((l - d.allowed_length) as i32)) as f32)).collect()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Free,
    Relay,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SayItem {
    pub tokens: Vec<u32>,
    pub said: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Interrupted {
    pub said: Vec<u32>,
    pub unsaid: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RowPlan {
    pub edits: Vec<(u32, f32)>,
    pub allow: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TextState {
    pub history: Vec<u32>,
    pub forced: u32,
    pub queue: VecDeque<SayItem>,
    pub mode: Mode,
    pub last: Option<u32>,
    pub pending: Vec<u32>,
    pub in_gap: bool,
}

impl TextState {
    pub fn new(forced: u32) -> TextState {
        TextState { forced, ..TextState::default() }
    }

    pub fn say(&mut self, tokens: Vec<u32>) {
        if !tokens.is_empty() {
            self.queue.push_back(SayItem { tokens, said: 0 });
        }
    }

    pub fn queued(&self) -> usize {
        self.queue.iter().map(|q| q.tokens.len() - q.said).sum()
    }

    pub fn next_queued(&self) -> Option<u32> {
        self.queue.front().map(|q| q.tokens[q.said])
    }

    pub fn interrupt(&mut self) -> Interrupted {
        let mut out = Interrupted::default();
        for q in self.queue.drain(..) {
            out.said.extend_from_slice(&q.tokens[..q.said]);
            out.unsaid.extend_from_slice(&q.tokens[q.said..]);
        }
        out
    }

    pub fn plan(&mut self, s: &DuplexSampling, pad: u32, eop: u32) -> RowPlan {
        let mut edits: Vec<(u32, f32)> = Vec::new();
        if let Some(d) = &s.dry {
            for (t, p) in dry_penalties(&self.history, d) {
                if !d.exempt_ids.contains(&t) {
                    edits.push((t, -p));
                }
            }
        }
        if s.eop_bias != 0.0 {
            edits.push((eop, s.eop_bias));
        }
        let forced = self.forced > 0;
        if forced {
            self.forced -= 1;
        }
        let allow = match (self.next_queued(), self.mode) {
            (Some(next), _) if forced => vec![next],
            (Some(next), _) => vec![pad, eop, next],
            (None, Mode::Relay) => vec![pad, eop],
            (None, Mode::Free) => {
                if forced {
                    edits.extend(s.forced_bans.iter().map(|&t| (t, f32::NEG_INFINITY)));
                }
                Vec::new()
            }
        };
        RowPlan { edits, allow }
    }

    pub fn accept(&mut self, token: u32, s: &DuplexSampling) {
        self.last = Some(token);
        if let Some(front) = self.queue.front_mut()
            && front.tokens[front.said] == token
        {
            front.said += 1;
            if front.said == front.tokens.len() {
                self.queue.pop_front();
            }
        }
        if let Some(d) = &s.dry
            && !d.exempt_ids.contains(&token)
        {
            self.history.push(token);
            let extra = self.history.len().saturating_sub(d.history_size());
            self.history.drain(..extra);
        }
    }

    pub fn replace_last(&mut self, token: u32, s: &DuplexSampling) {
        if let (Some(d), Some(prev)) = (&s.dry, self.last)
            && !d.exempt_ids.contains(&prev)
            && self.history.last() == Some(&prev)
        {
            self.history.pop();
        }
        self.last = Some(token);
        if let Some(d) = &s.dry
            && !d.exempt_ids.contains(&token)
        {
            self.history.push(token);
            let extra = self.history.len().saturating_sub(d.history_size());
            self.history.drain(..extra);
        }
    }

    pub fn text(&mut self, token: u32, silent: &[u32], decode: &dyn Fn(&[u32]) -> String) -> String {
        if silent.contains(&token) {
            if self.in_gap {
                return String::new();
            }
            self.in_gap = true;
            return " ".into();
        }
        self.in_gap = false;
        self.pending.push(token);
        let text = decode(&self.pending);
        if text.ends_with('\u{fffd}') {
            return String::new();
        }
        self.pending.clear();
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dry() -> DryOptions {
        DryOptions { multiplier: 0.8, base: 1.75, allowed_length: 1, lookback: 1024, max_match: 32, max_penalty: 32.0, exempt: Vec::new(), exempt_ids: vec![508, 509, 510] }
    }

    fn sampling() -> DuplexSampling {
        DuplexSampling { temperature: 0.7, top_k: 25, top_p: 1.0, eop_bias: 0.0, forced_text: 1, exact: vec![508, 509], silent: vec![508, 509], forced_bans: vec![508, 509, 510], dry: Some(dry()) }
    }

    #[test]
    fn dry_matches_the_reference_on_known_histories() {
        let p = dry_penalties(&[1, 2, 3, 1, 2], &dry());
        assert_eq!(p.len(), 1);
        assert!((p[&3] - 0.8 * 1.75).abs() < 1e-6);
        let p = dry_penalties(&[5, 6, 5, 6, 5], &dry());
        assert!((p[&6] - 0.8 * 1.75f32.powi(2)).abs() < 1e-5, "{p:?}");
        let p = dry_penalties(&[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7], &dry());
        assert_eq!(p[&7], 32.0, "the cap");
        assert!(dry_penalties(&[1, 2, 3], &dry()).is_empty());
        let short = DryOptions { lookback: 2, ..dry() };
        assert!(dry_penalties(&[9, 1, 2, 3, 4, 9], &short).is_empty(), "the match is outside the lookback");
    }

    #[test]
    fn exempt_tokens_never_enter_the_history_and_the_history_is_bounded() {
        let s = sampling();
        let mut t = TextState::new(0);
        for tok in [508, 1, 509, 2, 510] {
            t.accept(tok, &s);
        }
        assert_eq!(t.history, vec![1, 2]);
        for i in 0..3000 {
            t.accept(i % 400, &s);
        }
        assert_eq!(t.history.len(), dry().history_size());
    }

    #[test]
    fn forced_text_bans_the_silent_and_end_tokens_once() {
        let s = sampling();
        let mut t = TextState::new(1);
        let first = t.plan(&s, 508, 509);
        assert!(first.allow.is_empty());
        let bans: Vec<u32> = first.edits.iter().filter(|e| e.1 == f32::NEG_INFINITY).map(|e| e.0).collect();
        assert_eq!(bans, vec![508, 509, 510]);
        assert!(t.plan(&s, 508, 509).edits.iter().all(|e| e.1.is_finite()));
    }

    #[test]
    fn eop_bias_and_dry_become_logit_edits() {
        let s = DuplexSampling { eop_bias: 1.5, ..sampling() };
        let mut t = TextState::new(0);
        for tok in [1, 2, 3, 1, 2] {
            t.accept(tok, &s);
        }
        let plan = t.plan(&s, 508, 509);
        assert!(plan.edits.contains(&(509, 1.5)));
        assert!(plan.edits.iter().any(|&(tok, a)| tok == 3 && (a + 1.4).abs() < 1e-6));
    }

    #[test]
    fn the_say_queue_allows_only_silence_and_the_next_token_and_pops_in_order() {
        let s = sampling();
        let mut t = TextState::new(0);
        t.say(vec![10, 11]);
        t.say(vec![12]);
        assert_eq!(t.plan(&s, 508, 509).allow, vec![508, 509, 10]);
        t.accept(508, &s);
        assert_eq!(t.next_queued(), Some(10));
        t.accept(10, &s);
        assert_eq!(t.plan(&s, 508, 509).allow, vec![508, 509, 11]);
        t.accept(11, &s);
        t.accept(12, &s);
        assert_eq!(t.queued(), 0);
        assert!(t.plan(&s, 508, 509).allow.is_empty(), "free mode samples freely once the queue is empty");
        t.mode = Mode::Relay;
        assert_eq!(t.plan(&s, 508, 509).allow, vec![508, 509]);
    }

    #[test]
    fn a_fed_back_token_replaces_the_sampled_one_in_the_dry_history() {
        let s = sampling();
        let mut fed = TextState::new(0);
        let mut own = TextState::new(0);
        for tok in [1, 2, 3, 1] {
            fed.accept(tok, &s);
            own.accept(tok, &s);
        }
        fed.accept(7, &s);
        fed.replace_last(2, &s);
        own.accept(2, &s);
        assert_eq!(fed.history, own.history);
        assert_eq!(fed.last, Some(2));
        assert_eq!(fed.plan(&s, 508, 509), own.plan(&s, 508, 509));
        fed.replace_last(508, &s);
        assert_eq!(fed.history, vec![1, 2, 3, 1], "an exempt fed token leaves no history entry");
        let mut silent = TextState::new(0);
        silent.accept(509, &s);
        silent.replace_last(4, &s);
        assert_eq!(silent.history, vec![4], "an exempt sampled token had no entry to drop");
    }

    #[test]
    fn interrupt_drops_the_queue_and_reports_what_was_said() {
        let s = sampling();
        let mut t = TextState::new(0);
        t.say(vec![20, 21, 22]);
        t.say(vec![23]);
        t.accept(20, &s);
        let r = t.interrupt();
        assert_eq!(r, Interrupted { said: vec![20], unsaid: vec![21, 22, 23] });
        assert_eq!(t.queued(), 0);
    }

    #[test]
    fn forced_text_with_a_queue_allows_only_the_queued_word() {
        let s = sampling();
        let mut t = TextState::new(1);
        t.say(vec![30]);
        assert_eq!(t.plan(&s, 508, 509).allow, vec![30]);
        t.mode = Mode::Relay;
        let mut r = TextState { mode: Mode::Relay, ..TextState::new(1) };
        assert_eq!(r.plan(&s, 508, 509).allow, vec![508, 509], "relay with an empty queue stays silent even when text is forced");
    }

    #[test]
    fn the_detokeniser_renders_a_silent_run_as_one_space_and_holds_partial_characters() {
        let decode = |ids: &[u32]| -> String {
            match ids {
                [1] => "Hi".into(),
                [2] => "\u{fffd}".into(),
                [2, 3] => "é".into(),
                _ => "?".into(),
            }
        };
        let mut t = TextState::new(0);
        let silent = [508, 509];
        let out: Vec<String> = [1, 508, 509, 508, 2, 3].iter().map(|&tok| t.text(tok, &silent, &decode)).collect();
        assert_eq!(out, vec!["Hi", " ", "", "", "", "é"]);
    }
}
