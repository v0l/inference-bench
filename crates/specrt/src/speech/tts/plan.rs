use tokenizers::Tokenizer;

#[derive(Clone, Debug, PartialEq)]
pub struct Rules {
    pub max_tokens: usize,
    pub frame_rate: f64,
    pub short_words: usize,
    pub frames_after_eos: usize,
    pub frames_after_eos_short: usize,
    pub eos_threshold: f32,
    pub min_frames_before_eos: usize,
    pub frame_budget: kir::expr::Expr,
}

impl Rules {
    pub fn from_spec(tts: &crate::spec::Tts) -> anyhow::Result<Rules> {
        let size = |n: &str| -> anyhow::Result<usize> { Ok(tts.param(n)? as usize) };
        Ok(Rules {
            max_tokens: size("max_tokens_per_chunk")?,
            frame_rate: tts.param("frame_rate")?,
            short_words: size("short_words")?,
            frames_after_eos: size("frames_after_eos")?,
            frames_after_eos_short: size("frames_after_eos_short")?,
            eos_threshold: tts.param("eos_threshold")? as f32,
            min_frames_before_eos: size("min_frames_before_eos")?,
            frame_budget: kir::expr::Expr::parse(&tts.frame_budget).map_err(anyhow::Error::msg)?,
        })
    }

    pub fn frame_budget(&self, tokens: usize) -> anyhow::Result<usize> {
        let env = |n: &str| match n {
            "tokens" => Some(tokens as f64),
            "frame_rate" => Some(self.frame_rate),
            _ => None,
        };
        let v = self.frame_budget.eval(&env).ok_or_else(|| anyhow::anyhow!("[tts] frame_budget {} reads more than tokens and frame_rate", self.frame_budget))?;
        Ok(v.max(1.0) as usize)
    }

    pub fn eos(&self, tail: usize) -> Eos {
        Eos { after: tail, threshold: self.eos_threshold, min_frames: self.min_frames_before_eos, frame: 0, countdown: None }
    }
}

#[derive(Clone, Debug)]
pub struct Chunk {
    pub text: String,
    pub tokens: Vec<u32>,
    pub frames_after_eos: usize,
    pub frame_budget: usize,
}

pub fn prepare(text: &str, rules: &Rules) -> (String, usize) {
    let text = text.trim();
    if text.is_empty() {
        return (String::new(), rules.frames_after_eos_short);
    }
    let mut text: String = text.replace(['\n', '\r'], " ").split_whitespace().collect::<Vec<_>>().join(" ");
    let words = text.split_whitespace().count();
    let tail = if words <= rules.short_words { rules.frames_after_eos_short } else { rules.frames_after_eos };
    let mut chars = text.chars();
    if let Some(first) = chars.next() {
        text = first.to_uppercase().collect::<String>() + chars.as_str();
    }
    if text.chars().last().is_some_and(|c| c.is_alphanumeric()) {
        text.push('.');
    }
    (text, tail)
}

fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    Ok(tok.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec())
}

pub fn split(tok: &Tokenizer, text: &str, rules: &Rules) -> Result<Vec<String>, String> {
    let (prepared, _) = prepare(text, rules);
    let tokens = encode(tok, prepared.trim())?;
    let marks = encode(tok, ".!...?")?;
    let marks = if marks.len() > 1 { &marks[1..] } else { &marks[..] };
    let mut cuts = vec![0usize];
    let mut after_mark = false;
    for (i, t) in tokens.iter().enumerate() {
        if marks.contains(t) {
            after_mark = true;
        } else {
            if after_mark {
                cuts.push(i);
            }
            after_mark = false;
        }
    }
    cuts.push(tokens.len());
    let mut chunks = Vec::new();
    let (mut current, mut count) = (String::new(), 0usize);
    for w in cuts.windows(2) {
        let n = w[1] - w[0];
        let sentence = tok.decode(&tokens[w[0]..w[1]], true).map_err(|e| e.to_string())?;
        if current.is_empty() {
            (current, count) = (sentence, n);
        } else if count + n > rules.max_tokens {
            chunks.push(current.trim().to_string());
            (current, count) = (sentence, n);
        } else {
            current.push(' ');
            current.push_str(&sentence);
            count += n;
        }
    }
    if !current.is_empty() {
        chunks.push(current.trim().to_string());
    }
    Ok(chunks)
}

pub fn chunks(tok: &Tokenizer, text: &str, rules: &Rules) -> anyhow::Result<Vec<Chunk>> {
    split(tok, text, rules)
        .map_err(anyhow::Error::msg)?
        .into_iter()
        .map(|text| {
            let (prepared, frames_after_eos) = prepare(&text, rules);
            let tokens = encode(tok, &prepared).map_err(anyhow::Error::msg)?;
            let frame_budget = rules.frame_budget(tokens.len())?;
            Ok(Chunk { text, tokens, frames_after_eos, frame_budget })
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub struct Eos {
    after: usize,
    threshold: f32,
    min_frames: usize,
    frame: usize,
    countdown: Option<usize>,
}

impl Eos {
    pub fn should_stop(&mut self, logit: f32) -> bool {
        let can_end = self.frame >= self.min_frames;
        self.frame += 1;
        if can_end && logit > self.threshold && self.countdown.is_none() {
            self.countdown = Some(self.after);
        }
        match self.countdown.as_mut() {
            None => false,
            Some(0) => true,
            Some(c) => {
                *c -= 1;
                false
            }
        }
    }

    pub fn ended(&self) -> bool {
        self.countdown.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> Rules {
        Rules { max_tokens: 50, frame_rate: 12.5, short_words: 4, frames_after_eos: 1, frames_after_eos_short: 3, eos_threshold: 4.0, min_frames_before_eos: 0, frame_budget: kir::expr::Expr::parse("ceil((tokens / 3 + 2) * frame_rate)").unwrap() }
    }

    #[test]
    fn a_text_is_given_a_capital_and_a_full_stop() {
        let r = rules();
        assert_eq!(prepare("not a thing", &r), ("Not a thing.".into(), 3));
        assert_eq!(prepare("one two three four five", &r), ("One two three four five.".into(), 1));
        assert_eq!(prepare("is it?", &r), ("Is it?".into(), 3));
        assert_eq!(prepare("one\r\ntwo   three\nfour five", &r), ("One two three four five.".into(), 1));
        assert_eq!(prepare("  ", &r), (String::new(), 3));
    }

    #[test]
    fn the_end_of_speech_waits_its_frames_then_stops() {
        let mut e = rules().eos(2);
        assert!(!e.should_stop(0.0));
        assert!(!e.should_stop(5.0));
        assert!(!e.should_stop(0.0));
        assert!(e.should_stop(0.0));
    }

    #[test]
    fn the_frame_budget_is_the_family_expression() {
        assert_eq!(rules().frame_budget(9).unwrap(), 63);
        assert_eq!(rules().frame_budget(0).unwrap(), 25);
    }
}
