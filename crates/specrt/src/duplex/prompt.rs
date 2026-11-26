use crate::spec::Duplex;
use anyhow::{Result, ensure};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prompt {
    pub tokens: Vec<u32>,
    pub model_latents: Vec<f32>,
    pub user_latents: Vec<f32>,
    pub model_mask: Vec<f32>,
    pub user_mask: Vec<f32>,
}

impl Prompt {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

pub fn card_text(d: &Duplex, card: &str) -> String {
    if card.is_empty() { String::new() } else { d.card_template.replace("{card}", card) }
}

pub fn trim_reference(d: &Duplex, audio: &[f32]) -> Vec<f32> {
    let window = d.sample_rate / 50;
    let count = audio.len() / window.max(1);
    if count == 0 {
        return audio.to_vec();
    }
    let rms: Vec<f64> = (0..count).map(|i| (audio[i * window..(i + 1) * window].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / window as f64 + 1e-12).sqrt()).collect();
    let voiced: Vec<usize> = rms.iter().enumerate().filter(|(_, r)| 20.0 * r.log10() > d.trim_dbfs).map(|(i, _)| i).collect();
    let (Some(&first), Some(&last)) = (voiced.first(), voiced.last()) else { return Vec::new() };
    let margin = (d.trim_margin_seconds * d.sample_rate as f64) as usize;
    let start = (first * window).saturating_sub(margin);
    let end = ((last + 1) * window + margin).min(audio.len());
    let cut = &audio[start..end];
    let limit = (d.reference_seconds * d.sample_rate as f64) as usize;
    cut[cut.len().saturating_sub(limit)..].to_vec()
}

pub fn quietest_frame(d: &Duplex, reference: &[f32]) -> Option<usize> {
    let whole = reference.len() / d.frame_samples;
    (0..whole)
        .map(|i| (i, reference[i * d.frame_samples..(i + 1) * d.frame_samples].iter().map(|v| v * v).sum::<f32>() / d.frame_samples as f32))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

pub struct PromptParts<'a> {
    pub card: &'a [u32],
    pub reference: Option<&'a [f32]>,
    pub reference_latents: &'a [f32],
    pub silence: &'a [f32],
    pub speak_first: bool,
}

pub fn build(d: &Duplex, p: &PromptParts) -> Result<Prompt> {
    let dim = d.latent_dim;
    ensure!(p.reference_latents.len().is_multiple_of(dim) && p.silence.len() == dim, "reference latents or silence latent are not rows of {dim}");
    let n_ref = p.reference_latents.len() / dim;
    let mut grid: Vec<u32> = if p.speak_first { vec![d.pad, d.eop] } else { Vec::new() };
    if p.card.is_empty() && n_ref == 0 && grid.is_empty() {
        grid.push(d.pad);
    }
    let mut tokens = p.card.to_vec();
    if n_ref > 0 {
        tokens.extend(std::iter::repeat_n(d.pad, n_ref - 1));
        tokens.push(d.eop);
    }
    tokens.extend(&grid);
    let (n, n_card, n_grid) = (tokens.len(), p.card.len(), grid.len());
    let mut model_latents = vec![0.0; n * dim];
    let mut user_latents = vec![0.0; n * dim];
    model_latents[n_card * dim..(n_card + n_ref) * dim].copy_from_slice(p.reference_latents);
    if n_grid > 0 {
        let tone = match (p.reference, quietest_frame(d, p.reference.unwrap_or(&[]))) {
            (Some(_), Some(i)) if n_ref > 0 => &p.reference_latents[i * dim..(i + 1) * dim],
            _ => p.silence,
        };
        for row in n - n_grid..n {
            model_latents[row * dim..(row + 1) * dim].copy_from_slice(tone);
            user_latents[row * dim..(row + 1) * dim].copy_from_slice(p.silence);
        }
    }
    let model_mask = (0..n).map(|i| if i >= n_card { 1.0 } else { 0.0 }).collect();
    let user_mask = (0..n).map(|i| if i >= n - n_grid { 1.0 } else { 0.0 }).collect();
    Ok(Prompt { tokens, model_latents, user_latents, model_mask, user_mask })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn duplex() -> Duplex {
        let config = include_str!("../../tests/fixtures/haru/tiny/config.json");
        crate::spec::Spec::from_hf_config_duplex(config).unwrap().duplex.unwrap()
    }

    #[test]
    fn the_prompt_is_card_then_reference_then_the_opening_grid() {
        let d = duplex();
        let dim = d.latent_dim;
        let card = [1, 2, 3];
        let refs: Vec<f32> = (0..3 * dim).map(|i| i as f32).collect();
        let silence = vec![-1.0; dim];
        let mut clip = vec![0.5f32; 3 * d.frame_samples];
        clip[d.frame_samples..2 * d.frame_samples].fill(0.01);
        let p = build(&d, &PromptParts { card: &card, reference: Some(&clip), reference_latents: &refs, silence: &silence, speak_first: true }).unwrap();
        assert_eq!(p.tokens, vec![1, 2, 3, d.pad, d.pad, d.eop, d.pad, d.eop]);
        assert_eq!(p.model_mask, vec![0., 0., 0., 1., 1., 1., 1., 1.]);
        assert_eq!(p.user_mask, vec![0., 0., 0., 0., 0., 0., 1., 1.]);
        assert_eq!(&p.model_latents[6 * dim..7 * dim], &refs[dim..2 * dim], "room tone is the quietest reference frame");
        assert_eq!(&p.user_latents[7 * dim..8 * dim], &silence[..]);
        assert!(p.user_latents[..6 * dim].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn an_empty_prompt_is_one_pad_position_and_speak_first_without_a_voice_uses_silence() {
        let d = duplex();
        let silence = vec![2.0; d.latent_dim];
        let p = build(&d, &PromptParts { card: &[], reference: None, reference_latents: &[], silence: &silence, speak_first: false }).unwrap();
        assert_eq!((p.tokens.clone(), p.model_mask.clone(), p.user_mask.clone()), (vec![d.pad], vec![1.0], vec![1.0]));
        let q = build(&d, &PromptParts { card: &[5], reference: None, reference_latents: &[], silence: &silence, speak_first: true }).unwrap();
        assert_eq!(q.tokens, vec![5, d.pad, d.eop]);
        assert_eq!(&q.model_latents[d.latent_dim..2 * d.latent_dim], &silence[..]);
    }

    #[test]
    fn the_reference_is_trimmed_to_voiced_audio_with_a_margin_and_capped() {
        let d = duplex();
        let rate = d.sample_rate;
        let mut a = vec![0.0f32; 3 * rate];
        for v in &mut a[rate..2 * rate] {
            *v = 0.3;
        }
        let t = trim_reference(&d, &a);
        let margin = (d.trim_margin_seconds * rate as f64) as usize;
        assert_eq!(t.len(), rate + 2 * margin);
        assert!(trim_reference(&d, &vec![0.0; rate]).is_empty());
        let long = vec![0.3f32; (d.reference_seconds as usize + 5) * rate];
        assert_eq!(trim_reference(&d, &long).len(), d.reference_seconds as usize * rate);
    }

    #[test]
    fn card_fields_render_under_the_lead_with_headers_and_option_labels() {
        let d = duplex();
        let given: std::collections::BTreeMap<String, String> = [("name", "Haru"), ("backchannel", "mid"), ("situation", "A first chat.")].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let text = d.card.render(&given).unwrap();
        assert_eq!(text, "You are the person described below, talking with the other person by voice.\n[Your Name]\nHaru\n[Your Backchannels]\nmoderate\n[Situation]\nA first chat.");
        let bad: std::collections::BTreeMap<String, String> = [("mood".to_string(), "x".to_string())].into();
        assert!(d.card.render(&bad).unwrap_err().to_string().contains("mood"));
        let wrong: std::collections::BTreeMap<String, String> = [("gender".to_string(), "robot".to_string())].into();
        assert!(d.card.render(&wrong).is_err());
    }

    #[test]
    fn the_card_is_a_chatml_system_turn() {
        let d = duplex();
        assert_eq!(card_text(&d, "Be calm."), "<|im_start|>system\nBe calm.<|im_end|>\n");
        assert_eq!(card_text(&d, ""), "");
    }
}
