use specrt::chat::Thinking;

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Split {
    pub reasoning: Option<String>,
    pub answer: String,
    pub answering: bool,
}

pub fn opens_in_prompt(thinking: Thinking, enabled: bool) -> bool {
    enabled && thinking == Thinking::OpensInPrompt
}

pub fn split(text: &str, opens_in_prompt: bool, last: bool) -> Split {
    let lead = text.trim_start();
    let (reasoning_text, started) = match lead.strip_prefix(OPEN) {
        Some(rest) => (rest, true),
        None => (text, opens_in_prompt),
    };
    if !started {
        return Split { reasoning: None, answer: text.to_string(), answering: !text.is_empty() };
    }
    match reasoning_text.find(CLOSE) {
        Some(end) => {
            let thought = reasoning_text[..end].trim();
            let answer = reasoning_text[end + CLOSE.len()..].trim_start();
            Split { reasoning: (!thought.is_empty()).then(|| thought.to_string()), answer: answer.to_string(), answering: true }
        }
        None => {
            let held = if last { 0 } else { partial_close(reasoning_text) };
            let thought = reasoning_text[..reasoning_text.len() - held].trim();
            Split { reasoning: (!thought.is_empty()).then(|| thought.to_string()), answer: String::new(), answering: false }
        }
    }
}

fn partial_close(text: &str) -> usize {
    (1..CLOSE.len()).rev().find(|&n| text.ends_with(&CLOSE[..n])).unwrap_or(0)
}

pub fn canonical(text: &str, opens_in_prompt: bool, last: bool) -> String {
    let s = split(text, opens_in_prompt, last);
    match (s.reasoning, s.answering) {
        (Some(r), true) => format!("{OPEN}\n{r}\n{CLOSE}\n\n{}", s.answer),
        (Some(r), false) => format!("{OPEN}\n{r}"),
        (None, _) => s.answer,
    }
}

pub fn answer_of(text: &str) -> String {
    split(text, false, true).answer
}

pub fn reasoning_of(text: &str) -> Option<String> {
    split(text, false, true).reasoning
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_think_block_in_the_text_splits_from_the_answer() {
        let s = split("<think>a</think>b", false, true);
        assert_eq!((s.reasoning.as_deref(), s.answer.as_str(), s.answering), (Some("a"), "b", true));
    }

    #[test]
    fn a_template_that_opens_the_block_starts_in_reasoning() {
        let s = split("a</think>\n\nb", true, true);
        assert_eq!((s.reasoning.as_deref(), s.answer.as_str()), (Some("a"), "b"));
        let s = split("still thinking", true, false);
        assert_eq!((s.reasoning.as_deref(), s.answer.as_str(), s.answering), (Some("still thinking"), "", false));
    }

    #[test]
    fn text_without_a_think_block_is_all_answer() {
        let s = split("plain", false, true);
        assert_eq!((s.reasoning, s.answer.as_str()), (None, "plain"));
    }

    #[test]
    fn a_half_streamed_closing_tag_is_not_shown_as_reasoning() {
        assert_eq!(split("a</thi", true, false).reasoning.as_deref(), Some("a"));
        assert_eq!(split("a</thi", true, true).reasoning.as_deref(), Some("a</thi"));
    }

    #[test]
    fn the_canonical_form_keeps_the_block_open_until_the_answer_starts() {
        assert_eq!(canonical("a", true, false), "<think>\na");
        assert_eq!(canonical("a</think>b", true, false), "<think>\na\n</think>\n\nb");
        assert_eq!(canonical("", true, false), "");
        assert_eq!(canonical("plain", true, true), "plain");
        assert_eq!(answer_of(&canonical("a</think>b", true, true)), "b");
        assert_eq!(reasoning_of(&canonical("a</think>b", true, true)).as_deref(), Some("a"));
    }
}
