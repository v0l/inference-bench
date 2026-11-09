pub struct StopFilter {
    stops: Vec<String>,
    held: String,
    pub text: String,
    pub hit: bool,
}

impl StopFilter {
    pub fn new(stops: Vec<String>) -> Self {
        Self { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), held: String::new(), text: String::new(), hit: false }
    }

    pub fn feed(&mut self, piece: &str) -> String {
        if self.hit {
            return String::new();
        }
        self.held.push_str(piece);
        if let Some(at) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
            self.hit = true;
            self.held.truncate(at);
            return self.release(at);
        }
        let keep = self
            .stops
            .iter()
            .flat_map(|s| s.char_indices().skip(1).map(|(i, _)| &s[..i]))
            .filter(|prefix| self.held.ends_with(prefix))
            .map(str::len)
            .max()
            .unwrap_or(0);
        self.release(self.held.len() - keep)
    }

    pub fn finish(&mut self) -> String {
        let n = self.held.len();
        self.release(n)
    }

    fn release(&mut self, n: usize) -> String {
        let out: String = self.held.drain(..n).collect();
        self.text.push_str(&out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], pieces: &[&str]) -> (Vec<String>, String, bool) {
        let mut f = StopFilter::new(stops.iter().map(|s| s.to_string()).collect());
        let mut out: Vec<String> = pieces.iter().map(|p| f.feed(p)).collect();
        out.push(f.finish());
        (out, f.text.clone(), f.hit)
    }

    #[test]
    fn a_stop_string_split_across_chunks_is_held_back_and_cut() {
        let (out, text, hit) = run(&["STOP"], &["hello S", "TO", "P world"]);
        assert_eq!(out, ["hello ", "", "", ""]);
        assert_eq!((text.as_str(), hit), ("hello ", true));
    }

    #[test]
    fn a_partial_match_that_fails_is_released() {
        let (out, text, hit) = run(&["STOP"], &["a ST", "ar b"]);
        assert_eq!(out, ["a ", "STar b", ""]);
        assert_eq!((text.as_str(), hit), ("a STar b", false));
    }

    #[test]
    fn the_earliest_of_several_stops_wins() {
        let (_, text, hit) = run(&["\n\n", "END"], &["one END two\n\nthree"]);
        assert_eq!((text.as_str(), hit), ("one ", true));
    }

    #[test]
    fn a_held_tail_is_flushed_at_the_end() {
        let (out, text, _) = run(&["STOP"], &["abc ST"]);
        assert_eq!(out, ["abc ", "ST"]);
        assert_eq!(text, "abc ST");
    }

    #[test]
    fn multibyte_text_is_not_split_inside_a_character() {
        let (_, text, hit) = run(&["ünï"], &["xü", "nï y"]);
        assert_eq!((text.as_str(), hit), ("x", true));
    }
}
