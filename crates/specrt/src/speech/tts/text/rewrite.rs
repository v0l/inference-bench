
use super::Lang;

mod abbreviations;
mod currency;
mod dashed_digits;
mod dates;
mod elongations;
mod emails;
mod numbers;
mod phones;
mod times;
mod tlds;
mod urls;

pub(super) use currency::is_symbol as is_currency_symbol;
type Rewrite = fn(&str, Lang) -> Option<String>;
struct Rule {
    name: &'static str,
    rewrite: Rewrite,
    default: bool,
}
const RULES: &[Rule] = &[
    Rule { name: "numbers", rewrite: numbers::numbers, default: true },
    Rule { name: "currency", rewrite: currency::currency, default: true },
    Rule { name: "dashed-digits", rewrite: dashed_digits::dashed_digits, default: true },
    Rule { name: "emails", rewrite: emails::emails, default: true },
    Rule { name: "urls", rewrite: urls::urls, default: true },
    Rule { name: "abbreviations", rewrite: abbreviations::abbreviations, default: true },
    Rule { name: "elongations", rewrite: elongations::elongations, default: true },
    Rule { name: "phones", rewrite: phones::phones, default: false },
    Rule { name: "times", rewrite: times::times, default: false },
    Rule { name: "dates", rewrite: dates::dates, default: false },
];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules(u32);

impl Rules {
    pub const NONE: Self = Self(0);
    pub const DEFAULT: Self = {
        let mut bits = 0;
        let mut i = 0;
        while i < RULES.len() {
            if RULES[i].default {
                bits |= 1 << i;
            }
            i += 1;
        }
        Self(bits)
    };
    pub const ALL: Self = Self((1 << RULES.len()) - 1);
}
impl std::str::FromStr for Rules {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_lowercase().as_str() {
            "default" => Ok(Self::DEFAULT),
            "all" => Ok(Self::ALL),
            "none" | "off" => Ok(Self::NONE),
            list => list.split(',').try_fold(Self::NONE, |set, name| {
                match RULES.iter().position(|rule| rule.name == name.trim()) {
                    Some(i) => Ok(Self(set.0 | 1 << i)),
                    None => Err(format!(
                        "unknown rewrite rule: {name}; expected default, all, none, or any of: {}",
                        RULES.iter().map(|rule| rule.name).collect::<Vec<_>>().join(", ")
                    )),
                }
            }),
        }
    }
}
pub fn rewrite_word(word: &str, lang: Lang, rules: Rules) -> Option<String> {
    RULES
        .iter()
        .enumerate()
        .filter(|(i, _)| rules.0 & (1 << i) != 0)
        .find_map(|(_, rule)| (rule.rewrite)(word, lang))
}

impl Rules {
    fn has(self, name: &str) -> bool {
        RULES.iter().position(|rule| rule.name == name).is_some_and(|i| self.0 & (1 << i) != 0)
    }
}

pub(super) fn rewrite_text(
    text: &str,
    lang: Lang,
    rules: Rules,
    unclaimed: impl Fn(&str) -> String,
) -> String {
    let words: Vec<&str> = text.split(' ').collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        let prev = out.last().map(String::as_str);
        let span = None
            .or_else(|| {
                rules.has("currency").then(|| currency::scaled(&words[i..], lang)).flatten()
            })
            .or_else(|| {
                rules.has("numbers").then(|| numbers::digit_run(&words[i..], lang)).flatten()
            })
            .or_else(|| {
                rules.has("numbers").then(|| numbers::ending_in(prev, &words[i..], lang)).flatten()
            })
            .or_else(|| {
                rules
                    .has("abbreviations")
                    .then(|| abbreviations::ranks(&words[i..], lang))
                    .flatten()
            });
        if let Some((rewritten, used)) = span {
            out.push(rewritten);
            i += used;
            continue;
        }
        out.push(rewrite_word(words[i], lang, rules).unwrap_or_else(|| unclaimed(words[i])));
        i += 1;
    }
    out.join(" ")
}
const PUNCTUATION: &str = "!?.:,;…";
fn suffix(rest: &str) -> Option<&str> {
    rest.chars().all(|c| PUNCTUATION.contains(c)).then_some(rest)
}
fn split_suffix(word: &str) -> (&str, &str) {
    let body = word.trim_end_matches(|c| PUNCTUATION.contains(c));
    (body, &word[body.len()..])
}
fn digits(s: &str, min: usize, max: usize) -> Option<(&str, &str)> {
    let n = s.bytes().take_while(u8::is_ascii_digit).count().min(max);
    (n >= min).then(|| s.split_at(n))
}
fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}
fn spell_short(word: &str) -> std::borrow::Cow<'_, str> {
    match word.as_bytes() {
        b"www" | b"WWW" => "W-W-W".into(),
        [a, b] if a.is_ascii() && b.is_ascii() => {
            format!("{}-{}", a.to_ascii_uppercase() as char, b.to_ascii_uppercase() as char).into()
        }
        _ => word.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_pick_what_runs() {
        assert_eq!("all".parse::<Rules>().unwrap(), Rules::ALL);
        assert_eq!("default".parse::<Rules>().unwrap(), Rules::DEFAULT);
        let default = "numbers,currency,dashed-digits,emails,urls,abbreviations,elongations"
            .parse::<Rules>()
            .unwrap();
        assert_eq!(default, Rules::DEFAULT);
        let opt_in = "phones,times,dates".parse::<Rules>().unwrap();
        assert_eq!(Rules(default.0 | opt_in.0), Rules::ALL);
        assert_eq!("none".parse::<Rules>().unwrap(), Rules::NONE);
        assert!("colours".parse::<Rules>().is_err());
        assert_eq!(rewrite_word("1234", Lang::En, Rules::ALL).as_deref(), Some("1 thousand 234"));
        assert_eq!(rewrite_word("1234", Lang::Fr, Rules::ALL).as_deref(), Some("mille 234"));
        assert_eq!(rewrite_word("1234", Lang::En, Rules::NONE), None);
        assert_eq!(rewrite_word("1234", Lang::De, Rules::ALL).as_deref(), Some("ein Tausend 234"));
        assert_eq!(rewrite_word("$5", Lang::En, Rules::DEFAULT).as_deref(), Some("5 dollars"));
        assert_eq!(rewrite_word("3:45PM", Lang::En, Rules::DEFAULT), None);
        assert_eq!(rewrite_word("3:45PM", Lang::En, Rules::ALL).as_deref(), Some("3-45 PM"));
    }
    #[test]
    fn numbers_win_over_times() {
        let all = Rules::ALL;
        assert_eq!(rewrite_word("1.234", Lang::De, all).as_deref(), Some("ein Tausend 234"));
        assert_eq!(rewrite_word("22.45", Lang::De, all).as_deref(), Some("22 Uhr 45"));
    }

    #[test]
    fn rules_parse_lists_spaces_and_case() {
        let numbers = "numbers".parse::<Rules>().unwrap();
        assert_eq!(" Numbers ".parse::<Rules>().unwrap(), numbers);
        assert_eq!("numbers, numbers".parse::<Rules>().unwrap(), numbers);
        assert_eq!(" Emails , URLS".parse::<Rules>().unwrap(), "emails,urls".parse().unwrap());
        assert_eq!("ALL".parse::<Rules>().unwrap(), Rules::ALL);
        assert_eq!(" Default ".parse::<Rules>().unwrap(), Rules::DEFAULT);
        assert_eq!("Off".parse::<Rules>().unwrap(), Rules::NONE);
        let err = "numbers,colours".parse::<Rules>().unwrap_err();
        assert!(!err.is_empty());
        let msg = err.to_string();
        assert!(msg.contains("colours") && msg.contains("dashed-digits"), "{msg}");
        assert!("".parse::<Rules>().is_err());
        assert!("numbers,all".parse::<Rules>().is_err());
    }

    #[test]
    fn helpers() {
        assert_eq!(split_suffix("12,..."), ("12", ",..."));
        assert_eq!(split_suffix("12"), ("12", ""));
        assert_eq!(split_suffix("..."), ("", "..."));
        assert_eq!(digits("12345x", 2, 4), Some(("1234", "5x")));
        assert_eq!(digits("1x", 2, 4), None);
        assert_eq!(spell_short("fr"), "F-R");
        assert_eq!(spell_short("www"), "W-W-W");
        assert_eq!(spell_short("com"), "com");
        assert_eq!(spell_short("é"), "é");
    }
}
