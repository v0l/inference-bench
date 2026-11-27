
use super::split_suffix;
use crate::speech::tts::text::Lang;
pub(super) fn dashed_digits(word: &str, lang: Lang) -> Option<String> {
    let (body, suffix) = split_suffix(word);
    if !body.chars().all(|c| c.is_ascii_digit() || is_separator(c))
        || body.matches(is_separator).count() < 2
    {
        return None;
    }
    let groups = groups(body)?;
    if !body.contains('_') && looks_like_date(&groups) {
        return None;
    }
    let mut words = vec![];
    for (group, separator) in groups {
        if group.len() >= 4 {
            words.extend((0..group.len()).map(|i| &group[i..=i]));
        } else {
            words.push(group);
        }
        if separator == Some('_') {
            words.push(lang.special_chars().underscore);
        }
    }
    Some(format!("{}{suffix}", words.join(" ")))
}

fn is_separator(c: char) -> bool {
    c == '-' || c == '_'
}
fn groups(body: &str) -> Option<Vec<(&str, Option<char>)>> {
    let mut groups = vec![];
    let mut start = 0;
    for (i, c) in body.char_indices().filter(|&(_, c)| is_separator(c)) {
        if i == start {
            return None;
        }
        groups.push((&body[start..i], Some(c)));
        start = i + 1;
    }
    if start == body.len() {
        return None;
    }
    groups.push((&body[start..], None));
    Some(groups)
}
fn looks_like_date(groups: &[(&str, Option<char>)]) -> bool {
    fn day_month(x: &str, y: &str) -> bool {
        if x.len() > 2 || y.len() > 2 {
            return false;
        }
        match (x.parse::<u32>(), y.parse::<u32>()) {
            (Ok(x), Ok(y)) => {
                ((1..=31).contains(&x) && (1..=12).contains(&y))
                    || ((1..=12).contains(&x) && (1..=31).contains(&y))
            }
            _ => false,
        }
    }
    match groups {
        [(a, _), (b, _), (c, _)] => {
            (a.len() == 4 && day_month(b, c)) || ((c.len() == 4 || c.len() == 2) && day_month(a, b))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digit_groups_are_read_out() {
        let r = |s: &str| dashed_digits(s, Lang::En);
        let cases = [
            ("555-123-4567", "555 123 4 5 6 7"),
            ("1-800-555-1234", "1 800 555 1 2 3 4"),
            ("123-45-6789", "123 45 6 7 8 9"),
            ("978-3-16-148410-0", "978 3 16 1 4 8 4 1 0 0"),
            ("12-345-6789-01234", "12 345 6 7 8 9 0 1 2 3 4"),
            ("4111-1111-1111-1111", "4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"),
            ("1-2-3", "1 2 3"),
            ("0-1-2-3-4", "0 1 2 3 4"),
            ("90-60-90", "90 60 90"),
            ("205-55-16", "205 55 16"),
            ("555_123_4567", "555 underscore 123 underscore 4 5 6 7"),
            ("12-34_5678", "12 34 underscore 5 6 7 8"),
            ("1_2_3", "1 underscore 2 underscore 3"),
            ("1_000_000", "1 underscore 000 underscore 000"),
            ("2024_05_12", "2 0 2 4 underscore 05 underscore 12"),
            ("2024-05_12", "2 0 2 4 05 underscore 12"),
            ("2024-13-45", "2 0 2 4 13 45"),
            ("2024-05-123", "2 0 2 4 05 123"),
            ("123-05-2024", "123 05 2 0 2 4"),
            ("99-12-31", "99 12 31"),
            ("2024-05-12-1", "2 0 2 4 05 12 1"),
            ("800-1234-800", "800 1 2 3 4 800"),
            ("555-123-4567.", "555 123 4 5 6 7."),
            ("555-123-4567?!", "555 123 4 5 6 7?!"),
        ];
        for (input, expected) in cases {
            assert_eq!(r(input).as_deref(), Some(expected), "{input:?}");
        }
        let declined = [
            "2024-05-12",
            "2024-5-1",
            "12-05-2024",
            "31-12-1999",
            "12-05-24",
            "2024-05-12.",
            "1-2-30",
            "12-12-12",
            "10-15",
            "2024-2025",
            "1234-5678",
            "1_2",
            "-5",
            "12345678",
            "+1-555-123-4567",
            "555-123-4567-A",
            "555.123.4567",
            "-555-123-4567",
            "555-123-4567-",
            "555--123-4567",
            "555-_123-4567",
            "",
            "--",
        ];
        for input in declined {
            assert_eq!(r(input), None, "{input:?}");
        }
        assert_eq!(
            dashed_digits("555_123_4567", Lang::Es).as_deref(),
            Some("555 guion bajo 123 guion bajo 4 5 6 7")
        );
    }
}
