
use super::numbers::number_words;
use super::{all_digits, split_suffix};
use crate::speech::tts::text::Lang;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Currency {
    Dollar,
    Euro,
    Pound,
    Yen,
    Cent,
    Real,
}

impl Currency {
    const SYMBOLS: [char; 5] = ['$', '€', '£', '¥', '¢'];

    fn from_symbol(symbol: char) -> Option<Self> {
        match symbol {
            '$' => Some(Self::Dollar),
            '€' => Some(Self::Euro),
            '£' => Some(Self::Pound),
            '¥' => Some(Self::Yen),
            '¢' => Some(Self::Cent),
            _ => None,
        }
    }
    fn name(self, lang: Lang) -> (&'static str, &'static str) {
        match (self, lang) {
            (Self::Dollar, Lang::En | Lang::Fr) => ("dollar", "dollars"),
            (Self::Dollar, Lang::De) => ("Dollar", "Dollar"),
            (Self::Dollar, Lang::Es | Lang::Pt) => ("dólar", "dólares"),
            (Self::Euro, Lang::De) => ("Euro", "Euro"),
            (Self::Euro, _) => ("euro", "euros"),
            (Self::Pound, Lang::En) => ("pound", "pounds"),
            (Self::Pound, Lang::Fr) => ("livre", "livres"),
            (Self::Pound, Lang::De) => ("Pfund", "Pfund"),
            (Self::Pound, Lang::Es | Lang::Pt) => ("libra", "libras"),
            (Self::Yen, Lang::De) => ("Yen", "Yen"),
            (Self::Yen, Lang::En) => ("yen", "yen"),
            (Self::Yen, Lang::Fr) => ("yen", "yens"),
            (Self::Yen, Lang::Es) => ("yen", "yenes"),
            (Self::Yen, Lang::Pt) => ("iene", "ienes"),
            (Self::Cent, Lang::En | Lang::Fr) => ("cent", "cents"),
            (Self::Cent, Lang::De) => ("Cent", "Cent"),
            (Self::Cent, Lang::Es | Lang::Pt) => ("centavo", "centavos"),
            (Self::Real, _) => ("real", "reais"),
        }
    }
}
pub(in crate::speech::tts::text) fn is_symbol(c: char) -> bool {
    Currency::from_symbol(c).is_some()
}
pub(super) fn currency(word: &str, lang: Lang) -> Option<String> {
    let (body, suffix) = split_suffix(word);
    if let Some(body) = body.strip_prefix('-')
        && let Some(amount) = body
            .strip_prefix(Currency::SYMBOLS)
            .or_else(|| body.strip_prefix("R$").filter(|_| lang == Lang::Pt))
    {
        let symbol = &body[..body.len() - amount.len()];
        return currency(&format!("{symbol}-{amount}{suffix}"), lang);
    }
    let (currency, amount) = split_symbol(body, lang)?;
    if let Some((amount, scale)) = split_scale(amount, lang) {
        return with_scale(amount, scale, currency, lang).map(|spoken| format!("{spoken}{suffix}"));
    }
    if currency != Currency::Cent
        && let Some(spoken) = with_cents(amount, currency, lang)
    {
        return Some(format!("{spoken}{suffix}"));
    }
    let (spoken, _) = number_words(amount, lang)?;
    let (one, many) = currency.name(lang);
    let name = if currency == Currency::Cent && amount == "1" { one } else { many };
    Some(format!("{}{suffix}", with_name(&spoken, name, lang)))
}
fn split_scale(amount: &str, lang: Lang) -> Option<(&str, &'static str)> {
    let suffixes: &[(&str, &str)] = match lang {
        Lang::En => &[
            ("thousand", "thousand"),
            ("million", "million"),
            ("billion", "billion"),
            ("trillion", "trillion"),
            ("bn", "billion"),
            ("BN", "billion"),
            ("mn", "million"),
            ("tn", "trillion"),
            ("k", "thousand"),
            ("K", "thousand"),
            ("m", "million"),
            ("M", "million"),
            ("b", "billion"),
            ("B", "billion"),
        ],
        _ => &[],
    };
    suffixes.iter().find_map(|&(suffix, scale)| {
        let number = amount.strip_suffix(suffix)?;
        number.ends_with(|c: char| c.is_ascii_digit()).then_some((number, scale))
    })
}

fn scale_word(word: &str, lang: Lang) -> bool {
    let words: &[&str] = match lang {
        Lang::En => &["thousand", "million", "billion", "trillion"],
        Lang::Fr => &["mille", "million", "millions", "milliard", "milliards"],
        Lang::De => &["Tausend", "Million", "Millionen", "Milliarde", "Milliarden"],
        Lang::Es => &["mil", "millón", "millones"],
        Lang::Pt => &["mil", "milhão", "milhões", "bilhão", "bilhões"],
    };
    words.contains(&word)
}

fn with_scale(amount: &str, scale: &str, currency: Currency, lang: Lang) -> Option<String> {
    let (amount, rest) = number_words(amount, lang)?;
    if !rest.is_empty() {
        return None;
    }
    let (_, name) = currency.name(lang);
    Some(with_name(&format!("{amount} {scale}"), name, lang))
}

pub(super) fn scaled(words: &[&str], lang: Lang) -> Option<(String, usize)> {
    let [word, next, ..] = words else { return None };
    let (scale, suffix) = split_suffix(next);
    let (currency, amount) = split_symbol(word, lang)?;
    let scale = match (lang, scale, amount) {
        (Lang::Pt, "bi", "1") => "bilhão",
        (Lang::Pt, "bi", _) => "bilhões",
        (Lang::Pt, "mi", "1") => "milhão",
        (Lang::Pt, "mi", _) => "milhões",
        _ => scale,
    };
    if !scale_word(scale, lang) {
        return None;
    }
    let spoken = with_scale(amount, scale, currency, lang)?;
    Some((format!("{spoken}{suffix}"), 2))
}

fn split_symbol(word: &str, lang: Lang) -> Option<(Currency, &str)> {
    if lang == Lang::Pt
        && let Some(amount) = word.strip_prefix("R$")
    {
        return Some((Currency::Real, amount));
    }
    match word.strip_prefix(Currency::SYMBOLS) {
        Some(amount) => Some((Currency::from_symbol(word.chars().next()?)?, amount)),
        None => {
            let amount = word.strip_suffix(Currency::SYMBOLS)?;
            Some((Currency::from_symbol(word.chars().next_back()?)?, amount))
        }
    }
}

fn with_cents(amount: &str, currency: Currency, lang: Lang) -> Option<String> {
    let (whole, cents) = match lang {
        Lang::En => amount.rsplit_once('.').filter(|(whole, _)| {
            all_digits(whole)
                || whole.split(',').enumerate().all(|(i, group)| {
                    all_digits(group) && if i == 0 { group.len() <= 3 } else { group.len() == 3 }
                })
        })?,
        Lang::Fr => amount.rsplit_once(',').filter(|(whole, _)| all_digits(whole))?,
        Lang::De | Lang::Es | Lang::Pt => return None,
    };
    if cents.len() != 2 || !all_digits(cents) {
        return None;
    }
    let (whole, _) = number_words(whole, lang)?;
    let (one, many) = currency.name(lang);
    let units = with_name(&whole, if whole == "1" { one } else { many }, lang);
    if cents == "00" {
        return Some(units);
    }
    let cents = cents.strip_prefix('0').unwrap_or(cents);
    let (and, cent) = match (lang, currency) {
        (Lang::Fr, Currency::Euro) => ("et", "centimes"),
        (Lang::Fr, _) => ("et", "cents"),
        _ => ("and", "cents"),
    };
    if whole == "0" {
        return Some(format!("{cents} {cent}"));
    }
    Some(format!("{units} {and} {cents} {cent}"))
}
fn with_name(amount: &str, name: &str, lang: Lang) -> String {
    let nouns: &[&str] = match lang {
        Lang::Fr => &["million", "millions", "milliard", "milliards"],
        Lang::Es => &["millón", "millones"],
        Lang::Pt => &["milhão", "milhões", "bilhão", "bilhões"],
        Lang::En | Lang::De => &[],
    };
    if !nouns.iter().any(|noun| amount.ends_with(noun)) {
        format!("{amount} {name}")
    } else if lang == Lang::Fr && name.starts_with(['a', 'e', 'i', 'o', 'u', 'é']) {
        format!("{amount} d'{name}")
    } else {
        format!("{amount} de {name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amounts_are_read_with_their_currency() {
        let cases = [
            (Lang::En, "$1234", "1 thousand 234 dollars"),
            (Lang::En, "$1,234.56", "1 thousand 234 dollars and 56 cents"),
            (Lang::En, "$1,234.56...", "1 thousand 234 dollars and 56 cents..."),
            (Lang::En, "$4.59", "4 dollars and 59 cents"),
            (Lang::En, "$0.01", "1 cents"),
            (Lang::En, "$0.10", "10 cents"),
            (Lang::En, "$1.00", "1 dollar"),
            (Lang::En, "$1.10", "1 dollar and 10 cents"),
            (Lang::En, "$3.14", "3 dollars and 14 cents"),
            (Lang::En, "$0.00", "0 dollars"),
            (Lang::En, "$1", "1 dollars"),
            (Lang::En, "£1.50", "1 pound and 50 cents"),
            (Lang::Fr, "$4,59", "4 dollars et 59 cents"),
            (Lang::Fr, "$0,01", "1 cents"),
            (Lang::Fr, "$1,00", "1 dollar"),
            (Lang::Fr, "$1,10", "1 dollar et 10 cents"),
            (Lang::Fr, "$0,00", "0 dollars"),
            (Lang::Fr, "$100", "100 dollars"),
            (Lang::De, "$2500", "2 Tausend 500 Dollar"),
            (Lang::Es, "$50", "50 dólares"),
            (Lang::Pt, "$50!", "50 dólares!"),
            (Lang::En, "€500...", "500 euros..."),
            (Lang::En, "€0", "0 euros"),
            (Lang::En, "€0.5", "0 point 5 euros"),
            (Lang::Fr, "€1000", "mille euros"),
            (Lang::De, "€1000", "ein Tausend Euro"),
            (Lang::De, "$1000000", "eine Million Dollar"),
            (Lang::En, "£250", "250 pounds"),
            (Lang::Fr, "£250", "250 livres"),
            (Lang::En, "30¢", "30 cents"),
            (Lang::En, "30¢.", "30 cents."),
            (Lang::En, "1¢", "1 cent"),
            (Lang::En, "99¢!", "99 cents!"),
            (Lang::En, "1.50¢", "1 point 50 cents"),
            (Lang::Fr, "30¢", "30 cents"),
            (Lang::De, "30¢", "30 Cent"),
            (Lang::Es, "30¢", "30 centavos"),
            (Lang::Pt, "1¢", "1 centavo"),
            (Lang::De, "£250", "250 Pfund"),
            (Lang::Es, "£250", "250 libras"),
            (Lang::En, "1234$", "1 thousand 234 dollars"),
            (Lang::Fr, "4,59€", "4 euros et 59 centimes"),
            (Lang::Fr, "0,01€", "1 centimes"),
            (Lang::Fr, "1,00€", "1 euro"),
            (Lang::Fr, "0,00€", "0 euros"),
            (Lang::Fr, "500€!", "500 euros!"),
            (Lang::De, "100£", "100 Pfund"),
            (Lang::De, "1000000000€", "eine Milliarde Euro"),
            (Lang::De, "1,23$", "1 Komma 23 Dollar"),
            (Lang::De, "1,23$!!!", "1 Komma 23 Dollar!!!"),
            (Lang::En, "$-500", "minus 500 dollars"),
            (Lang::En, "-$500.", "minus 500 dollars."),
            (Lang::De, "-€1.000", "minus ein Tausend Euro"),
            (Lang::Fr, "-0,50€", "moins 0 virgule 50 euros"),
            (Lang::Fr, "€2000000", "2 millions d'euros"),
            (Lang::Fr, "$2000000", "2 millions de dollars"),
            (Lang::Fr, "£3000000000.", "3 milliards de livres."),
            (Lang::Fr, "€2000000,00", "2 millions d'euros"),
            (Lang::Fr, "2000000,50€", "2 millions d'euros et 50 centimes"),
            (Lang::Fr, "€2500000", "2 millions 500 mille euros"),
            (Lang::Es, "€2000000", "2 millones de euros"),
            (Lang::Pt, "$2000000", "2 milhões de dólares"),
            (Lang::En, "$2000000", "2 million dollars"),
            (Lang::De, "$2000000", "2 Millionen Dollar"),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(currency(input, lang).as_deref(), Some(expected), "{lang:?} {input:?}");
        }
        for input in ["1234", "$abc", "$", "$5€", "5$5", "$007", "$1.2.3", "$ 5", "-$", "--$5"] {
            assert_eq!(currency(input, Lang::En), None, "{input:?}");
        }
    }
}
