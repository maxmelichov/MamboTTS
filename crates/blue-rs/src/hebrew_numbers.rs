//! Hebrew number normalization, run before grapheme-to-phoneme conversion.
//!
//! Hebrew spells digits out as words, and which words depend on the context.
//! A year, a counted noun, an ordinal in a date and a percentage all take
//! different forms, and cardinals additionally agree in gender with the noun
//! they count. The generic pass in `handling.rs` read every run of digits as a
//! bare cardinal and gave up above 9999, so `2011` came out as three separate
//! numbers, "shtayim elef ahat esre", instead of the single year "alpayim
//! ve-ahat esre" (issue #8).
//!
//! ## Other languages
//!
//! This module is Hebrew only, on purpose. English, Spanish, German and
//! Italian go through the same `expand_numbers` helper, which knows words for
//! 0 to 20 only and spells everything above that digit by digit, so `2011`
//! becomes "two zero one one" there as well. That is the same gap, but it is
//! not the same fix: those four languages are handed to eSpeak, which already
//! reads digits correctly in every one of them, so the repair there is to stop
//! expanding numbers before eSpeak sees them rather than to port this module.
//! That is a separate change and is deliberately left alone here.

use regex::{Captures, Regex};

/// Grammatical gender of the noun a cardinal agrees with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gender {
    Masculine,
    Feminine,
}

const UNITS_FEMININE: [&str; 10] = [
    "אפס", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע",
];
const UNITS_MASCULINE: [&str; 10] = [
    "אפס", "אחד", "שניים", "שלושה", "ארבעה", "חמישה", "שישה", "שבעה", "שמונה", "תשעה",
];
const TEENS_FEMININE: [&str; 10] = [
    "עשר",
    "אחת עשרה",
    "שתים עשרה",
    "שלוש עשרה",
    "ארבע עשרה",
    "חמש עשרה",
    "שש עשרה",
    "שבע עשרה",
    "שמונה עשרה",
    "תשע עשרה",
];
const TEENS_MASCULINE: [&str; 10] = [
    "עשרה",
    "אחד עשר",
    "שנים עשר",
    "שלושה עשר",
    "ארבעה עשר",
    "חמישה עשר",
    "שישה עשר",
    "שבעה עשר",
    "שמונה עשר",
    "תשעה עשר",
];
const TENS: [&str; 10] = [
    "", "", "עשרים", "שלושים", "ארבעים", "חמישים", "שישים", "שבעים", "שמונים", "תשעים",
];
const HUNDREDS: [&str; 10] = [
    "",
    "מאה",
    "מאתיים",
    "שלוש מאות",
    "ארבע מאות",
    "חמש מאות",
    "שש מאות",
    "שבע מאות",
    "שמונה מאות",
    "תשע מאות",
];
/// Construct-state thousands. These forms are fixed and do not vary by gender.
const THOUSANDS: [&str; 11] = [
    "",
    "אלף",
    "אלפיים",
    "שלושת אלפים",
    "ארבעת אלפים",
    "חמשת אלפים",
    "ששת אלפים",
    "שבעת אלפים",
    "שמונת אלפים",
    "תשעת אלפים",
    "עשרת אלפים",
];
/// Definite masculine ordinals, used for days 1 to 10 of a month.
const ORDINALS_MASCULINE: [&str; 11] = [
    "",
    "הראשון",
    "השני",
    "השלישי",
    "הרביעי",
    "החמישי",
    "השישי",
    "השביעי",
    "השמיני",
    "התשיעי",
    "העשירי",
];
/// Definite feminine ordinals, for the cases where the counted noun is feminine.
const ORDINALS_FEMININE: [&str; 11] = [
    "",
    "הראשונה",
    "השנייה",
    "השלישית",
    "הרביעית",
    "החמישית",
    "השישית",
    "השביעית",
    "השמינית",
    "התשיעית",
    "העשירית",
];

const GREGORIAN_MONTHS: [&str; 13] = [
    "ינואר",
    "פברואר",
    "מרץ",
    "מרס",
    "אפריל",
    "מאי",
    "יוני",
    "יולי",
    "אוגוסט",
    "ספטמבר",
    "אוקטובר",
    "נובמבר",
    "דצמבר",
];

/// Nouns whose ending does not give their gender away, or gives it away wrongly.
const FEMININE_NOUNS: [&str; 26] = [
    "שנים",
    "שנה",
    "נשים",
    "אישה",
    "מילים",
    "מילה",
    "ערים",
    "עיר",
    "פעמים",
    "פעם",
    "ביצים",
    "דרכים",
    "דרך",
    "אבנים",
    "אבן",
    "עיניים",
    "ידיים",
    "רגליים",
    "שיניים",
    "כוסות",
    "דקות",
    "דקה",
    "שעות",
    "שעה",
    "נפש",
    "כיתה",
];
const MASCULINE_NOUNS: [&str; 30] = [
    "דונם",
    "שקל",
    "שקלים",
    "דולר",
    "דולרים",
    "אירו",
    "אחוז",
    "אחוזים",
    "אלף",
    "אלפים",
    "מיליון",
    "מיליארד",
    "טריליון",
    "אדם",
    "בני",
    "בן",
    "איש",
    "אנשים",
    "צמחים",
    "צמח",
    "צופים",
    "צופה",
    "מקומות",
    "שולחנות",
    "לילות",
    "קירות",
    "חלונות",
    "רחובות",
    "קולות",
    "שמות",
];
/// Words that can follow a number without being the noun it counts.
const NON_NOUNS: [&str; 22] = [
    "עד", "של", "את", "על", "אל", "עם", "או", "גם", "כי", "לא", "אך", "אבל", "רק", "כבר", "יותר",
    "פחות", "בערך", "כמו", "מתוך", "בין", "לבין", "ועד",
];

/// Hebrew one-letter prefixes that can be glued to a number with a maqaf.
const PREFIX_LETTERS: [char; 7] = ['ב', 'כ', 'ל', 'מ', 'ה', 'ש', 'ו'];

/// Expand every number in Hebrew text into spoken Hebrew words.
///
/// Runs after the date, time and code passes in `handling.rs`, so what reaches
/// it is plain running text. Spans already tagged for another language are left
/// alone, because eSpeak reads those, not the Hebrew G2P.
pub fn normalize_hebrew_numbers(text: &str) -> String {
    let tagged = Regex::new(r"(?is)<(en|en-us|he|es|de|ge|it)>.*?</(?:en|en-us|he|es|de|ge|it)>")
        .expect("valid regex");
    let mut output = String::with_capacity(text.len());
    let mut last = 0;
    for span in tagged.find_iter(text) {
        output.push_str(&normalize_plain_span(&text[last..span.start()]));
        output.push_str(span.as_str());
        last = span.end();
    }
    output.push_str(&normalize_plain_span(&text[last..]));
    output
}

fn normalize_plain_span(text: &str) -> String {
    if !text.chars().any(|c| c.is_ascii_digit()) {
        return text.to_owned();
    }
    let text = expand_currency_symbols(text);
    let text = expand_percent(&text);
    let text = expand_ratios(&text);
    let text = expand_ranges(&text);
    let text = expand_dates(&text);
    expand_cardinals(&text)
}

/// `12%` becomes `12 אחוז`, and the cardinal pass then agrees with `אחוז`.
fn expand_percent(text: &str) -> String {
    Regex::new(r"(?u)([\d,]*\d(?:\.\d+)?)\s*%")
        .expect("valid regex")
        .replace_all(text, "$1 אחוז")
        .into_owned()
}

fn expand_currency_symbols(text: &str) -> String {
    let after = Regex::new(r"(?u)([\d,]*\d(?:\.\d+)?)\s*(₪|\$|€)").expect("valid regex");
    let text = after
        .replace_all(text, |caps: &Captures| {
            format!("{} {}", &caps[1], currency_word(&caps[2]))
        })
        .into_owned();
    let before = Regex::new(r"(?u)(₪|\$|€)\s*([\d,]*\d(?:\.\d+)?)").expect("valid regex");
    before
        .replace_all(&text, |caps: &Captures| {
            format!("{} {}", &caps[2], currency_word(&caps[1]))
        })
        .into_owned()
}

fn currency_word(symbol: &str) -> &'static str {
    match symbol {
        "₪" => "שקלים",
        "€" => "אירו",
        _ => "דולר",
    }
}

/// `3:1` is a ratio, read "three to one". Clock times were consumed upstream.
fn expand_ratios(text: &str) -> String {
    Regex::new(r"(\d+)\s*:\s*(\d+)")
        .expect("valid regex")
        .replace_all(text, "$1 ל-$2")
        .into_owned()
}

/// `2022 - 2025` is a range, not a subtraction, so the hyphen becomes "until".
fn expand_ranges(text: &str) -> String {
    Regex::new(
        r"(?u)([\d,]*\d(?:\.\d+)?)\s*[-\x{2013}\x{2014}\x{2011}\x{05be}]\s*([\d,]*\d(?:\.\d+)?)",
    )
    .expect("valid regex")
    .replace_all(text, "$1 עד $2")
    .into_owned()
}

/// `3 בספטמבר` is an ordinal date, not a cardinal.
///
/// Days 1 to 10 take the definite ordinal, "the third of September", which is
/// the form the issue asks for. From 11 up Hebrew switches to the plain
/// masculine cardinal, because "the fifteenth of September" is not idiomatic.
fn expand_dates(text: &str) -> String {
    let months = GREGORIAN_MONTHS.join("|");
    let date = Regex::new(&format!(
        r"(?u)(?:ה\s*[-\x{{2013}}\x{{2014}}\x{{2011}}\x{{05be}}]\s*)?(\d{{1,2}})\s+ב({months})"
    ))
    .expect("valid regex");
    date.replace_all(text, |caps: &Captures| {
        let day: usize = caps[1].parse().unwrap_or_default();
        let month = &caps[2];
        if !(1..=31).contains(&day) {
            return caps[0].to_owned();
        }
        if day <= 10 {
            format!("{} ב{month}", ORDINALS_MASCULINE[day])
        } else {
            format!("{} ב{month}", cardinal(day as u64, Gender::Masculine))
        }
    })
    .into_owned()
}

fn expand_cardinals(text: &str) -> String {
    let number = Regex::new(
        r"(?xu)
        (?:
            (?P<bound>^|[\s,.;:!?()\[\]\x{201c}\x{201d}])
            (?P<pre>[\x{05d1}\x{05db}\x{05dc}\x{05de}\x{05d4}\x{05e9}\x{05d5}])
            \s*[-\x{2013}\x{2014}\x{2011}\x{05be}]\s*
        )?
        (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
        (?P<tail>[\x{20}\t]+(?P<word>[\x{05d0}-\x{05ea}]+))?
        ",
    )
    .expect("valid regex");
    number
        .replace_all(text, |caps: &Captures| {
            let bound = caps.name("bound").map_or("", |m| m.as_str());
            let prefix = caps.name("pre").map_or("", |m| m.as_str());
            let raw = &caps["num"];
            let tail = caps.name("tail").map_or("", |m| m.as_str());
            let noun = caps.name("word").map(|m| m.as_str()).filter(|w| is_noun(w));
            let gender = noun.map_or(Gender::Feminine, noun_gender);

            // `ה-3` is "the third", and the definite ordinal already carries
            // that ה, so the prefix is not glued on a second time.
            if prefix == "ה" && !raw.contains(['.', ',']) {
                if let Ok(day) = raw.parse::<usize>() {
                    if (1..=10).contains(&day) {
                        let ordinal = match gender {
                            Gender::Masculine => ORDINALS_MASCULINE[day],
                            Gender::Feminine => ORDINALS_FEMININE[day],
                        };
                        return format!("{bound}{ordinal}{tail}");
                    }
                }
            }

            // Hebrew puts a bare "one" after the noun it counts.
            if raw == "1" && prefix.is_empty() {
                if let Some(noun) = noun {
                    let one = match gender {
                        Gender::Masculine => "אחד",
                        Gender::Feminine => "אחת",
                    };
                    return format!("{bound}{noun} {one}");
                }
            }

            let spoken = spoken_number(raw, noun, gender);
            format!("{bound}{prefix}{spoken}{tail}")
        })
        .into_owned()
}

/// Render one written number, decimal or integer, as Hebrew words.
fn spoken_number(raw: &str, noun: Option<&str>, gender: Gender) -> String {
    let digits = raw.replace(',', "");
    if let Some((whole, fraction)) = digits.split_once('.') {
        // Decimals are read in the counting (feminine) form whatever they
        // measure: "shesh nekuda arba miliard shkalim".
        let whole = whole
            .parse::<u64>()
            .map_or_else(|_| spell_digits(whole), |n| cardinal(n, Gender::Feminine));
        return format!("{whole} נקודה {}", spell_digits(fraction));
    }
    let Ok(value) = digits.parse::<u64>() else {
        return spell_digits(&digits);
    };
    if value == 2 && noun.is_some() {
        // Two takes its construct form directly before the noun it counts.
        return match gender {
            Gender::Masculine => "שני".to_owned(),
            Gender::Feminine => "שתי".to_owned(),
        };
    }
    cardinal(value, gender)
}

fn spell_digits(digits: &str) -> String {
    digits
        .chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| UNITS_FEMININE[d as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

/// A cardinal in words, with the single "and" Hebrew puts before the last part.
pub fn cardinal(value: u64, gender: Gender) -> String {
    let parts = cardinal_parts(value, gender);
    match parts.split_last() {
        None => String::new(),
        Some((last, [])) => last.clone(),
        Some((last, head)) => format!("{} ו{last}", head.join(" ")),
    }
}

fn cardinal_parts(value: u64, gender: Gender) -> Vec<String> {
    if value == 0 {
        return vec!["אפס".to_owned()];
    }
    if value >= 1_000_000_000_000 {
        return vec![spell_digits(&value.to_string())];
    }
    let mut parts = Vec::new();
    let mut rest = value;
    for (scale, singular, dual) in [
        (1_000_000_000u64, "מיליארד", "שני מיליארד"),
        (1_000_000, "מיליון", "שני מיליון"),
    ] {
        let count = rest / scale;
        rest %= scale;
        if count > 0 {
            parts.extend(scaled_parts(count, singular, dual));
        }
    }
    let thousands = rest / 1_000;
    rest %= 1_000;
    if thousands > 0 {
        if thousands <= 10 {
            parts.push(THOUSANDS[thousands as usize].to_owned());
        } else {
            parts.extend(scaled_parts(thousands, "אלף", "אלפיים"));
        }
    }
    if rest > 0 {
        parts.extend(under_thousand_parts(rest, gender));
    }
    parts
}

/// The count of a scale word is itself masculine: "twenty one thousand".
fn scaled_parts(count: u64, singular: &str, dual: &str) -> Vec<String> {
    match count {
        1 => vec![singular.to_owned()],
        2 => vec![dual.to_owned()],
        _ => {
            let mut parts = cardinal_parts(count, Gender::Masculine);
            if let Some(last) = parts.last_mut() {
                *last = format!("{last} {singular}");
            }
            parts
        }
    }
}

fn under_thousand_parts(value: u64, gender: Gender) -> Vec<String> {
    let mut parts = Vec::new();
    let hundreds = (value / 100) as usize;
    let rest = (value % 100) as usize;
    if hundreds > 0 {
        parts.push(HUNDREDS[hundreds].to_owned());
    }
    if rest == 0 {
        return parts;
    }
    if rest < 10 {
        parts.push(unit_word(rest, gender).to_owned());
    } else if rest < 20 {
        parts.push(teen_word(rest - 10, gender).to_owned());
    } else {
        parts.push(TENS[rest / 10].to_owned());
        if rest % 10 > 0 {
            parts.push(unit_word(rest % 10, gender).to_owned());
        }
    }
    parts
}

fn unit_word(digit: usize, gender: Gender) -> &'static str {
    match gender {
        Gender::Masculine => UNITS_MASCULINE[digit],
        Gender::Feminine => UNITS_FEMININE[digit],
    }
}

fn teen_word(digit: usize, gender: Gender) -> &'static str {
    match gender {
        Gender::Masculine => TEENS_MASCULINE[digit],
        Gender::Feminine => TEENS_FEMININE[digit],
    }
}

fn is_noun(word: &str) -> bool {
    let bare = strip_prefix_letter(word);
    !NON_NOUNS.contains(&word)
        && !NON_NOUNS.contains(&bare)
        && !GREGORIAN_MONTHS.contains(&word)
        && !GREGORIAN_MONTHS.contains(&bare)
        && word.chars().count() > 1
}

/// Best guess at the gender of the noun a number counts.
///
/// A short lexicon covers the words whose ending lies, and the endings decide
/// the rest: `ות` is feminine, `ים` is masculine, and a lone `ה` or `ת` on a
/// singular is feminine. Whatever is left is read as masculine, the commoner
/// case in running text.
pub fn noun_gender(word: &str) -> Gender {
    let bare = strip_prefix_letter(word);
    for candidate in [word, bare] {
        if FEMININE_NOUNS.contains(&candidate) {
            return Gender::Feminine;
        }
        if MASCULINE_NOUNS.contains(&candidate) {
            return Gender::Masculine;
        }
    }
    if bare.ends_with("ות") {
        return Gender::Feminine;
    }
    if bare.ends_with("ים") || bare.ends_with("ין") {
        return Gender::Masculine;
    }
    if bare.ends_with('ה') || bare.ends_with('ת') {
        return Gender::Feminine;
    }
    Gender::Masculine
}

/// Drop one attached Hebrew prefix letter when a real word is left behind.
fn strip_prefix_letter(word: &str) -> &str {
    let mut chars = word.chars();
    let Some(first) = chars.next() else {
        return word;
    };
    let rest = chars.as_str();
    if PREFIX_LETTERS.contains(&first) && rest.chars().count() >= 3 {
        rest
    } else {
        word
    }
}

/// Read a Hebrew calendar year written in letters as the word it is.
///
/// `תשפ"ז` is neither an abbreviation to spell out letter by letter nor the
/// number five thousand seven hundred and eighty seven. It is said as a single
/// word, so the gershayim is dropped and the letters are left joined for the
/// Hebrew G2P to vocalize. This runs before the punctuation pass, while the
/// gershayim is still present to recognize the form by, and only fires when
/// the letters add up to a plausible year of the current millennium.
pub fn normalize_gematria_years(text: &str) -> String {
    let token = Regex::new(
        r#"(?u)(?:ה['\x{05f3}])?[\x{05d0}-\x{05ea}]{1,7}["\x{05f4}][\x{05d0}-\x{05ea}]"#,
    )
    .expect("valid regex");
    token
        .replace_all(text, |caps: &Captures| {
            let raw = &caps[0];
            let letters: String = raw
                .chars()
                .filter(|c| ('\u{05d0}'..='\u{05ea}').contains(c))
                .collect();
            let millennium_marked = raw.starts_with('ה')
                && raw
                    .chars()
                    .nth(1)
                    .is_some_and(|c| c == '\'' || c == '\u{05f3}');
            // ה'תשפ"ז spells the millennium out; the year is the rest.
            let body: String = if millennium_marked {
                letters.chars().skip(1).collect()
            } else {
                letters
            };
            if is_year_value(gematria(&body)) {
                return body;
            }
            // A grammatical prefix such as ב or ל belongs to the word but not
            // to the year, so it is peeled off before the value is checked.
            if let Some(first) = body.chars().next() {
                if PREFIX_LETTERS.contains(&first) {
                    let rest: String = body.chars().skip(1).collect();
                    if is_year_value(gematria(&rest)) {
                        return body;
                    }
                }
            }
            raw.to_owned()
        })
        .into_owned()
}

/// Years of the current Hebrew millennium, written without the leading ה,
/// land between 300 and 999. Anything outside that is an acronym, not a year.
fn is_year_value(value: u32) -> bool {
    (300..=999).contains(&value)
}

fn gematria(letters: &str) -> u32 {
    letters
        .chars()
        .map(|letter| match letter {
            'א'..='ט' => letter as u32 - 'א' as u32 + 1,
            'י' => 10,
            'כ' | 'ך' => 20,
            'ל' => 30,
            'מ' | 'ם' => 40,
            'נ' | 'ן' => 50,
            'ס' => 60,
            'ע' => 70,
            'פ' | 'ף' => 80,
            'צ' | 'ץ' => 90,
            'ק' => 100,
            'ר' => 200,
            'ש' => 300,
            'ת' => 400,
            _ => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two passes as `prepare_text_for_synthesis` runs them.
    fn normalized(text: &str) -> String {
        normalize_hebrew_numbers(&normalize_gematria_years(text))
    }

    #[test]
    fn reads_two_thousands_years_as_one_number() {
        // The reported failure: 2011 was read as two, thousand, eleven.
        assert_eq!(cardinal(2011, Gender::Feminine), "אלפיים ואחת עשרה");
        assert_eq!(cardinal(2026, Gender::Feminine), "אלפיים עשרים ושש");
        assert_eq!(cardinal(2000, Gender::Feminine), "אלפיים");
        assert_eq!(
            cardinal(1948, Gender::Feminine),
            "אלף תשע מאות ארבעים ושמונה"
        );
    }

    #[test]
    fn masthead_line_from_the_issue() {
        assert_eq!(
            normalized("זמן קיבוץ, גיליון 2011, 3 בספטמבר 2026"),
            "זמן קיבוץ, גיליון אלפיים ואחת עשרה, השלישי בספטמבר אלפיים עשרים ושש"
        );
    }

    #[test]
    fn cardinals_agree_with_the_noun_that_follows() {
        assert_eq!(normalized("4500 דונם"), "ארבעת אלפים וחמש מאות דונם");
        assert_eq!(
            normalized("346 אלף צופים"),
            "שלוש מאות ארבעים ושישה אלף צופים"
        );
        assert_eq!(
            normalized("7,400 מיני צמחים"),
            "שבעת אלפים וארבע מאות מיני צמחים"
        );
        assert_eq!(normalized("30,000 שקלים"), "שלושים אלף שקלים");
        assert_eq!(normalized("בן 90"), "בן תשעים");
        assert_eq!(normalized("3 שנים"), "שלוש שנים");
        assert_eq!(normalized("3 ילדים"), "שלושה ילדים");
        // Two is a construct form before its noun, and one follows the noun.
        assert_eq!(normalized("2 שקלים"), "שני שקלים");
        assert_eq!(normalized("2 שנים"), "שתי שנים");
        assert_eq!(normalized("1 דונם"), "דונם אחד");
    }

    #[test]
    fn ordinals_in_dates() {
        assert_eq!(normalized("7 באוקטובר"), "השביעי באוקטובר");
        assert_eq!(normalized("3 בספטמבר"), "השלישי בספטמבר");
        assert_eq!(normalized("ה-1 בינואר"), "הראשון בינואר");
        // Above ten Hebrew goes back to the plain cardinal.
        assert_eq!(normalized("15 במאי"), "חמישה עשר במאי");
    }

    #[test]
    fn percentages_currency_and_decimals() {
        assert_eq!(
            normalized("12.3% רייטינג"),
            "שתים עשרה נקודה שלוש אחוז רייטינג"
        );
        assert_eq!(normalized("12% רייטינג"), "שנים עשר אחוז רייטינג");
        assert_eq!(
            normalized("6.4 מיליארד שקלים"),
            "שש נקודה ארבע מיליארד שקלים"
        );
        assert_eq!(normalized("30,000 ₪"), "שלושים אלף שקלים");
        assert_eq!(normalized("$250"), "מאתיים וחמישים דולר");
    }

    #[test]
    fn ranges_and_about_are_not_minus_signs() {
        assert_eq!(
            normalized("בשנים 2022 - 2025"),
            "בשנים אלפיים עשרים ושתיים עד אלפיים עשרים וחמש"
        );
        assert_eq!(normalized("כ-150 דונם"), "כמאה וחמישים דונם");
        assert_eq!(normalized("כ-600 בני אדם"), "כשש מאות בני אדם");
        assert_eq!(normalized("ב-2011"), "באלפיים ואחת עשרה");
    }

    #[test]
    fn hebrew_calendar_years_are_read_as_one_word() {
        assert_eq!(normalize_gematria_years("תשפ\"ז"), "תשפז");
        assert_eq!(normalize_gematria_years("בתשפ\"ה"), "בתשפה");
        assert_eq!(normalize_gematria_years("ה'תשפ\"ז"), "תשפז");
        // An ordinary acronym is not a year and is left alone here.
        assert_eq!(normalize_gematria_years("ארה\"ב"), "ארה\"ב");
    }

    #[test]
    fn scales_above_a_thousand() {
        assert_eq!(cardinal(1_000_000, Gender::Feminine), "מיליון");
        assert_eq!(
            cardinal(2_500_000, Gender::Feminine),
            "שני מיליון וחמש מאות אלף"
        );
        assert_eq!(cardinal(21_000, Gender::Feminine), "עשרים ואחד אלף");
        assert_eq!(cardinal(11_000, Gender::Feminine), "אחד עשר אלף");
        assert_eq!(cardinal(0, Gender::Feminine), "אפס");
    }

    /// Every corpus line from the issue, through the whole preparation
    /// pipeline rather than through this module alone.
    #[test]
    fn issue_corpus_through_the_full_pipeline() {
        let corpus = [
            (
                "זמן קיבוץ, גיליון 2011, 3 בספטמבר 2026",
                "זמן קיבוץ, גיליון אלפיים ואחת עשרה, השלישי בספטמבר אלפיים עשרים ושש",
            ),
            ("4500 דונם", "ארבעת אלפים וחמש מאות דונם"),
            ("6.4 מיליארד שקלים", "שש נקודה ארבע מיליארד שקלים"),
            ("30,000 שקלים", "שלושים אלף שקלים"),
            ("כ-150 דונם", "כמאה וחמישים דונם"),
            ("7,400 מיני צמחים", "שבעת אלפים וארבע מאות מיני צמחים"),
            ("12.3% רייטינג", "שתים עשרה נקודה שלוש אחוז רייטינג"),
            ("346 אלף צופים", "שלוש מאות ארבעים ושישה אלף צופים"),
            (
                "בשנים 2022 - 2025",
                "בשנים אלפיים עשרים ושתיים עד אלפיים עשרים וחמש",
            ),
            ("7 באוקטובר", "השביעי באוקטובר"),
            ("כ-600 בני אדם", "כשש מאות בני אדם"),
            ("בן 90", "בן תשעים"),
            ("תשפ\"ז", "תשפז"),
        ];
        for (input, expected) in corpus {
            let prepared = crate::handling::prepare_text_for_synthesis(input, "he");
            assert_eq!(prepared, expected, "input: {input}");
        }
    }

    #[test]
    fn leaves_other_language_spans_alone() {
        let text = normalize_hebrew_numbers("יש <en>iPhone 15</en> ו-3 מכשירים");
        assert!(text.contains("<en>iPhone 15</en>"), "unexpected: {text}");
        assert!(text.contains("ושלושה מכשירים"), "unexpected: {text}");
    }
}
