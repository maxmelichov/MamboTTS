//! Hebrew number front end: digits in, spoken Hebrew out.
//!
//! A port of the `hebrew-num2words` package (0.1.0), which upstream RenikudPlus
//! depends on and re-exports. Digits never appear in the model's training text,
//! so a digit that reaches the model is guaranteed-wrong output; every reading
//! here (gender agreement, construct forms, clock, date, year, percent,
//! decimal, identifier) follows the Python module rule for rule.
//!
//! The one deliberate difference: integers are read with 128-bit arithmetic
//! rather than Python's unbounded integers, so a number with more than 38
//! digits is passed through untouched instead of being spelled out. Anything
//! that long is read digit by digit as an identifier anyway unless it is a
//! percent or carries thousands separators.

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::{Captures, Regex};

use crate::pychars::{self, is_digit, split_whitespace, strip_chars};

/// Grammatical gender of a Hebrew numeral.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gender {
    Masc,
    Fem,
}

use Gender::{Fem, Masc};

const MASC_UNITS: [&str; 11] = [
    "אפס",
    "אחד",
    "שניים",
    "שלושה",
    "ארבעה",
    "חמישה",
    "שישה",
    "שבעה",
    "שמונה",
    "תשעה",
    "עשרה",
];
const FEM_UNITS: [&str; 11] = [
    "אפס",
    "אחת",
    "שתיים",
    "שלוש",
    "ארבע",
    "חמש",
    "שש",
    "שבע",
    "שמונה",
    "תשע",
    "עשר",
];

/// Construct ("smikhut") forms, used for 2 before a counted noun and for the
/// 3..10 multipliers of אלפים.
const MASC_CONSTRUCT: [(i128, &str); 9] = [
    (2, "שני"),
    (3, "שלושת"),
    (4, "ארבעת"),
    (5, "חמשת"),
    (6, "ששת"),
    (7, "שבעת"),
    (8, "שמונת"),
    (9, "תשעת"),
    (10, "עשרת"),
];
const FEM_CONSTRUCT: [(i128, &str); 1] = [(2, "שתי")];

const MASC_TEENS: [&str; 9] = [
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
const FEM_TEENS: [&str; 9] = [
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

const TENS: [&str; 8] = [
    "עשרים",
    "שלושים",
    "ארבעים",
    "חמישים",
    "שישים",
    "שבעים",
    "שמונים",
    "תשעים",
];

const HUNDREDS: [&str; 9] = [
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

const ORD_MASC: [&str; 10] = [
    "ראשון",
    "שני",
    "שלישי",
    "רביעי",
    "חמישי",
    "שישי",
    "שביעי",
    "שמיני",
    "תשיעי",
    "עשירי",
];
const ORD_FEM: [&str; 10] = [
    "ראשונה",
    "שנייה",
    "שלישית",
    "רביעית",
    "חמישית",
    "שישית",
    "שביעית",
    "שמינית",
    "תשיעית",
    "עשירית",
];

const MONTHS: [&str; 12] = [
    "ינואר",
    "פברואר",
    "מרץ",
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

const FRACTIONS: [(i128, i128, &str); 13] = [
    (1, 2, "חצי"),
    (1, 3, "שליש"),
    (1, 4, "רבע"),
    (2, 3, "שני שלישים"),
    (3, 4, "שלושת רבעי"),
    (1, 5, "חמישית"),
    (2, 5, "שתי חמישיות"),
    (3, 5, "שלוש חמישיות"),
    (4, 5, "ארבע חמישיות"),
    (1, 6, "שישית"),
    (1, 8, "שמינית"),
    (3, 8, "שלוש שמיניות"),
    (1, 10, "עשירית"),
];

/// `־ים` plurals that are feminine, and `־ות` plurals that are masculine: the
/// two classes the ending heuristic gets wrong.
const FEM_IM_PLURALS: [&str; 13] = [
    "שנים",
    "נשים",
    "מילים",
    "ביצים",
    "דבורים",
    "חיטים",
    "תאנים",
    "פעמים",
    "אבנים",
    "עצים",
    "יונים",
    "צפרניים",
    "שעורים",
];
const MASC_OT_PLURALS: [&str; 20] = [
    "שולחנות",
    "כיסאות",
    "מקומות",
    "לילות",
    "קולות",
    "חלומות",
    "רחובות",
    "אבות",
    "שמות",
    "לוחות",
    "קירות",
    "מטבעות",
    "שבועות",
    "רעיונות",
    "חלונות",
    "זנבות",
    "מזלגות",
    "סכינים",
    "אריות",
    "נכסים",
];

/// Explicit singular / irregular entries; everything else falls through to the
/// ending heuristic in [`noun_gender`].
fn noun_gender_table() -> &'static HashMap<&'static str, Gender> {
    static TABLE: OnceLock<HashMap<&'static str, Gender>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let feminine = [
            "שעה",
            "שעות",
            "דקה",
            "דקות",
            "שניה",
            "שניות",
            "שנה",
            "שנות",
            "מנה",
            "מנות",
            "כוס",
            "כוסות",
            "קומה",
            "קומות",
            "פעם",
            "פעמים",
            "בת",
            "בנות",
            "מעלה",
            "מעלות",
            "טיסה",
            "טיסות",
            "אישה",
            "נשים",
            "דירה",
            "דירות",
            "עונה",
            "עונות",
            "שורה",
            "שורות",
            "חנות",
            "חנויות",
            "מסעדה",
            "מסעדות",
            "פגישה",
            "פגישות",
            "יחידה",
            "יחידות",
            "משימה",
            "משימות",
            "מחברת",
            "מחברות",
            "מעטפה",
            "מעטפות",
            "חולצה",
            "חולצות",
            "מיטה",
            "מיטות",
            "דלת",
            "דלתות",
            "רגל",
            "רגליים",
            "יד",
            "ידיים",
            "עין",
            "עיניים",
            "אוזן",
            "אוזניים",
            "עיר",
            "ערים",
            "ארץ",
            "דרך",
            "נפש",
            "כיתה",
            "כיתות",
            "אגורה",
            "אגורות",
        ];
        let masculine = [
            "שקל",
            "שקלים",
            "ש\"ח",
            "דולר",
            "דולרים",
            "יורו",
            "אירו",
            "יום",
            "ימים",
            "ימי",
            "חודש",
            "חודשים",
            "חודשי",
            "שבוע",
            "שבועות",
            "איש",
            "אנשים",
            "ילד",
            "ילדים",
            "בן",
            "בנים",
            "חבר",
            "חברים",
            "עובד",
            "עובדים",
            "ספר",
            "ספרים",
            "כלב",
            "כלבים",
            "חתול",
            "חתולים",
            "תפוח",
            "תפוחים",
            "כרטיס",
            "כרטיסים",
            "מטר",
            "מטרים",
            "קילומטר",
            "קילו",
            "ליטר",
            "מיליון",
            "מיליארד",
            "אלף",
            "אחוז",
            "אחוזים",
            "מוצר",
            "מוצרים",
            "משתתף",
            "משתתפים",
            "בניין",
            "בניינים",
            "משרד",
            "משרדים",
            "סטודנט",
            "סטודנטים",
            "דייר",
            "דיירים",
            "עט",
            "עטים",
            "מכתב",
            "מכתבים",
            "כדור",
            "כדורים",
            "נרשם",
            "נרשמים",
            "פועל",
            "פועלים",
            "ק\"ג",
            "ק\"מ",
            "מ\"ר",
            "קמ\"ש",
            "מ\"מ",
            "ס\"מ",
        ];
        feminine
            .into_iter()
            .map(|word| (word, Fem))
            .chain(masculine.into_iter().map(|word| (word, Masc)))
            .collect()
    })
}

/// A numeral within 3 tokens after one of these is an identifier, not a count.
const CODE_NOUNS: [&str; 30] = [
    "קוד",
    "מספר",
    "מיקוד",
    "סיסמה",
    "סיסמא",
    "סיומת",
    "ברקוד",
    "טלפון",
    "שלוחה",
    "סניף",
    "מזהה",
    "כרטיס",
    "אשראי",
    "פוליסה",
    "הזמנה",
    "משלוח",
    "זהות",
    "ת.ז",
    "ת\"ז",
    "רישיון",
    "אימות",
    "פנימי",
    "סודי",
    "סידורי",
    "חשאי",
    "אישי",
    "בנק",
    "גישה",
    "ז",
    "מנעול",
];
/// A bare numeral right after one of these is an ordinal.
const ORDINAL_NOUNS: [&str; 13] = [
    "קומה",
    "פעם",
    "מקום",
    "פרק",
    "חלק",
    "עונה",
    "טסט",
    "שורה",
    "שלב",
    "כיתה",
    "רבעון",
    "סיבוב",
    "מחזור",
];
const HOUR_CUES: [&str; 6] = ["שעה", "השעה", "בשעה", "לשעה", "שעון", "השעון"];
const DAY_OF_MONTH_CUES: [&str; 3] = ["לחודש", "בחודש", "החודש"];
const YEAR_CUES: [&str; 6] = ["שנת", "בשנת", "משנת", "לשנת", "שנה", "השנה"];

const PUNCT_TAIL: &str = ".,!?;:\"')]…";
const NOUN_STRIP: &str = ".,!?;:\"'()[]—–-";

fn regex(pattern: &'static str) -> Regex {
    Regex::new(pattern).expect("valid regex")
}

fn cached(slot: &'static OnceLock<Regex>, pattern: &'static str) -> &'static Regex {
    slot.get_or_init(|| regex(pattern))
}

macro_rules! re {
    ($pattern:expr) => {{
        static SLOT: OnceLock<Regex> = OnceLock::new();
        cached(&SLOT, $pattern)
    }};
}

// --------------------------------------------------------------------------
// Number words
// --------------------------------------------------------------------------

/// Attach the conjunction ו to a word (orthography only; the model does the
/// phonology, so no u/ve sandhi is applied here).
fn vav(word: &str) -> String {
    format!("ו{word}")
}

fn units(n: i128, gender: Gender) -> &'static str {
    let table = if gender == Masc {
        MASC_UNITS
    } else {
        FEM_UNITS
    };
    table[n as usize]
}

fn teens(n: i128, gender: Gender) -> &'static str {
    let table = if gender == Masc {
        MASC_TEENS
    } else {
        FEM_TEENS
    };
    table[(n - 11) as usize]
}

fn construct_form(n: i128, gender: Gender) -> Option<&'static str> {
    let table: &[(i128, &str)] = if gender == Masc {
        &MASC_CONSTRUCT
    } else {
        &FEM_CONSTRUCT
    };
    table
        .iter()
        .find(|&&(key, _)| key == n)
        .map(|&(_, word)| word)
}

fn words(text: &str) -> Vec<String> {
    text.split(' ').map(str::to_owned).collect()
}

/// The אלף / אלפים group for a multiplier `k`.
fn thousands_words(k: i128) -> Vec<String> {
    if k == 1 {
        return vec!["אלף".to_owned()];
    }
    if k == 2 {
        return vec!["אלפיים".to_owned()];
    }
    if let Some(form) = construct_form(k, Masc) {
        return vec![form.to_owned(), "אלפים".to_owned()];
    }
    let mut out = words(&cardinal(k, Masc, false));
    out.push("אלף".to_owned());
    out
}

/// Split 1..999 into word groups (hundreds / tens+units).
fn sub_thousand_groups(mut n: i128, gender: Gender) -> Vec<Vec<String>> {
    let mut groups = Vec::new();
    if n >= 100 {
        groups.push(words(HUNDREDS[(n / 100 - 1) as usize]));
        n %= 100;
    }
    if n >= 20 {
        let mut tail = vec![TENS[(n / 10 - 2) as usize].to_owned()];
        if n % 10 != 0 {
            tail.push(vav(units(n % 10, gender)));
        }
        groups.push(tail);
    } else if n >= 11 {
        groups.push(words(teens(n, gender)));
    } else if n >= 1 {
        groups.push(vec![units(n, gender).to_owned()]);
    }
    groups
}

/// Hebrew cardinal.
///
/// `construct` yields שני / שתי (2) and שלושת … עשרת (3..10), the forms used
/// before a counted noun.
pub fn cardinal(mut n: i128, gender: Gender, construct: bool) -> String {
    if n < 0 {
        return format!("מינוס {}", cardinal(-n, gender, construct));
    }
    if n == 0 {
        return "אפס".to_owned();
    }
    if construct && let Some(form) = construct_form(n, gender) {
        return form.to_owned();
    }
    let mut groups: Vec<Vec<String>> = Vec::new();
    if n >= 1000 {
        groups.push(thousands_words(n / 1000));
        n %= 1000;
    }
    groups.extend(sub_thousand_groups(n, gender));
    // The conjunction lands on the final component only, and only when that
    // component does not already carry one (…שלושים ושניים keeps a single ו).
    if groups.len() > 1
        && let Some(last) = groups.last_mut()
        && !last.iter().any(|word| word.starts_with('ו'))
    {
        last[0] = vav(&last[0]);
    }
    groups.into_iter().flatten().collect::<Vec<_>>().join(" ")
}

/// Hebrew ordinal. Above 10 Hebrew uses the definite cardinal (`ה-11` →
/// האחת עשרה).
pub fn ordinal(n: i128, gender: Gender, definite: bool) -> String {
    let table = if gender == Masc { ORD_MASC } else { ORD_FEM };
    if (1..=10).contains(&n) {
        let word = table[(n - 1) as usize];
        return if definite {
            format!("ה{word}")
        } else {
            word.to_owned()
        };
    }
    let card = cardinal(n, gender, false);
    if definite { format!("ה{card}") } else { card }
}

/// Digit-by-digit reading (identifiers, phone numbers, codes).
///
/// A character that is not a decimal digit is passed through; the callers that
/// have to reproduce Python's `int()` failure there use [`try_spell_digits`].
pub fn spell_digits(digits: &str) -> String {
    try_spell_digits(digits).unwrap_or_else(|| {
        digits
            .chars()
            .map(|d| match pychars::decimal_value(d) {
                Some(value) => FEM_UNITS[value as usize].to_owned(),
                None => d.to_string(),
            })
            .collect::<Vec<_>>()
            .join(" ")
    })
}

/// [`spell_digits`], or `None` when a character is not a decimal digit — the
/// point where the Python original raises `ValueError` and the token is left
/// as written.
fn try_spell_digits(digits: &str) -> Option<String> {
    digits
        .chars()
        .map(|d| Some(FEM_UNITS[pychars::decimal_value(d)? as usize].to_owned()))
        .collect::<Option<Vec<_>>>()
        .map(|words| words.join(" "))
}

// --------------------------------------------------------------------------
// Gender of the counted noun
// --------------------------------------------------------------------------

/// Prefix letters that can be glued onto a Hebrew word.
fn strip_prefix(word: &str) -> String {
    if word.chars().count() > 3 {
        let stripped = re!(r"^(?:[בכלמשו]?ה|[בכלמשוה])").replacen(word, 1, "");
        if stripped.chars().count() >= 2 {
            return stripped.into_owned();
        }
    }
    word.to_owned()
}

/// Gender of a counted noun, or `None` when `word` does not look like one.
///
/// Order: explicit lexicon → irregular-plural sets → ending heuristic.
/// Returning `None` (rather than defaulting) is what lets the caller tell
/// `8 שעות` (counted) apart from `ב-8 שילמתי` (an hour).
pub fn noun_gender(word: &str) -> Option<Gender> {
    let word = strip_chars(word, NOUN_STRIP);
    if word.is_empty() {
        return None;
    }
    let stripped = strip_prefix(word);
    for candidate in [word, stripped.as_str()] {
        if let Some(&gender) = noun_gender_table().get(candidate) {
            return Some(gender);
        }
        if FEM_IM_PLURALS.contains(&candidate) {
            return Some(Fem);
        }
        if MASC_OT_PLURALS.contains(&candidate) {
            return Some(Masc);
        }
    }
    if stripped.chars().count() < 3 {
        return None;
    }
    if stripped.ends_with("יות") || stripped.ends_with("ות") {
        return Some(Fem);
    }
    if stripped.ends_with("ים") || stripped.ends_with("יים") {
        return Some(Masc);
    }
    None
}

// --------------------------------------------------------------------------
// Sub-expanders
// --------------------------------------------------------------------------

/// 12-hour clock hour, feminine.
fn hour_word(h: i128) -> String {
    let mut h = h.rem_euclid(24);
    if h == 0 {
        h = 12;
    }
    if h > 12 {
        h -= 12;
    }
    cardinal(h, Fem, false)
}

/// `hh:mm` → spoken Hebrew: ורבע / וחצי / רבע ל־ / עשרים ל־.
pub fn expand_time(hh: i128, mm: i128) -> String {
    let hour = hour_word(hh);
    match mm {
        0 => hour,
        15 => format!("{hour} ורבע"),
        30 => format!("{hour} וחצי"),
        45 => format!("רבע ל{}", hour_word(hh + 1)),
        mm if mm < 30 => format!("{hour} {}", vav(&cardinal(mm, Masc, false))),
        mm => format!("{} ל{}", cardinal(60 - mm, Masc, false), hour_word(hh + 1)),
    }
}

/// `D.M[.YYYY]` → masculine day + ב+month + feminine year.
pub fn expand_date(day: i128, month: i128, year: Option<i128>) -> Option<String> {
    let name = MONTHS.get((month - 1) as usize)?;
    let mut out = format!("{} ב{name}", cardinal(day, Masc, false));
    if let Some(year) = year {
        out.push(' ');
        out.push_str(&cardinal(year, Fem, false));
    }
    Some(out)
}

pub fn expand_year(n: i128) -> String {
    cardinal(n, Fem, false)
}

fn decimal_words(int_part: &str, frac_part: &str, gender: Gender) -> Option<String> {
    let ip = parse_decimal(int_part)?;
    let fp = frac_part;
    if fp == "5" {
        // X.5 → "X וחצי", the dominant reading in the gold asset.
        if ip == 0 {
            return Some("חצי".to_owned());
        }
        return Some(format!("{} וחצי", cardinal(ip, gender, false)));
    }
    if fp.chars().count() == 2 {
        // Price reading: 19.90 → "תשע עשרה תשעים".
        let head = cardinal(ip, if ip != 1 { Fem } else { Masc }, false);
        return Some(format!(
            "{head} {}",
            cardinal(parse_decimal(fp)?, Fem, false)
        ));
    }
    let head = cardinal(ip, if ip == 1 { Masc } else { Fem }, false);
    Some(format!("{head} נקודה {}", try_spell_digits(fp)?))
}

/// Phone / serial numbers: hyphen groups read digit by digit, except the
/// 1-800 style service numbers whose 3-digit groups are read as numbers.
fn grouped_digits(groups: &[&str]) -> Option<String> {
    let first = *groups.first()?;
    if (first == "1" || first == "*") && groups.len() > 1 && groups[1].chars().count() == 3 {
        let mut out = vec![if first.chars().count() == 1 {
            MASC_UNITS[parse_decimal(first)? as usize].to_owned()
        } else {
            try_spell_digits(first)?
        }];
        for group in &groups[1..] {
            out.push(cardinal(parse_decimal(group)?, Fem, false));
        }
        return Some(out.join(" "));
    }
    Some(
        groups
            .iter()
            .map(|group| try_spell_digits(group))
            .collect::<Option<Vec<_>>>()?
            .join(" "),
    )
}

// --------------------------------------------------------------------------
// Token-level driver
// --------------------------------------------------------------------------

fn token_re() -> &'static Regex {
    re!(concat!(
        r#"^(?P<pre>[\u{0590}-\u{05EA}"']{0,4})(?P<sep>-?)"#,
        r#"(?P<body>\d[\d.,:/%\-]*?)(?P<post>[\.,!\?;:"'\)\]…]*)$"#
    ))
}

/// `int(s)` for a string of decimal digits.
fn parse_decimal(text: &str) -> Option<i128> {
    if text.is_empty() {
        return None;
    }
    let mut value: i128 = 0;
    for c in text.chars() {
        let digit = pychars::decimal_value(c)? as i128;
        value = value.checked_mul(10)?.checked_add(digit)?;
    }
    Some(value)
}

/// `(gender, noun)` of the following token, when it looks like a noun. Only the
/// immediately-following token is consulted — a numeral binds to the noun it
/// precedes.
fn next_noun_gender(next_words: &[&str]) -> Option<Gender> {
    noun_gender(next_words.first()?)
}

fn is_code_context(prev_words: &[&str]) -> bool {
    // A cue belongs to the first numeral that follows it: stop the search at the
    // previous numeral so "קוד 500 לעומת 500 איש" reads the second 500 as a
    // number, not as a code.
    let mut prev_words = prev_words;
    for i in (0..prev_words.len()).rev() {
        if prev_words[i].chars().any(is_digit) {
            prev_words = &prev_words[i + 1..];
            break;
        }
    }
    let tail = &prev_words[prev_words.len().saturating_sub(3)..];
    tail.iter().any(|word| {
        let base = strip_chars(word, &format!("{PUNCT_TAIL}-"));
        CODE_NOUNS.contains(&base) || CODE_NOUNS.contains(&strip_prefix(base).as_str())
    })
}

fn prev_base<'a>(prev_words: &[&'a str], k: usize) -> &'a str {
    if prev_words.len() < k {
        return "";
    }
    strip_chars(prev_words[prev_words.len() - k], &format!("{PUNCT_TAIL}-"))
}

fn month_follows(next_words: &[&str]) -> bool {
    let Some(word) = next_words.first() else {
        return false;
    };
    let word = strip_chars(word, PUNCT_TAIL);
    if DAY_OF_MONTH_CUES.contains(&word) {
        return true;
    }
    let stripped = re!(r"^[בלמה]").replacen(word, 1, "");
    MONTHS.contains(&word) || MONTHS.contains(&stripped.as_ref())
}

/// The main decision tree for a bare integer token.
pub fn expand_integer(
    body: &str,
    pre: &str,
    prev_words: &[&str],
    next_words: &[&str],
) -> Option<String> {
    let digits = body;
    let length = digits.chars().count();
    // Python reads the value first, so a body that `int()` rejects leaves the
    // token as written.
    let spelled = try_spell_digits(digits)?;

    // 1. identifiers -> digit by digit
    if length >= 5
        || (length >= 3 && is_code_context(prev_words))
        || (length > 1 && digits.starts_with('0'))
    {
        return Some(spelled);
    }
    let n = parse_decimal(digits)?;

    // 2. years
    if YEAR_CUES.contains(&prev_base(prev_words, 1)) {
        return Some(expand_year(n));
    }

    // 3. day of month
    if month_follows(next_words) && (1..=31).contains(&n) {
        return Some(cardinal(n, Masc, false));
    }

    // 4. ordinal: the "ה-N" construction, or a bare numeral after an ordinal noun
    let prev = prev_base(prev_words, 1);
    let prev_stripped = if ORDINAL_NOUNS.contains(&prev) {
        prev.to_owned()
    } else {
        strip_prefix(prev)
    };
    if pre.ends_with('ה') && n <= 999 {
        // "ה-N": the definite article is already in `pre` and gets re-attached
        // by the caller, so the ordinal itself is produced bare.
        let gender = noun_gender_table()
            .get(prev_stripped.as_str())
            .copied()
            .or_else(|| noun_gender(prev))
            .unwrap_or(Masc);
        return Some(ordinal(n, gender, false));
    }
    if ORDINAL_NOUNS.contains(&prev_stripped.as_str()) && n <= 10 {
        let gender = noun_gender_table()
            .get(prev_stripped.as_str())
            .copied()
            .unwrap_or(Masc);
        return Some(ordinal(n, gender, false));
    }

    // 5. counted noun follows -> agree with it
    if let Some(gender) = next_noun_gender(next_words) {
        if n == 2 {
            return Some(cardinal(2, gender, true));
        }
        return Some(cardinal(n, gender, false));
    }

    // 6. hour context
    if (matches!(pre, "ב" | "מ" | "ל")
        || matches!(prev, "עד" | "ועד")
        || HOUR_CUES.contains(&prev)
        || HOUR_CUES.contains(&prev_base(prev_words, 2)))
        && (0..=24).contains(&n)
        && !is_code_context(prev_words)
    {
        return Some(hour_word(n));
    }

    // 7. bare numeral: absolute (feminine) series, 1 -> אחד
    Some(cardinal(n, if n == 1 { Masc } else { Fem }, false))
}

/// Expand digit runs in place, reading each as a feminine cardinal.
fn expand_digit_runs(text: &str) -> String {
    re!(r"\d+")
        .replace_all(text, |caps: &Captures| {
            let run = &caps[0];
            match parse_decimal(run) {
                Some(n) => cardinal(n, Fem, false),
                None => run.to_owned(),
            }
        })
        .into_owned()
}

/// Expand the numeric core of a token (no prefix, no trailing punctuation).
pub fn expand_body(
    body: &str,
    pre: &str,
    prev_words: &[&str],
    next_words: &[&str],
) -> Option<String> {
    // -- percent -------------------------------------------------------
    if let Some(core) = body.strip_suffix('%') {
        // thousands separators may appear inside a percent core ("51,034%")
        let core = if re!(r"^\d{1,3}(?:,\d{3})+$").is_match(core) {
            core.replace(',', "")
        } else {
            core.to_owned()
        };
        // The source sometimes writes both the sign and the word ("10% אחוז").
        let spelled_out = next_words
            .first()
            .is_some_and(|word| matches!(strip_chars(word, PUNCT_TAIL), "אחוז" | "אחוזים"));
        let tail = if spelled_out { "" } else { " אחוז" };
        if let Some((ip, fp)) = core.split_once('.') {
            return Some(decimal_words(ip, fp, Masc)? + tail);
        }
        return Some(cardinal(parse_decimal(&core)?, Masc, false) + tail);
    }

    // -- clock ---------------------------------------------------------
    if let Some(caps) = re!(r"^(\d{1,2}):(\d{2})$").captures(body) {
        return Some(expand_time(
            parse_decimal(&caps[1])?,
            parse_decimal(&caps[2])?,
        ));
    }

    // -- fraction / date with slash ------------------------------------
    if let Some(caps) = re!(r"^(\d{1,2})/(\d{1,4})$").captures(body) {
        let (a, b) = (parse_decimal(&caps[1])?, parse_decimal(&caps[2])?);
        if let Some(&(_, _, word)) = FRACTIONS
            .iter()
            .find(|&&(num, den, _)| num == a && den == b)
        {
            return Some(word.to_owned());
        }
        if (1..=12).contains(&b) && (1..=31).contains(&a) {
            return expand_date(a, b, None);
        }
        return Some(format!(
            "{} {}",
            cardinal(a, Fem, false),
            cardinal(b, Fem, false)
        ));
    }

    // -- phone / serial with hyphens -----------------------------------
    if body.contains('-') {
        let groups: Vec<&str> = body.split('-').collect();
        if groups.len() >= 2
            && groups
                .iter()
                .all(|group| !group.is_empty() && group.chars().all(is_digit))
        {
            if groups.len() >= 3 || groups.iter().any(|group| group.chars().count() >= 5) {
                return grouped_digits(&groups);
            }
            if groups.iter().all(|group| group.chars().count() <= 2) {
                // score line, e.g. 3-0
                return Some(
                    groups
                        .iter()
                        .map(|group| Some(cardinal(parse_decimal(group)?, Fem, false)))
                        .collect::<Option<Vec<_>>>()?
                        .join(" "),
                );
            }
            return grouped_digits(&groups);
        }
    }

    // -- thousands separators ------------------------------------------
    if re!(r"^\d{1,3}(?:,\d{3})+$").is_match(body) {
        let n = parse_decimal(&body.replace(',', ""))?;
        let gender = next_noun_gender(next_words);
        if n >= 1_000_000 {
            let head = n / 1_000_000;
            let rest = n % 1_000_000;
            let word = if head == 1 {
                "מיליון".to_owned()
            } else {
                format!("{} מיליון", cardinal(head, Masc, true))
            };
            return Some(if rest == 0 {
                word
            } else {
                format!("{word} {}", cardinal(rest, gender.unwrap_or(Masc), false))
            });
        }
        return Some(cardinal(n, gender.unwrap_or(Masc), false));
    }

    // -- dates ----------------------------------------------------------
    if let Some(caps) = re!(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$").captures(body) {
        let (d, mo, y) = (
            parse_decimal(&caps[1])?,
            parse_decimal(&caps[2])?,
            parse_decimal(&caps[3])?,
        );
        if (1..=31).contains(&d) && (1..=12).contains(&mo) {
            return expand_date(d, mo, Some(y));
        }
    }
    if let Some(caps) = re!(r"^(\d{1,2})\.(\d{4})$").captures(body) {
        let month = parse_decimal(&caps[1])?;
        if (1..=12).contains(&month) {
            return Some(format!(
                "ב{} {}",
                MONTHS[(month - 1) as usize],
                cardinal(parse_decimal(&caps[2])?, Fem, false)
            ));
        }
    }

    // -- one decimal point ---------------------------------------------
    if let Some(caps) = re!(r"^(\d+)\.(\d+)$").captures(body) {
        let (ip_s, fp_s) = (caps[1].to_owned(), caps[2].to_owned());
        let (ip, fp) = (parse_decimal(&ip_s)?, parse_decimal(&fp_s)?);
        let clock_cue = HOUR_CUES.contains(&prev_base(prev_words, 1));
        // "ב-1.9" style: a prefixed D.M with a plausible month and no unit noun
        // after it is a date, not a decimal.
        let unit_follows = next_words
            .first()
            .is_some_and(|word| noun_gender(word).is_some());
        if !clock_cue
            && matches!(pre, "ב" | "ל" | "ה")
            && fp_s.chars().count() <= 2
            && (1..=12).contains(&fp)
            && (1..=31).contains(&ip)
            && !unit_follows
        {
            return expand_date(ip, fp, None);
        }
        if clock_cue && fp_s.chars().count() == 2 {
            return Some(expand_time(ip, fp));
        }
        let gender = next_noun_gender(next_words).unwrap_or(Fem);
        return decimal_words(&ip_s, &fp_s, gender);
    }

    // -- plain integer ---------------------------------------------------
    if !body.is_empty() && body.chars().all(is_digit) {
        return expand_integer(body, pre, prev_words, next_words);
    }

    // -- anything else: read the digits, keep separators as words --------
    Some(expand_digit_runs(body))
}

/// Expand a single whitespace-delimited token; returns it unchanged when it
/// carries no digit.
pub fn expand_token(token: &str, prev_words: &[&str], next_words: &[&str]) -> String {
    if !token.chars().any(is_digit) {
        return token.to_owned();
    }
    let Some(caps) = token_re().captures(token) else {
        // Fall back: expand digit runs in place.
        return expand_digit_runs(token);
    };
    let (pre, body, post) = (&caps["pre"], &caps["body"], &caps["post"]);
    // An unsupported numeric format is passed through rather than guessed at.
    let Some(expanded) = expand_body(body, pre, prev_words, next_words) else {
        return token.to_owned();
    };
    format!("{pre}{expanded}{post}")
}

/// True when `text` contains a decimal digit.
pub fn contains_digit(text: &str) -> bool {
    text.chars().any(pychars::is_decimal)
}

/// Expand every numeric expression in `text`. No-op on digit-free text.
///
/// Like the Python original, this re-joins the tokens with single spaces, so
/// runs of whitespace in digit-bearing text collapse. Use
/// [`normalize_numbers_keep_spacing`] where the original layout matters.
pub fn normalize_numbers(text: &str) -> String {
    if !contains_digit(text) {
        return text.to_owned();
    }
    let tokens = split_whitespace(text);
    expand_tokens(&tokens).join(" ")
}

/// [`normalize_numbers`] with every separator between tokens left as written.
///
/// The token context each expansion sees is the same, so the words are the
/// same; only the whitespace between them differs. MamboTTS splits text into
/// chunks on its paragraph breaks after this pass, so it cannot afford to have
/// them collapsed into spaces.
pub fn normalize_numbers_keep_spacing(text: &str) -> String {
    if !contains_digit(text) {
        return text.to_owned();
    }
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        if pychars::is_space(c) {
            if let Some(s) = start.take() {
                spans.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }
    let tokens: Vec<&str> = spans.iter().map(|&(s, e)| &text[s..e]).collect();
    let expanded = expand_tokens(&tokens);

    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    for (&(s, e), word) in spans.iter().zip(expanded) {
        out.push_str(&text[cursor..s]);
        out.push_str(&word);
        cursor = e;
    }
    out.push_str(&text[cursor..]);
    out
}

fn expand_tokens(tokens: &[&str]) -> Vec<String> {
    tokens
        .iter()
        .enumerate()
        .map(|(i, token)| {
            if token.chars().any(pychars::is_decimal) {
                expand_token(token, &tokens[..i], &tokens[i + 1..])
            } else {
                (*token).to_owned()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_agree_with_the_noun_that_follows() {
        assert_eq!(normalize_numbers("8 שעות"), "שמונה שעות");
        assert_eq!(normalize_numbers("8 שקלים"), "שמונה שקלים");
        assert_eq!(normalize_numbers("2 ספרים"), "שני ספרים");
        assert_eq!(normalize_numbers("2 שעות"), "שתי שעות");
    }

    #[test]
    fn the_readme_price_example() {
        assert_eq!(
            normalize_numbers("המחיר 1250 שקלים"),
            "המחיר אלף מאתיים וחמישים שקלים"
        );
    }

    #[test]
    fn clock_date_percent_and_identifier_readings() {
        assert_eq!(normalize_numbers("בשעה 8:15"), "בשעה שמונה ורבע");
        assert_eq!(normalize_numbers("ב 10:45"), "ב רבע לאחת עשרה");
        assert_eq!(
            normalize_numbers("14.3.2024"),
            "ארבעה עשר במרץ אלפיים עשרים וארבע"
        );
        assert_eq!(normalize_numbers("50%"), "חמישים אחוז");
        assert_eq!(normalize_numbers("קוד 4821"), "קוד ארבע שמונה שתיים אחת");
        assert_eq!(
            normalize_numbers("בשנת 1948"),
            "בשנת אלף תשע מאות ארבעים ושמונה"
        );
    }

    #[test]
    fn digit_free_text_is_returned_unchanged() {
        let text = "שלום  עולם\n\nמה נשמע";
        assert_eq!(normalize_numbers(text), text);
    }

    #[test]
    fn keeping_the_spacing_changes_only_the_whitespace() {
        let text = "שורה ראשונה\n\nשורה שנייה 12";
        assert_eq!(
            normalize_numbers_keep_spacing(text),
            "שורה ראשונה\n\nשורה שנייה שתים עשרה"
        );
        assert_eq!(normalize_numbers(text), "שורה ראשונה שורה שנייה שתים עשרה");
        assert_eq!(
            split_whitespace(&normalize_numbers_keep_spacing(text)),
            split_whitespace(&normalize_numbers(text))
        );
    }

    #[test]
    fn an_integer_too_long_for_i128_passes_through() {
        let text = format!("{}%", "9".repeat(60));
        assert_eq!(normalize_numbers(&text), text);
    }
}
