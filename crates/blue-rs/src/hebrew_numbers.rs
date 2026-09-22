//! Hebrew number reading, from the RenikudPlus front end.
//!
//! Hebrew numbers agree in gender with the noun they count, take construct
//! forms in the thousands, and read differently as a year, an ordinal, a clock
//! time or an identifier. RenikudPlus ships that lexicon (a port of the
//! `hebrew-num2words` package it uses upstream), and its G2P was tuned against
//! exactly those readings, so this module hands the work to it rather than
//! keeping a second, differently-opinionated copy.

use regex::Regex;
use renikud_plus_rs::numbers::normalize_numbers_keep_spacing;
use std::sync::OnceLock;

/// Digits inside an angle bracket span are markup rather than something to read
/// aloud, and the expander would otherwise turn `<break time="300ms"/>` into
/// `<break time="שלוש מאותms"/>`.
///
/// Nothing in the text path relies on this today: `normalize_common_text` runs
/// first and rewrites such a tag beyond recognition anyway. It is here so that
/// this pass is not the thing that breaks markup if the steps above it ever
/// learn to carry it.
fn literal_spans() -> &'static Regex {
    static SPANS: OnceLock<Regex> = OnceLock::new();
    SPANS.get_or_init(|| Regex::new(r"<[^>]*>").expect("valid regex"))
}

/// Read digits in Hebrew text as Hebrew words, leaving markup literals alone.
///
/// Whitespace is kept as written — the chunker splits on the paragraph breaks
/// this pass leaves behind, so collapsing them would move where chunks fall.
pub fn normalize_hebrew_numbers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    for literal in literal_spans().find_iter(text) {
        out.push_str(&normalize_numbers_keep_spacing(
            &text[cursor..literal.start()],
        ));
        out.push_str(literal.as_str());
        cursor = literal.end();
    }
    out.push_str(&normalize_numbers_keep_spacing(&text[cursor..]));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_year_as_a_year() {
        assert_eq!(
            normalize_hebrew_numbers("גיליון 2011"),
            "גיליון אלפיים ואחת עשרה"
        );
        assert_eq!(
            normalize_hebrew_numbers("בשנת 1948 קמה המדינה"),
            "בשנת אלף תשע מאות ארבעים ושמונה קמה המדינה"
        );
    }

    #[test]
    fn reads_a_grouping_comma_as_a_thousands_separator() {
        assert_eq!(normalize_hebrew_numbers("30,000 שקלים"), "שלושים אלף שקלים");
    }

    #[test]
    fn a_long_digit_run_reads_as_an_identifier() {
        // Five digits or more with no separators is a code, a phone number or
        // an account number far more often than a count, so RenikudPlus reads
        // it digit by digit. A grouped number is still a number.
        assert_eq!(
            normalize_hebrew_numbers("500000 איש"),
            "חמש אפס אפס אפס אפס אפס איש"
        );
        assert_eq!(normalize_hebrew_numbers("500,000 איש"), "חמש מאות אלף איש");
        assert_eq!(normalize_hebrew_numbers("12,000 איש"), "שנים עשר אלף איש");
    }

    #[test]
    fn keeps_the_prefix_attached_across_the_maqaf() {
        assert_eq!(normalize_hebrew_numbers("כ-150 דונם"), "כמאה וחמישים דונם");
    }

    #[test]
    fn counts_agree_with_the_noun_they_count() {
        assert_eq!(normalize_hebrew_numbers("8 שעות"), "שמונה שעות");
        assert_eq!(normalize_hebrew_numbers("8 שקלים"), "שמונה שקלים");
        assert_eq!(normalize_hebrew_numbers("2 ספרים"), "שני ספרים");
    }

    #[test]
    fn leaves_markup_literals_alone() {
        assert_eq!(
            normalize_hebrew_numbers("<break time=\"300ms\"/> 5 דונם"),
            "<break time=\"300ms\"/> חמש דונם"
        );
    }

    #[test]
    fn keeps_paragraph_breaks_the_chunker_splits_on() {
        assert_eq!(
            normalize_hebrew_numbers("שורה ראשונה\n\nשורה שנייה 12"),
            "שורה ראשונה\n\nשורה שנייה שתים עשרה"
        );
    }
}
