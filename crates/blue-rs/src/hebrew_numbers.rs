//! Hebrew number reading, delegated to `heb-tts-normalizer`.
//!
//! Hebrew numbers agree in gender with the noun they count, take construct
//! forms in the thousands, and read differently as a year, an ordinal or a
//! range. That is a lexicon problem rather than an arithmetic one, so this
//! module hands the work to a library that maintains the lexicon instead of
//! keeping a second copy of it here.

use heb_tts_normalizer::{Config, normalize};
use regex::Regex;
use std::sync::OnceLock;

fn config() -> &'static Config {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let mut config = Config::default();
        // `prepare_text_for_synthesis` already strips markup and settles
        // whitespace, and the paragraph breaks it leaves are what the chunker
        // splits on. Letting the normalizer collapse them would glue
        // paragraphs together and change where chunks fall.
        config.strip_markdown = false;
        config.clean_whitespace = false;
        config
    })
}

/// Digits inside an angle bracket span are markup rather than something to read
/// aloud, and the normalizer would otherwise turn `<break time="300ms"/>` into
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
pub fn normalize_hebrew_numbers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    for literal in literal_spans().find_iter(text) {
        out.push_str(&normalize(&text[cursor..literal.start()], config()));
        out.push_str(literal.as_str());
        cursor = literal.end();
    }
    out.push_str(&normalize(&text[cursor..], config()));
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
    }

    #[test]
    fn reads_a_grouping_comma_as_a_thousands_separator() {
        // This one was worse than the reported bug: the old pass treated the
        // comma as a decimal point and said "thirty point zero shekels".
        assert_eq!(
            normalize_hebrew_numbers("30,000 שקלים"),
            "שלושים אלף שקלים"
        );
    }

    #[test]
    fn counts_above_the_old_u16_ceiling() {
        assert_eq!(
            normalize_hebrew_numbers("500000 איש"),
            "חמש מאות אלף איש"
        );
    }

    #[test]
    fn keeps_the_prefix_attached_across_the_maqaf() {
        assert_eq!(
            normalize_hebrew_numbers("כ-150 דונם"),
            "כמאה וחמישים דונמים"
        );
    }

    #[test]
    fn reads_a_span_of_years_as_a_range() {
        assert_eq!(
            normalize_hebrew_numbers("בשנים 2022 - 2025"),
            "בשנים אלפיים עשרים ושתיים עד אלפיים עשרים וחמש"
        );
    }

    #[test]
    fn leaves_markup_literals_alone() {
        assert_eq!(
            normalize_hebrew_numbers("<break time=\"300ms\"/> 5 דונם"),
            "<break time=\"300ms\"/> חמישה דונמים"
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
