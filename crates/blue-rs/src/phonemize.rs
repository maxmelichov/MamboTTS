use std::path::Path;
use std::sync::Once;

use anyhow::{Result, bail};
use ort::session::Session;
use renikud_plus_rs::lexicon::{ForceLexicon, OnInvalidEntry};
use renikud_plus_rs::{G2P, G2PConfig, NiqqudMode, Options};

use crate::chunking::{is_py_word_char, split_sentences};
use crate::handling::{
    canonical_lang, inline_lang_spans, is_js_space, js_collapse_spaces, js_trim, prepare,
    strip_emoji, strip_silent_separator_tokens, tag_latin_runs,
};
use crate::ipa::normalize_ipa_with;

/// RenikudPlus as the site and the BlueV3 reference run it: points in the
/// input are dropped and the bare skeleton is read (`niqqud="strip"`,
/// js/g2p/hebrew.js `NIKUD_RE`). [`Phonemizer::set_honor_niqqud`] turns on
/// reading them as constraints instead, for text whose points someone chose
/// (the app's opt-in niqqud editor); the hatama (U+05AB) is read then too.
fn hebrew_config() -> G2PConfig {
    G2PConfig {
        niqqud: NiqqudMode::Strip,
        ..G2PConfig::default()
    }
}

/// Languages the BlueV3 release reads.
///
/// Codes:
/// - `he` Hebrew, via the Hebrew text normaliser and RenikudPlus.
/// - `en` English, `es` Spanish (Castilian), `de` German and `it` Italian via
///   polyg2p, the G2P the model's labels were made with.
/// - `ru` Russian, via polyg2p's Russian (the site runs its own port of
///   Russian-G2P with RUAccent models instead, which the app does not ship).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Language {
    Hebrew,
    English,
    Spanish,
    German,
    Italian,
    Russian,
}

impl Language {
    pub const ALL: [Language; 6] = [
        Self::Hebrew,
        Self::English,
        Self::Spanish,
        Self::German,
        Self::Italian,
        Self::Russian,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::Hebrew => "he",
            Self::English => "en",
            Self::Spanish => "es",
            Self::German => "de",
            Self::Italian => "it",
            Self::Russian => "ru",
        }
    }
}

impl TryFrom<&str> for Language {
    type Error = anyhow::Error;

    /// Codes are read case-insensitively (`EN`, `en-US`), as the
    /// reference's `_canonical_lang` reads them.
    fn try_from(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "he" => Ok(Self::Hebrew),
            "en" | "en-us" => Ok(Self::English),
            "es" => Ok(Self::Spanish),
            "de" | "ge" => Ok(Self::German),
            "it" => Ok(Self::Italian),
            "ru" | "ru-ru" => Ok(Self::Russian),
            _ => bail!("unsupported language code `{value}`; expected he, en, es, de, it, or ru"),
        }
    }
}

/// polyg2p data the release's labels were made with. English reads from its
/// lexicon and the large out-of-vocabulary model, German and Italian from
/// their lexicons, Russian from its stress and ё dictionaries; Spanish is
/// rules alone. `en.g2p` is the small model Russian uses for Latin words.
pub const POLYG2P_DATA_FILES: &[&str] = &[
    "en.lex",
    "en-large.g2p",
    "en.g2p",
    "de.lex",
    "it.lex",
    "ru_accents.lex",
    "ru_homographs.lex",
    "ru_suffix.lex",
    "ru_yo.lex",
    "ru_yo_homographs.lex",
];

fn open_polyg2p(data_dir: Option<&Path>) -> Result<polyg2p::G2p> {
    if let Some(dir) = data_dir {
        let missing: Vec<&str> = POLYG2P_DATA_FILES
            .iter()
            .copied()
            .filter(|name| !dir.join(name).is_file())
            .collect();
        if !missing.is_empty() {
            bail!(
                "polyg2p data is missing from {}: {}",
                dir.display(),
                missing.join(", ")
            );
        }
    }
    // polyg2p picks its English out-of-vocabulary model from this variable
    // when English first loads, and the release's labels used the large one.
    // Set once per process, and never over a value the user chose.
    static SET_EN_MODEL: Once = Once::new();
    SET_EN_MODEL.call_once(|| {
        if let Some(model) = en_model_to_set(std::env::var_os(EN_MODEL_VAR)) {
            // SAFETY: runs once, while the first phonemizer is being built
            // and before any polyg2p call can read the variable; nothing
            // else in this process writes it.
            unsafe { std::env::set_var(EN_MODEL_VAR, model) };
        }
    });
    Ok(polyg2p::G2p::new(data_dir))
}

const EN_MODEL_VAR: &str = "POLYG2P_EN_MODEL";

/// The English model to put in [`EN_MODEL_VAR`]: the release's large one,
/// unless the variable already holds the user's own choice.
fn en_model_to_set(current: Option<std::ffi::OsString>) -> Option<&'static str> {
    match current {
        Some(value) if !value.is_empty() => None,
        _ => Some("large"),
    }
}


/// The languages the site's polyg2p reads (`POLY_LANGS`).
const POLY_LANGS: [&str; 4] = ["en", "de", "es", "it"];

fn poly_lang(code: &str) -> Option<polyg2p::Lang> {
    match code {
        "en" => Some(polyg2p::Lang::En),
        "de" => Some(polyg2p::Lang::De),
        "es" => Some(polyg2p::Lang::Es),
        "it" => Some(polyg2p::Lang::It),
        "ru" => Some(polyg2p::Lang::Ru),
        _ => None,
    }
}

/// The site's `LANG_TAG_RE` (`/<\/?\w+>/g`, ASCII `\w`).
pub(crate) fn strip_lang_tags(text: &str) -> String {
    regex!(r"</?[A-Za-z0-9_]+>").replace_all(text, "").into_owned()
}

pub struct Phonemizer {
    hebrew: Option<G2P>,
    poly: polyg2p::G2p,
    language: Language,
    /// RenikudPlus speaker conditioning (0=unknown, 1=male, 2=female).
    speaker: u8,
    target_speaker: u8,
    /// (language, input, output) of every native G2P call, when tracing.
    trace: Option<Vec<(String, String, String)>>,
    /// The bot rules of BlueV3 `site_flow.py` before and after
    /// `prepareText` ([`crate::handling::prepare`]); on by default.
    bot_rules: bool,
    /// Read niqqud in Hebrew input as constraints rather than drop it (the
    /// site's and the reference's reading); off by default.
    honor_niqqud: bool,
}

impl Phonemizer {
    /// Create a phonemizer with Hebrew as the default language and no
    /// polyg2p data: see [`Self::open`] for the front end a bundle ships.
    ///
    /// `renikud_model` is only required when phonemizing Hebrew text (RenikudPlus ONNX).
    pub fn new(renikud_model: Option<impl AsRef<Path>>) -> Result<Self> {
        Self::with_language(renikud_model, Language::Hebrew)
    }

    /// Create a phonemizer with an explicit default language and no polyg2p
    /// data directory. polyg2p then reads with its built-in rules only, which
    /// is exact for Spanish and well short of the release for the rest.
    pub fn with_language(
        renikud_model: Option<impl AsRef<Path>>,
        language: Language,
    ) -> Result<Self> {
        Self::open(renikud_model, None::<&Path>, language)
    }

    /// Create the release's front end: RenikudPlus for Hebrew and polyg2p with
    /// the bundle's data directory for the other languages.
    ///
    /// Every file in [`POLYG2P_DATA_FILES`] has to be in `polyg2p_data`. A
    /// missing lexicon would not fail a request, it would quietly read every
    /// word by rule, so it fails here instead.
    pub fn open(
        renikud_model: Option<impl AsRef<Path>>,
        polyg2p_data: Option<impl AsRef<Path>>,
        language: Language,
    ) -> Result<Self> {
        let hebrew = match renikud_model {
            Some(path) => Some(G2P::with_config(
                path.as_ref().to_string_lossy().as_ref(),
                hebrew_config(),
            )?),
            None => None,
        };
        Self::from_parts(
            hebrew,
            open_polyg2p(polyg2p_data.as_ref().map(AsRef::as_ref))?,
            language,
        )
    }

    /// Create a phonemizer from embedded RenikudPlus ONNX bytes.
    pub fn from_renikud_bytes(bytes: &[u8], language: Language) -> Result<Self> {
        let builder = Session::builder()?;
        let session = builder.commit_from_memory(bytes)?;
        Self::from_parts(
            Some(G2P::from_session_with_config(session, hebrew_config())?),
            open_polyg2p(None)?,
            language,
        )
    }

    fn from_parts(hebrew: Option<G2P>, poly: polyg2p::G2p, language: Language) -> Result<Self> {
        Ok(Self {
            hebrew,
            poly,
            language,
            speaker: 0,
            target_speaker: 0,
            trace: None,
            bot_rules: true,
            honor_niqqud: false,
        })
    }

    /// Turn the bot rules ([`crate::handling::prepare`]) off to read text
    /// exactly as the BlueTTS site does, or back on (the default).
    pub fn set_bot_rules(&mut self, on: bool) {
        self.bot_rules = on;
    }

    /// Read the points of pointed Hebrew as constraints (`true`), or drop
    /// them and read the skeleton as the site and the reference do (`false`,
    /// the default). Only text whose points someone chose on purpose, such
    /// as the app's niqqud editor's, should be read with them.
    pub fn set_honor_niqqud(&mut self, on: bool) {
        self.honor_niqqud = on;
    }

    /// The text the G2P reads: [`crate::handling::prepare`] with this
    /// phonemizer's bot rules.
    pub fn prepare(&self, text: &str, language: Language) -> String {
        prepare(text, language.code(), self.bot_rules)
    }

    /// Set RenikudPlus speaker conditioning (0=unknown, 1=male, 2=female).
    /// The site decodes every voice with 0, the default.
    pub fn set_speakers(&mut self, speaker: u8, target_speaker: u8) {
        self.speaker = speaker;
        self.target_speaker = target_speaker;
    }

    /// Record every Hebrew and Russian G2P call (for parity tests against
    /// the site's front end, which has its own copies of those models).
    #[doc(hidden)]
    pub fn set_native_trace(&mut self, on: bool) {
        self.trace = on.then(Vec::new);
    }

    /// The (language, input, output) native calls recorded so far.
    #[doc(hidden)]
    pub fn take_native_trace(&mut self) -> Vec<(String, String, String)> {
        self.trace.as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// Add niqqud to the Hebrew in `text` with RenikudPlus.
    ///
    /// Non-Hebrew text and punctuation are passed through; digits are read as
    /// Hebrew number words first. With `with_stress`, each stressed syllable
    /// also gets the hatama (U+05AB), which [`Self::g2p`] reads back, so the
    /// pointed text phonemizes the same as the original. The whole text is
    /// read in one pass, as [`Self::g2p`] reads it, because the model's
    /// reading of a word depends on its context.
    pub fn diacritize(&mut self, text: &str, with_stress: bool) -> Result<String> {
        if !contains_hebrew(text) {
            return Ok(text.to_owned());
        }
        let Some(g2p) = self.hebrew.as_mut() else {
            bail!("Hebrew diacritization needs a RenikudPlus model path");
        };
        g2p.vocalize_with(
            text,
            &Options {
                with_stress,
                // The text goes back to the user to edit and then returns
                // through `g2p`, so it has to read back as what it came from.
                round_trip: true,
                ..Options::speakers(self.speaker, self.target_speaker)
            },
        )
    }

    /// Install a force lexicon over the Hebrew G2P: a `surface<TAB>IPA` file
    /// whose entries always win where they match.
    ///
    /// `align_cells` is `""` for the standard aligner or `"extended"` to widen
    /// the validation the entries have to pass.
    pub fn set_hebrew_lexicon(&mut self, path: impl AsRef<Path>, align_cells: &str) -> Result<()> {
        let Some(g2p) = self.hebrew.as_mut() else {
            bail!("a Hebrew force lexicon needs a RenikudPlus model path");
        };
        let cells = G2P::align_cells(align_cells)?;
        let lexicon = ForceLexicon::from_tsv(path, &cells, OnInvalidEntry::Report)?;
        for (surface, ipa, reason) in &lexicon.rejected {
            eprintln!("WARNING: lexicon entry {surface:?} -> {ipa:?} rejected: {reason}");
        }
        g2p.set_lexicon(Some(lexicon));
        Ok(())
    }

    /// Phonemize text using the default language.
    pub fn phonemize(&mut self, text: &str) -> Result<String> {
        self.phonemize_lang(text, self.language)
    }

    /// Phonemize text using an explicit supported model language.
    pub fn phonemize_lang(&mut self, text: &str, language: Language) -> Result<String> {
        self.g2p(text, language)
    }

    /// The site's front end over raw text: `prepareText`, then G2P over the
    /// whole prepared text (Russian a sentence at a time), returned wrapped
    /// once in the request's language tag (`<he>…</he>`), the form
    /// [`crate::BlueTts::create`] and the language detection of the callers
    /// read. Empty when there is nothing to phonemize.
    pub fn g2p(&mut self, text: &str, language: Language) -> Result<String> {
        let prepared = self.prepare(text, language);
        let ipa = self
            .phonemize_units(&prepared, language)?
            .into_iter()
            .filter(|unit| !unit.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if ipa.is_empty() {
            return Ok(String::new());
        }
        let code = language.code();
        Ok(format!("<{code}>{ipa}</{code}>"))
    }

    /// The G2P units of prepared text, phonemized: the whole text, or for
    /// Russian a sentence each (the site's `makeChunker` units).
    pub fn phonemize_units(
        &mut self,
        prepared: &str,
        language: Language,
    ) -> Result<Vec<String>> {
        let units = if language == Language::Russian {
            split_sentences(prepared)
        } else {
            vec![prepared.to_owned()]
        };
        units
            .iter()
            .map(|unit| self.phonemize_prepared(unit, language.code()))
            .collect()
    }

    /// `Text.phonemize(text, lang, g2p)` with a release bundle: Hebrew text
    /// first gets its Latin runs tagged ([`tag_latin_runs`]). With inline
    /// `<xx>…</xx>` spans (`_phonemize_mixed_release`, BlueTTS a6cc157) each
    /// piece is read by its own language's G2P and given that language's IPA
    /// view without a final `.`; a piece's `.` goes too when it is not the
    /// last and its source ends in a word character; pieces with no letter or
    /// digit stay as written, without double quotes. The result is one
    /// tag-free IPA string.
    pub(crate) fn phonemize_prepared(&mut self, text: &str, lang: &str) -> Result<String> {
        let lang = canonical_lang(lang);
        let text = tag_latin_runs(text, &lang);
        let clean = |s: &str| -> String {
            let s = regex!(r"</?[A-Za-z0-9_]+>").replace_all(s, " ");
            js_trim(&js_collapse_spaces(&s)).to_owned()
        };
        let spans = inline_lang_spans(&text);
        if spans.is_empty() {
            return Ok(clean(&self.phonemize_segment(&text, &lang)?));
        }
        let mut segs: Vec<(String, &str)> = Vec::new();
        let mut last = 0;
        for span in &spans {
            segs.push((lang.clone(), &text[last..span.start]));
            segs.push((canonical_lang(&span.name), &text[span.content.0..span.content.1]));
            last = span.end;
        }
        segs.push((lang.clone(), &text[last..]));
        let live: Vec<(String, &str)> = segs
            .into_iter()
            .map(|(lg, t)| (lg, js_trim(t)))
            .filter(|(_, t)| !t.is_empty())
            .collect();
        let alnum = regex!(r"[\p{L}\p{N}]");
        let double_quotes = regex!(r#"["“”„]"#);
        let mut parts = Vec::new();
        for (k, (lg, src)) in live.iter().enumerate() {
            if !alnum.is_match(src) {
                parts.push(double_quotes.replace_all(src, "").into_owned());
                continue;
            }
            let ipa = strip_lang_tags(&self.phonemize_segment(src, lg)?);
            let mut ipa = normalize_ipa_with(&ipa, lg, false);
            if k + 1 < live.len()
                && ipa.ends_with('.')
                && !ipa.ends_with("..")
                && src.chars().next_back().is_some_and(is_py_word_char)
            {
                ipa.pop();
                ipa = ipa.trim_end_matches(is_js_space).to_owned();
            }
            parts.push(ipa);
        }
        let joined = parts
            .into_iter()
            .filter(|p| !js_trim(p).is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(clean(&joined))
    }

    /// `phonemizeSegment` with the release G2Ps.
    fn phonemize_segment(&mut self, content: &str, lang: &str) -> Result<String> {
        let content = strip_emoji(js_trim(&strip_lang_tags(content)));
        if content.is_empty() {
            return Ok(String::new());
        }
        let has_hebrew = contains_hebrew(&content);
        if has_hebrew || lang == "he" {
            if !has_hebrew {
                return Ok(content);
            }
            return self.native(&content, "he");
        }
        if lang == "ru" || contains_cyrillic(&content) {
            return self.native(&content, "ru");
        }
        if POLY_LANGS.contains(&lang) {
            // _release_g2p: bracketed tags out; a piece with no letters or
            // digits passes as its marks.
            let t = bracketed_spans().replace_all(&content, " ");
            let t = js_trim(&t);
            if !regex!(r"[\p{L}\p{N}]").is_match(t) {
                return Ok(t.to_owned());
            }
            let ipa = self
                .poly
                .phonemize(t, poly_lang(lang).expect("a polyg2p language"));
            return Ok(ipa);
        }
        Ok(content)
    }

    /// The site's `g2p.native` (js/g2p/index.js `phonemizeSegment`): Hebrew
    /// through RenikudPlus as it is, Russian through its G2P and then
    /// `stripSilentSeparatorTokens`.
    fn native(&mut self, text: &str, lang: &str) -> Result<String> {
        let content = regex!(r"(?i)<lang_list\b[^>]*>[\s\S]*?</lang_list>").replace_all(text, " ");
        let content = regex!(r"(?i)</?lang_list\b[^>]*>").replace_all(&content, " ");
        let content = regex!(r"</?[^>]+>").replace_all(&content, " ");
        let content = js_trim(&content).to_owned();
        if content.is_empty() {
            return Ok(String::new());
        }
        let out = if lang == "he" {
            self.phonemize_renikud(&content)?
        } else {
            strip_silent_separator_tokens(&self.poly.phonemize(&content, polyg2p::Lang::Ru))
        };
        if let Some(trace) = self.trace.as_mut() {
            trace.push((lang.to_owned(), content, out.clone()));
        }
        Ok(out)
    }

    fn phonemize_renikud(&mut self, text: &str) -> Result<String> {
        // The site's phonemizeHebrew: the points go first, and what is left
        // without a Hebrew character comes back as it is.
        let stripped;
        let text = if self.honor_niqqud {
            text
        } else {
            stripped = renikud_plus_rs::text::strip_niqqud(text);
            stripped.as_str()
        };
        if !contains_hebrew(text) {
            return Ok(text.to_owned());
        }
        let Some(g2p) = self.hebrew.as_mut() else {
            bail!("Hebrew phonemization needs a RenikudPlus model path");
        };
        g2p.phonemize_with(
            text,
            &Options {
                niqqud: Some(if self.honor_niqqud {
                    NiqqudMode::Use
                } else {
                    NiqqudMode::Strip
                }),
                ..Options::speakers(self.speaker, self.target_speaker)
            },
        )
    }
}

pub fn phonemize(text: &str, renikud_model: Option<impl AsRef<Path>>) -> Result<String> {
    let mut phonemizer = Phonemizer::new(renikud_model)?;
    phonemizer.phonemize(text)
}

fn bracketed_spans() -> &'static regex::Regex {
    regex!(&format!(
        r"{}*\[[^\]]*\]{}*",
        crate::handling::JS_S,
        crate::handling::JS_S
    ))
}

fn contains_cyrillic(text: &str) -> bool {
    text.chars().any(|c| ('\u{0400}'..='\u{04ff}').contains(&c))
}

fn contains_hebrew(text: &str) -> bool {
    text.chars().any(|c| ('\u{0590}'..='\u{05ff}').contains(&c))
}

#[cfg(test)]
mod polyg2p_tests {
    use super::*;

    /// The release front end over an installed bundle, for the tests that
    /// need its lexicons: `MAMBOTTS_BLUE_MODEL_DIR=<bundle> cargo test -- --ignored`.
    fn bundle_phonemizer() -> Phonemizer {
        let dir = std::path::PathBuf::from(
            std::env::var("MAMBOTTS_BLUE_MODEL_DIR").expect("set MAMBOTTS_BLUE_MODEL_DIR"),
        );
        Phonemizer::open(None::<&Path>, Some(dir.join("polyg2p")), Language::English).unwrap()
    }

    /// Spanish is rules alone in polyg2p, so it reads the same with no data
    /// directory: word breaks and punctuation survive, numbers are read.
    #[test]
    fn spanish_keeps_word_breaks_and_punctuation() {
        let mut phonemizer = Phonemizer::with_language(None::<&str>, Language::Spanish).unwrap();
        let ipa = phonemizer
            .g2p("Hola, mundo. ¡Adiós! Tengo 3 gatos.", Language::Spanish)
            .unwrap();
        assert_eq!(ipa, "<es>ˈola, mˈundo. aðjˈos! tˈɛŋɡo tɾˈes ɣˈatos.</es>");
    }

    #[test]
    fn bracketed_spans_are_dropped_before_g2p() {
        let mut phonemizer = Phonemizer::with_language(None::<&str>, Language::Spanish).unwrap();
        let plain = phonemizer.g2p("Hola, mundo.", Language::Spanish).unwrap();
        let tagged = phonemizer
            .g2p("Hola, [laughs] mundo.", Language::Spanish)
            .unwrap();
        assert_eq!(tagged, plain);
        // A piece with no letter or digit passes through as its punctuation.
        assert_eq!(phonemizer.phonemize_segment(" ... ", "es").unwrap(), "...");
    }

    #[test]
    fn language_codes_and_inline_tags_are_read_as_the_site_reads_them() {
        assert_eq!(Language::try_from("EN").unwrap(), Language::English);
        assert_eq!(Language::try_from("He").unwrap(), Language::Hebrew);
        assert_eq!(Language::try_from("en-US").unwrap(), Language::English);
        assert_eq!(Language::try_from("GE").unwrap(), Language::German);
        let mut phonemizer = Phonemizer::with_language(None::<&str>, Language::Spanish).unwrap();
        let plain = phonemizer.g2p("Hola mundo.", Language::Spanish).unwrap();
        let tagged = phonemizer
            .g2p("<ES>Hola</ES> mundo.", Language::Spanish)
            .unwrap();
        assert_eq!(tagged, plain);
        // A tag that names no language is dropped (a bot rule)...
        let ipa = phonemizer.g2p("<b>Hola</b> mundo.", Language::Spanish).unwrap();
        assert_eq!(ipa, plain);
        // ...where on the site any `<x>…</x>` pair is a language span, and
        // one in a language it has no G2P for passes through as written.
        phonemizer.set_bot_rules(false);
        let ipa = phonemizer.g2p("<b>Hola</b> mundo.", Language::Spanish).unwrap();
        assert_eq!(ipa, "<es>Hola mˈundo.</es>");
    }

    #[test]
    fn latin_text_in_a_hebrew_request_is_read_as_english_span_by_span() {
        let mut phonemizer = Phonemizer::with_language(None::<&str>, Language::Hebrew).unwrap();
        let ipa = phonemizer.phonemize_prepared("hello PDF, world.", "he").unwrap();
        // Each run through English G2P without its '.', the marks between
        // and after them kept as written.
        assert!(!ipa.contains("hello") && !ipa.contains('<'), "{ipa}");
        assert!(ipa.contains(" , ") && ipa.ends_with(" ."), "{ipa}");
    }

    #[test]
    fn the_english_model_is_never_chosen_over_the_user() {
        assert_eq!(en_model_to_set(None), Some("large"));
        assert_eq!(en_model_to_set(Some("".into())), Some("large"));
        assert_eq!(en_model_to_set(Some("small".into())), None);
    }

    #[test]
    fn a_bundle_with_a_missing_lexicon_fails_at_load() {
        let dir = std::env::temp_dir().join("mambotts-empty-polyg2p");
        std::fs::create_dir_all(&dir).unwrap();
        let error = Phonemizer::open(None::<&Path>, Some(&dir), Language::English)
            .err()
            .expect("an empty data directory is not a front end")
            .to_string();
        assert!(error.contains("en.lex"), "{error}");
    }

    #[test]
    #[ignore = "requires MAMBOTTS_BLUE_MODEL_DIR pointing to an installed BlueV3 bundle"]
    fn every_polyg2p_language_reads_text_as_written() {
        let mut phonemizer = bundle_phonemizer();
        for (language, text, expected) in [
            (
                Language::English,
                "Hello, world. Goodbye!",
                "<en>həlˈoʊ, wˈɜːld. ɡʊdbˈaɪ!</en>",
            ),
            (
                Language::English,
                "Dr. Smith paid $25 on March 3rd, 2024.",
                "<en>dˈɑːktɚ smˈɪθ pˈeɪd twˈɛnti fˈaɪv dˈɑːlɚz ˌɔn mˈɑːɹʧ θˈɜːd, twˈɛnti twˈɛnti fˈɔːɹ.</en>",
            ),
            (
                Language::German,
                "Hallo, Welt. Tschüss!",
                "<de>halˈoː, vˈɛlt. ʧˈʏs!</de>",
            ),
            (
                Language::Italian,
                "Ciao, mondo. Arrivederci!",
                "<it>ʧˈao, mˈondo. arɾivedˈerʧɪ!</it>",
            ),
            (
                Language::Russian,
                "Всё хорошо, дверь закрыта.",
                "<ru>fsʲˈɵ xərɐʂˈo, dvʲerʲ zɐkrˈɨtə.</ru>",
            ),
        ] {
            let ipa = phonemizer.g2p(text, language).unwrap();
            assert_eq!(ipa, expected, "{}", language.code());
        }
    }

    #[test]
    #[ignore = "requires MAMBOTTS_BLUE_MODEL_DIR pointing to an installed BlueV3 bundle"]
    fn cyrillic_is_russian_whatever_language_was_asked_for() {
        let mut phonemizer = bundle_phonemizer();
        let russian = phonemizer.g2p("дверь", Language::Russian).unwrap();
        let routed = phonemizer.g2p("дверь", Language::English).unwrap();
        assert_eq!(strip_lang_tags(&routed), strip_lang_tags(&russian));
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    #[ignore = "requires MAMBOTTS_RENIKUD_PATH pointing to renikud-plus.onnx"]
    fn user_nikud_with_real_model() {
        let model = std::env::var("MAMBOTTS_RENIKUD_PATH").expect("set MAMBOTTS_RENIKUD_PATH");
        let mut phonemizer = Phonemizer::new(Some(model)).unwrap();
        // By default points are dropped, as on the site: pointed and bare
        // text read the same.
        assert_eq!(
            phonemizer.g2p("שָׁלוֹם עוֹלָם", Language::Hebrew).unwrap(),
            phonemizer.g2p("שלום עולם", Language::Hebrew).unwrap()
        );
        // The niqqud editor's text is read with its points.
        phonemizer.set_honor_niqqud(true);
        // MamboTTS's own letter-label reading (ב׳ as "bet") is gone with the
        // rest of its Hebrew pass: the site reads ב׳ as RenikudPlus does.
        for (text, expected) in [("אֶן", "en"), ("קֶלְוִין", "kelvin")] {
            let ipa = phonemizer.g2p(text, Language::Hebrew).unwrap();
            println!("{text}: {ipa}");
            assert!(ipa.replace('ˈ', "").contains(expected), "{text}: {ipa}");
        }
        let plain = phonemizer.g2p("שלום", Language::Hebrew).unwrap();
        let mixed = phonemizer.g2p("אֶן שלום קֶלְוִין", Language::Hebrew).unwrap();
        println!("plain: {plain}; mixed: {mixed}");
        assert!(mixed.replace('ˈ', "").contains("en"));
        assert!(mixed.replace('ˈ', "").contains("kelvin"));
        assert!(!mixed.chars().any(|c| ('א'..='ת').contains(&c)));
        assert!(mixed.contains(plain.trim_start_matches("<he>").trim_end_matches("</he>")));

        let text = "שבת שלום, hello!\nספר";
        let nikud = phonemizer.diacritize(text, true).unwrap();
        println!("diacritized: {nikud}");
        assert!(nikud.contains(", hello!\n"), "{nikud}");
        // A lone letter with a geresh (ב׳) no longer round-trips: MamboTTS
        // used to spell it out as a pointed letter name before G2P, and the
        // site does not.
        for text in [text, "צה\"ל הודיע שהתשובה נכונה.", "ג׳אז וצ׳יפס, רוח ותקווה."] {
            let nikud = phonemizer.diacritize(text, true).unwrap();
            let original = phonemizer.g2p(text, Language::Hebrew).unwrap();
            let round_trip = phonemizer.g2p(&nikud, Language::Hebrew).unwrap();
            println!("{text}\n  {nikud}\n  {original}\n  {round_trip}");
            // The points pin the reading, stress included, so phonemizing the
            // pointed text gives back what the plain text gave.
            assert_eq!(
                round_trip, original,
                "{text} -> {nikud}: {original} vs {round_trip}"
            );
        }
    }
}
