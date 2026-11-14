use std::path::Path;
use std::sync::Once;

use anyhow::{Result, bail};
use ort::session::Session;
use regex::Regex;
use renikud_plus_rs::lexicon::{ForceLexicon, OnInvalidEntry};
use renikud_plus_rs::{G2P, G2PConfig, NiqqudMode, Options};

use crate::handling::{prepare_text_for_synthesis, strip_reference_code_markers};

/// RenikudPlus as MamboTTS runs it: niqqud in the input is read as evidence
/// rather than dropped, because the app lets people type and edit the points
/// and expects them honored.
fn hebrew_config() -> G2PConfig {
    G2PConfig {
        niqqud: NiqqudMode::Use,
        ..G2PConfig::default()
    }
}

/// Languages the BlueV3 release reads.
///
/// Codes:
/// - `he` Hebrew, via the Hebrew text normaliser and RenikudPlus.
/// - `en` English, `es` Spanish (Castilian), `de` German, `it` Italian and
///   `ru` Russian, via polyg2p, the G2P the model's labels were made with.
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

    /// The polyg2p language that reads this one; Hebrew has its own front end.
    fn polyg2p(self) -> Option<polyg2p::Lang> {
        match self {
            Self::Hebrew => None,
            Self::English => Some(polyg2p::Lang::En),
            Self::Spanish => Some(polyg2p::Lang::Es),
            Self::German => Some(polyg2p::Lang::De),
            Self::Italian => Some(polyg2p::Lang::It),
            Self::Russian => Some(polyg2p::Lang::Ru),
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

const MIXED_LATIN: &str = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\d+[A-Za-z]+|[A-Za-z]+(?:[.'’\-][A-Za-z0-9]+)*";
const INLINE_TAG: &str =
    r"(?is)<(en|en-us|he|es|de|ge|it|ru)>(.*?)</(?:en|en-us|he|es|de|ge|it|ru)>";

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

pub struct Phonemizer {
    hebrew: Option<G2P>,
    poly: polyg2p::G2p,
    language: Language,
    mixed_re: Regex,
    inline_tag_re: Regex,
    /// RenikudPlus speaker conditioning (0=unknown, 1=male, 2=female).
    speaker: u8,
    target_speaker: u8,
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
            mixed_re: Regex::new(MIXED_LATIN)?,
            inline_tag_re: Regex::new(INLINE_TAG)?,
            speaker: 0,
            target_speaker: 0,
        })
    }

    /// Set RenikudPlus speaker conditioning (0=unknown, 1=male, 2=female).
    pub fn set_speakers(&mut self, speaker: u8, target_speaker: u8) {
        self.speaker = speaker;
        self.target_speaker = target_speaker;
    }

    /// Add niqqud to the Hebrew in `text` with RenikudPlus.
    ///
    /// Non-Hebrew text and punctuation are passed through; digits are read as
    /// Hebrew number words first, as they are before synthesis. With
    /// `with_stress`, each stressed syllable also gets the hatama (U+05AB),
    /// which [`Self::g2p`] reads back, so the pointed text phonemizes the same
    /// as the original. The whole text is read in one pass, as [`Self::g2p`]
    /// reads it, because the model's reading of a word depends on its context.
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
    ///
    /// For mixed Hebrew/Latin input, Hebrew spans use RenikudPlus and Latin
    /// spans are read as English.
    pub fn phonemize(&mut self, text: &str) -> Result<String> {
        self.phonemize_lang(text, self.language)
    }

    /// Prepare raw text and return BlueTTS-ready, language-tagged IPA.
    pub fn g2p(&mut self, text: &str, language: Language) -> Result<String> {
        // The slow-segment markers pace synthesis; they are not phonemes, and
        // phonemes handed back here can be synthesized again.
        let prepared =
            strip_reference_code_markers(&prepare_text_for_synthesis(text, language.code()));
        self.phonemize_prepared(&prepared, language)
    }

    /// Phonemize text using an explicit supported model language.
    pub fn phonemize_lang(&mut self, text: &str, language: Language) -> Result<String> {
        self.g2p(text, language)
    }

    fn phonemize_prepared(&mut self, text: &str, language: Language) -> Result<String> {
        if !self.inline_tag_re.is_match(text) {
            let segments = self.phonemize_segments(text, language)?;
            return Ok(self.wrap_segments(segments));
        }

        let mut segments = Vec::new();
        let mut last = 0;
        let tags: Vec<(usize, usize, Language, String)> = self
            .inline_tag_re
            .captures_iter(text)
            .map(|caps| {
                let all = caps.get(0).expect("full tag match");
                let language = Language::try_from(caps.get(1).expect("tag language").as_str())?;
                Ok((
                    all.start(),
                    all.end(),
                    language,
                    caps.get(2).expect("tag content").as_str().to_owned(),
                ))
            })
            .collect::<Result<_>>()?;
        for (start, end, tagged_language, content) in tags {
            if start > last {
                segments.extend(self.phonemize_segments(&text[last..start], language)?);
            }
            segments.extend(self.phonemize_segments(&content, tagged_language)?);
            last = end;
        }
        if last < text.len() {
            segments.extend(self.phonemize_segments(&text[last..], language)?);
        }
        Ok(self.wrap_segments(segments))
    }

    fn phonemize_segments(
        &mut self,
        text: &str,
        language: Language,
    ) -> Result<Vec<(Language, String)>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        if language != Language::Hebrew && !contains_hebrew(text) {
            let ipa = self.phonemize_polyg2p(text, language);
            return Ok((!ipa.is_empty())
                .then_some((language, ipa))
                .into_iter()
                .collect());
        }
        if language != Language::Hebrew || !contains_latin_or_digit(text) {
            let ipa = self.phonemize_renikud(text)?;
            return Ok((!ipa.is_empty())
                .then_some((language, ipa))
                .into_iter()
                .collect());
        }

        let spans: Vec<(usize, usize)> = self
            .mixed_re
            .find_iter(text)
            .map(|m| (m.start(), m.end()))
            .collect();
        let mut result = Vec::new();
        let mut last = 0;
        for (start, end) in spans {
            if start > last {
                self.push_segment(&mut result, &text[last..start], Language::Hebrew)?;
            }
            let latin = &text[start..end];
            let latin = if is_email(latin) {
                email_to_spoken_english(latin)
            } else {
                latin.to_owned()
            };
            self.push_segment(&mut result, &latin, Language::English)?;
            last = end;
        }
        if last < text.len() {
            self.push_segment(&mut result, &text[last..], Language::Hebrew)?;
        }
        Ok(result)
    }

    fn push_segment(
        &mut self,
        output: &mut Vec<(Language, String)>,
        text: &str,
        language: Language,
    ) -> Result<()> {
        let ipa = if language == Language::Hebrew {
            self.phonemize_renikud(text)?
        } else {
            self.phonemize_polyg2p(text, language)
        };
        if !ipa.trim().is_empty() {
            output.push((language, ipa));
        }
        Ok(())
    }

    fn wrap_segments(&self, segments: Vec<(Language, String)>) -> String {
        wrap_segments(segments)
    }

    /// polyg2p's reading of `text`, as the release calls it: bracketed spans
    /// such as `[laughs]` are dropped first, and a piece with no letter or
    /// digit passes through as its punctuation. Cyrillic is Russian whatever
    /// language was asked for.
    fn phonemize_polyg2p(&self, text: &str, language: Language) -> String {
        let text = bracketed_spans().replace_all(text, " ");
        let text = text.trim();
        if !text.chars().any(char::is_alphanumeric) {
            return text.to_owned();
        }
        let language = if contains_cyrillic(text) {
            Language::Russian
        } else {
            language
        };
        let lang = language.polyg2p().unwrap_or(polyg2p::Lang::En);
        self.poly.phonemize(text, lang)
    }

    fn phonemize_renikud(&mut self, text: &str) -> Result<String> {
        if !contains_hebrew(text) {
            return Ok(text.to_owned());
        }
        let Some(g2p) = self.hebrew.as_mut() else {
            bail!("Hebrew phonemization needs a RenikudPlus model path");
        };
        let speaker = self.speaker;
        let target_speaker = self.target_speaker;
        g2p.phonemize(text, speaker, target_speaker)
    }
}

pub fn phonemize(text: &str, renikud_model: Option<impl AsRef<Path>>) -> Result<String> {
    let mut phonemizer = Phonemizer::new(renikud_model)?;
    phonemizer.phonemize(text)
}

/// Join language segments into one tagged string. A change of language is
/// only a word break: the text's own punctuation is the only pause, so a
/// segment that starts with punctuation, or is nothing but punctuation (the
/// period after a Latin word), attaches to what came before it.
fn wrap_segments(segments: Vec<(Language, String)>) -> String {
    let mut merged: Vec<(Language, String)> = Vec::new();
    for (language, ipa) in segments {
        let ipa = normalize_spaces(&ipa);
        if ipa.is_empty() {
            continue;
        }
        match merged.last_mut() {
            Some(last) if !ipa.chars().any(char::is_alphanumeric) => last.1.push_str(&ipa),
            Some(last) if last.0 == language => {
                if !starts_with_punctuation(&ipa) {
                    last.1.push(' ');
                }
                last.1.push_str(&ipa);
            }
            _ => merged.push((language, ipa)),
        }
    }
    let mut result = String::new();
    for (language, ipa) in merged {
        if !result.is_empty() && !starts_with_punctuation(&ipa) {
            result.push(' ');
        }
        let code = language.code();
        result.push_str(&format!("<{code}>{ipa}</{code}>"));
    }
    result
}

fn starts_with_punctuation(text: &str) -> bool {
    text.starts_with(['.', ',', '!', '?', ';', ':', ')', ']', '}'])
}

fn bracketed_spans() -> &'static Regex {
    regex!(r"\s*\[[^\]]*\]\s*")
}

fn contains_cyrillic(text: &str) -> bool {
    text.chars().any(|c| ('\u{0400}'..='\u{04ff}').contains(&c))
}

fn contains_hebrew(text: &str) -> bool {
    text.chars().any(|c| ('\u{0590}'..='\u{05ff}').contains(&c))
}

fn normalize_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn contains_latin_or_digit(text: &str) -> bool {
    text.chars().any(|c| c.is_ascii_alphanumeric())
}

fn is_email(text: &str) -> bool {
    regex!(r"(?i)^[A-Z0-9._%+\-]+@[A-Z0-9.-]+\.[A-Z]{2,}$").is_match(text)
}

fn email_to_spoken_english(email: &str) -> String {
    let (local, domain) = email.split_once('@').unwrap_or((email, ""));
    let local = local
        .replace(['.', '_'], " dot ")
        .replace('-', " dash ")
        .replace('+', " plus ");
    let domain = domain
        .split('.')
        .filter(|part| !part.is_empty())
        .map(|part| {
            if part.len() <= 2 && part.chars().all(|c| c.is_ascii_alphabetic()) {
                part.chars()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                part.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" dot ");
    format!("{local} at {domain}")
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

    /// A change of language is a word break, not a pause: no comma between
    /// segments, and a trailing period attaches to the last word.
    /// The BlueV3 reference inserts "," at a switch; MamboTTS deliberately does not.
    #[test]
    fn joins_language_segments_with_a_space() {
        let he = |ipa: &str| (Language::Hebrew, ipa.to_owned());
        let en = |ipa: &str| (Language::English, ipa.to_owned());
        for (segments, expected) in [
            (
                vec![he("pʁatˈim mˈeha"), en("pˌiːdˌiːˈɛf"), he("hametsuʁˈaf.")],
                "<he>pʁatˈim mˈeha</he> <en>pˌiːdˌiːˈɛf</en> <he>hametsuʁˈaf.</he>",
            ),
            (
                vec![he("kitˈa ʔˈalef, mˈeha"), en("pˌiːdˌiːˈɛf"), he(".")],
                "<he>kitˈa ʔˈalef, mˈeha</he> <en>pˌiːdˌiːˈɛf.</en>",
            ),
            (
                vec![en("pˌiːdˌiːˈɛf"), he(", ʃalˈom")],
                "<en>pˌiːdˌiːˈɛf</en><he>, ʃalˈom</he>",
            ),
            (vec![he("ʃalˈom"), he(" ʔolˈam ")], "<he>ʃalˈom ʔolˈam</he>"),
        ] {
            assert_eq!(wrap_segments(segments), expected);
        }
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
        assert_eq!(
            phonemizer.phonemize_polyg2p(" ... ", Language::Spanish),
            "..."
        );
    }

    #[test]
    fn language_codes_and_inline_tags_are_case_insensitive() {
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
    }

    #[test]
    fn markup_that_is_not_a_language_tag_is_never_spoken() {
        let mut phonemizer = Phonemizer::with_language(None::<&str>, Language::Spanish).unwrap();
        let plain = phonemizer.g2p("Hola mundo.", Language::Spanish).unwrap();
        for text in [
            "<b>Hola</b> mundo.",
            "<i>Hola</i> <span class=\"x\">mundo</span>.",
            "Hola<br/> mundo.",
        ] {
            assert_eq!(
                phonemizer.g2p(text, Language::Spanish).unwrap(),
                plain,
                "{text}"
            );
        }
        // Routing spans still route.
        let routed = phonemizer
            .g2p("<b>Hola</b> <es>mundo</es>.", Language::Spanish)
            .unwrap();
        assert_eq!(routed, plain);
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
    fn english_text_in_a_hebrew_request_reads_its_numbers_in_english() {
        let mut phonemizer = bundle_phonemizer();
        let text = "The meeting at 10:30 had 36 participants.";
        let english = phonemizer.g2p(text, Language::English).unwrap();
        assert_eq!(phonemizer.g2p(text, Language::Hebrew).unwrap(), english);
        assert!(english.contains("θˈɜːɾi sˈɪks"), "{english}");
    }

    #[test]
    #[ignore = "requires MAMBOTTS_BLUE_MODEL_DIR pointing to an installed BlueV3 bundle"]
    fn mixed_latin_and_punctuation_have_no_stray_commas() {
        // Without a Hebrew model the Hebrew segments pass through as-is, which
        // is enough to see how Latin segments and punctuation are joined.
        let mut phonemizer = bundle_phonemizer();
        let ipa = phonemizer
            .phonemize_prepared("hello PDF, world.", Language::Hebrew)
            .unwrap();
        assert_eq!(ipa, "<en>həlˈoʊ pˌiːdˌiːˈɛf, wˈɜːld.</en>");
        // Cyrillic is Russian whatever language was asked for.
        let russian = phonemizer.g2p("дверь", Language::Russian).unwrap();
        let routed = phonemizer.g2p("дверь", Language::English).unwrap();
        assert_eq!(
            routed.replace("<en>", "").replace("</en>", ""),
            russian.replace("<ru>", "").replace("</ru>", "")
        );
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
        for (text, expected) in [("אֶן", "en"), ("קֶלְוִין", "kelvin"), ("לכן תשובה ב׳", "bet")]
        {
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
        let paused = phonemizer
            .g2p("לכן תשובה ב׳: אֶן", Language::Hebrew)
            .unwrap();
        assert!(paused.contains('.'), "{paused}");
        // Slow-segment markers pace synthesis and never reach the phonemes.
        let code = phonemizer
            .g2p("קוד GPT-4 עכשיו.", Language::Hebrew)
            .unwrap();
        assert!(!code.contains(['【', '】']), "{code}");

        let text = "שבת שלום, hello!\nספר";
        let nikud = phonemizer.diacritize(text, true).unwrap();
        println!("diacritized: {nikud}");
        assert!(nikud.contains(", hello!\n"), "{nikud}");
        for text in [
            text,
            "צה\"ל הודיע שתשובה ב׳ נכונה.",
            "הוא עבר ל-GPU חדש עם 12 ליבות.",
            "ג׳אז וצ׳יפס, רוח ותקווה.",
        ] {
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
