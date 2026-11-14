use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result, bail};
use ndarray::{Array2, Array3};
use regex::Regex;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use crate::ipa::{normalize_ipa, strip_language_tags, tag_code};

pub(crate) struct Tokenizer {
    pad_id: i64,
    char_to_id: HashMap<char, i64>,
    tag_re: Regex,
    /// Present on BlueV3 release bundles, whose ids carry the language tags.
    lang_tag: Option<LangTag>,
}

/// The vocab characters a release bundle spells `<code>` ... `</code>` with.
#[derive(Clone, Copy)]
struct LangTag {
    open: char,
    close: char,
    slash: char,
}

impl Tokenizer {
    pub(crate) fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let raw = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read vocab {}", path.as_ref().display()))?;
        Self::from_json_bytes(raw.as_bytes())
    }

    pub(crate) fn from_json_bytes(raw: &[u8]) -> Result<Self> {
        let json: Value = serde_json::from_slice(raw)?;
        let pad_id = json["pad_id"].as_i64().unwrap_or(0);
        let mut char_to_id = HashMap::new();
        for (k, v) in json["char_to_id"]
            .as_object()
            .context("char_to_id missing")?
        {
            if let Some(ch) = k.chars().next() {
                char_to_id.insert(ch, v.as_i64().unwrap_or(pad_id));
            }
        }
        let lang_tag = json
            .get("lang_tag")
            .filter(|tag| tag.is_object())
            .map(|tag| {
                let character = |key: &str, default: char| {
                    tag[key]
                        .as_str()
                        .and_then(|value| value.chars().next())
                        .unwrap_or(default)
                };
                LangTag {
                    open: character("open", '<'),
                    close: character("close", '>'),
                    slash: character("slash", '/'),
                }
            });
        Ok(Self {
            pad_id,
            char_to_id,
            tag_re: Regex::new(r"</?[^>]+>")?,
            lang_tag,
        })
    }

    /// True when ids are wrapped in language tags (a BlueV3 release bundle).
    pub(crate) fn is_tagged(&self) -> bool {
        self.lang_tag.is_some()
    }

    /// Ids one chunk of phonemes becomes, tags included: the unit the
    /// release's chunk budget is counted in.
    pub(crate) fn token_count(&self, phonemes: &str, lang: &str) -> Result<usize> {
        Ok(self.encode_one(phonemes, lang)?.len())
    }

    pub(crate) fn supported_characters(&self) -> Vec<char> {
        let mut characters: Vec<_> = self.char_to_id.keys().copied().collect();
        characters.sort_unstable();
        characters
    }

    pub(crate) fn encode_batch(
        &self,
        texts: &[&str],
        langs: &[&str],
    ) -> Result<(Array2<i64>, Array3<f32>)> {
        if texts.len() != langs.len() {
            bail!("texts/langs length mismatch");
        }
        let encoded: Vec<Vec<i64>> = texts
            .iter()
            .zip(langs)
            .map(|(text, lang)| self.encode_one(text, lang))
            .collect::<Result<_>>()?;
        let max_len = encoded.iter().map(Vec::len).max().unwrap_or(0);
        let mut ids = Array2::from_elem((texts.len(), max_len), self.pad_id);
        let mut mask = Array3::zeros((texts.len(), 1, max_len));
        for (row, seq) in encoded.iter().enumerate() {
            for (col, id) in seq.iter().copied().enumerate() {
                ids[[row, col]] = id;
                mask[[row, 0, col]] = 1.0;
            }
        }
        Ok((ids, mask))
    }

    fn encode_one(&self, text: &str, lang: &str) -> Result<Vec<i64>> {
        if let Some(tag) = self.lang_tag {
            return self.encode_tagged(text, lang, tag);
        }
        if !matches!(lang, "en" | "es" | "de" | "it" | "he") {
            bail!("invalid language: {lang}");
        }
        let mut text = preprocess_phonemes(text);
        if !text.ends_with(|c| {
            matches!(
                c,
                '.' | '!' | '?' | ';' | ':' | ',' | '\'' | '"' | ')' | ']' | '}'
            )
        }) {
            text.push('.');
        }
        let wrapped = format!("<{lang}>{text}</{lang}>");
        let stripped = self.tag_re.replace_all(&wrapped, "");
        Ok(stripped
            .chars()
            .map(|ch| *self.char_to_id.get(&ch).unwrap_or(&self.pad_id))
            .collect())
    }
}

impl Tokenizer {
    /// `encode_tagged` of the release: the ids of `<code>`, the normalised
    /// phonemes and `</code>`. Inline routing tags from the G2P are dropped
    /// and the chunk is wrapped once in its own language. A symbol the vocab
    /// does not know becomes the pad id, as the reference does.
    fn encode_tagged(&self, text: &str, lang: &str, tag: LangTag) -> Result<Vec<i64>> {
        let code = tag_code(lang);
        if code.is_empty() {
            bail!("invalid language: {lang}");
        }
        let id = |character: char| {
            self.char_to_id.get(&character).copied().with_context(|| {
                format!("vocab.json has no id for the language tag character `{character}`")
            })
        };
        let text = normalize_ipa(&strip_language_tags(text), &code);
        let mut ids = Vec::with_capacity(text.chars().count() + 2 * code.len() + 5);
        ids.push(id(tag.open)?);
        for character in code.chars() {
            ids.push(id(character)?);
        }
        ids.push(id(tag.close)?);
        ids.extend(
            text.chars()
                .map(|character| *self.char_to_id.get(&character).unwrap_or(&self.pad_id)),
        );
        ids.push(id(tag.open)?);
        ids.push(id(tag.slash)?);
        for character in code.chars() {
            ids.push(id(character)?);
        }
        ids.push(id(tag.close)?);
        Ok(ids)
    }
}

fn preprocess_phonemes(text: &str) -> String {
    let mut text: String = text
        .nfkd()
        .filter(|character| !is_hebrew_nikud(*character) && !is_emoji(*character))
        .collect();
    for (from, to) in [
        ("–", "-"),
        ("‑", "-"),
        ("—", "-"),
        ("_", " "),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
        ("´", "'"),
        ("`", "'"),
        ("[", " "),
        ("]", " "),
        ("|", " "),
        ("/", " "),
        ("#", " "),
        ("→", " "),
        ("←", " "),
    ] {
        text = text.replace(from, to);
    }
    text = text.replace('@', " at ");
    text = text.replace("e.g.,", "for example, ");
    text = text.replace("i.e.,", "that is, ");
    text = regex!(r"(?u)([\u{0590}-\u{05ff}])[-–—‑]+([A-Za-z0-9])")
        .replace_all(&text, "$1 $2")
        .into_owned();
    text = regex!(r"(?u)([A-Za-z0-9])[-–—‑]+([\u{0590}-\u{05ff}])")
        .replace_all(&text, "$1 $2")
        .into_owned();
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_hebrew_nikud(character: char) -> bool {
    matches!(character, '\u{0591}'..='\u{05bd}' | '\u{05bf}' | '\u{05c1}'..='\u{05c2}' | '\u{05c4}'..='\u{05c5}' | '\u{05c7}')
}

fn is_emoji(character: char) -> bool {
    matches!(
        character as u32,
        0x1f600..=0x1f64f
            | 0x1f300..=0x1f5ff
            | 0x1f680..=0x1faff
            | 0x2600..=0x26ff
            | 0x2700..=0x27bf
            | 0x1f1e6..=0x1f1ff
    )
}
