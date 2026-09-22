#[derive(Clone, Debug)]
pub struct ChunkingOptions {
    pub enabled: bool,
    pub silence_seconds: f32,
    pub max_chars: Option<usize>,
}

impl Default for ChunkingOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            silence_seconds: 0.2,
            max_chars: None,
        }
    }
}

pub(crate) fn split_phonemes(input: &str, max_chars: Option<usize>) -> Vec<String> {
    let input = input.trim();
    if input.is_empty() {
        return Vec::new();
    }
    let Some(max_chars) = max_chars.filter(|max| *max > 0) else {
        return vec![input.to_string()];
    };

    let mut chunks = Vec::new();
    for sentence in split_sentences(input) {
        pack_sentence(&sentence, max_chars, &mut chunks);
    }
    chunks
}

/// Split raw (pre-phonemization) text into chunks, mirroring the reference
/// `chunk_text`: break into sentences, then greedily combine consecutive
/// sentences up to `max_chars`, hard-splitting any single sentence that is
/// still too long. Chunking raw text (not phonemes) is what keeps short inputs
/// as a single clean chunk instead of over-splitting into tiny trailing
/// fragments that the vocoder renders as noise.
pub(crate) fn split_text(input: &str, max_chars: usize) -> Vec<String> {
    let input = input.trim();
    if input.is_empty() {
        return Vec::new();
    }
    if max_chars == 0 {
        return vec![input.to_string()];
    }

    let mut chunks = Vec::new();
    for paragraph in input.split("\n\n") {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }
        let mut current = String::new();
        let mut current_len = 0;
        for sentence in split_sentences_text(paragraph) {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let sentence_len = sentence.chars().count();
            let separator = usize::from(!current.is_empty());
            if current_len + separator + sentence_len <= max_chars {
                if !current.is_empty() {
                    current.push(' ');
                    current_len += 1;
                }
                current.push_str(sentence);
                current_len += sentence_len;
            } else {
                if !current.is_empty() {
                    chunks.push(std::mem::take(&mut current));
                    current_len = 0;
                }
                if sentence_len > max_chars {
                    chunks.extend(hard_split_words(sentence, max_chars));
                } else {
                    current = sentence.to_string();
                    current_len = sentence_len;
                }
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
    }
    chunks
}

pub(crate) fn append_silence(audio: &mut Vec<f32>, sample_rate: u32, seconds: f32) {
    let n = (seconds.max(0.0) * sample_rate as f32).round() as usize;
    audio.extend(std::iter::repeat_n(0.0, n));
}

/// Split only after sentence-ending punctuation outside a `<…>` tag.
///
/// Language routing tags occur in phoneme input and must stay literal. A
/// partial tag such as `</en` leaks into the tokenizer as ordinary characters.
fn split_sentences(input: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let mut in_tag = false;

    for (index, character) in input.char_indices() {
        match character {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag && is_sentence_boundary(character) => {
                let end = index + character.len_utf8();
                push_trimmed(&mut sentences, &input[start..end]);
                start = end;
            }
            _ => {}
        }
    }
    push_trimmed(&mut sentences, &input[start..]);
    sentences
}

/// Greedily combine whole whitespace-delimited phoneme words. Length is
/// maintained incrementally, avoiding the previous repeated `chars().count()`.
fn pack_sentence(sentence: &str, max_chars: usize, chunks: &mut Vec<String>) {
    let mut current = String::new();
    let mut current_len = 0;

    for word in split_words_preserving_tags(sentence) {
        let word_len = word.chars().count();
        let separator_len = usize::from(!current.is_empty());
        if !current.is_empty() && current_len + separator_len + word_len > max_chars {
            chunks.push(current);
            current = String::new();
            current_len = 0;
        }

        if word_len > max_chars {
            if !current.is_empty() {
                chunks.push(current);
                current = String::new();
                current_len = 0;
            }
            chunks.extend(hard_split_token(&word, max_chars));
            continue;
        }

        if !current.is_empty() {
            current.push(' ');
            current_len += 1;
        }
        current.push_str(&word);
        current_len += word_len;
    }

    if !current.is_empty() {
        chunks.push(current);
    }
}

/// Whitespace separates words only outside a `<…>` literal.
fn split_words_preserving_tags(input: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut in_tag = false;

    for character in input.chars() {
        match character {
            '<' => {
                in_tag = true;
                current.push(character);
            }
            '>' if in_tag => {
                in_tag = false;
                current.push(character);
            }
            _ if !in_tag && character.is_whitespace() => {
                push_word(&mut words, &mut current);
            }
            _ => current.push(character),
        }
    }
    push_word(&mut words, &mut current);
    words
}

/// Hard-split an overlong token without ever cutting a `<…>` literal.
fn hard_split_token(token: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_len = 0;
    let mut index = 0;

    while index < token.len() {
        let remaining = &token[index..];
        let unit = if remaining.starts_with('<') {
            remaining
                .find('>')
                .map(|end| &remaining[..=end])
                .unwrap_or_else(|| &remaining[..remaining.chars().next().unwrap().len_utf8()])
        } else {
            &remaining[..remaining.chars().next().unwrap().len_utf8()]
        };
        let unit_len = unit.chars().count();

        if !current.is_empty() && current_len + unit_len > max_chars {
            chunks.push(current);
            current = String::new();
            current_len = 0;
        }
        current.push_str(unit);
        current_len += unit_len;
        index += unit.len();
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Split raw text into sentences after `.`/`!`/`?`/`…` when followed by
/// whitespace or end-of-text. Punctuation stays with the preceding sentence,
/// and a boundary mark hugged by a digit (a decimal) is never a split point.
fn split_sentences_text(input: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let chars: Vec<(usize, char)> = input.char_indices().collect();
    for (position, &(index, character)) in chars.iter().enumerate() {
        if matches!(character, '.' | '!' | '?' | '…') {
            let next_is_break = chars
                .get(position + 1)
                .map_or(true, |&(_, next)| next.is_whitespace());
            if next_is_break {
                let end = index + character.len_utf8();
                push_trimmed(&mut sentences, &input[start..end]);
                start = end;
            }
        }
    }
    push_trimmed(&mut sentences, &input[start..]);
    sentences
}

/// Break an overlong sentence on word boundaries, only ever character-splitting
/// a single word that itself exceeds `max_chars`.
fn hard_split_words(sentence: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_len = 0;

    for word in sentence.split_whitespace() {
        let word_len = word.chars().count();
        let separator = usize::from(!current.is_empty());
        if !current.is_empty() && current_len + separator + word_len > max_chars {
            chunks.push(std::mem::take(&mut current));
            current_len = 0;
        }
        if word_len > max_chars {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
                current_len = 0;
            }
            let mut piece = String::new();
            let mut piece_len = 0;
            for character in word.chars() {
                if piece_len >= max_chars {
                    chunks.push(std::mem::take(&mut piece));
                    piece_len = 0;
                }
                piece.push(character);
                piece_len += 1;
            }
            if !piece.is_empty() {
                chunks.push(piece);
            }
            continue;
        }
        if !current.is_empty() {
            current.push(' ');
            current_len += 1;
        }
        current.push_str(word);
        current_len += word_len;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn push_trimmed(items: &mut Vec<String>, value: &str) {
    let value = value.trim();
    if !value.is_empty() {
        items.push(value.to_string());
    }
}

fn push_word(words: &mut Vec<String>, current: &mut String) {
    if !current.is_empty() {
        words.push(std::mem::take(current));
    }
}

fn is_sentence_boundary(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | ';' | '…' | '।' | '؟' | '。')
}

#[cfg(test)]
mod tests {
    use super::split_phonemes;

    #[test]
    fn nasty_document_survives_chunking() {
        let mut document = String::new();
        for index in 0..60 {
            document.push_str(&format!(
                "עמוד {index} - היא שליחות חיי\n\n[תיאור תמונה: אישה עומדת ליד חלון]\n\n\
                 \"אמר מזכ\"ל התנועה, ח\"כ פלוני, כי \"זה הזמן\" לפעול\", והוסיף כי צה\"ל ער לכך.\n\n\
                 The lecture was titled Nature, STEM and Education, and AI was mentioned too.\n\n\
                 לפרטים: talula.hba@gmail.com או בטלפון 03-5551234 בתאריך 3.9.2026\n\n\
                 שִׁיר הַשִּׁירִים אֲשֶׁר לִשְׁלֹמֹה\n\n"
            ));
        }
        let prepared = crate::handling::prepare_text_for_synthesis(&document, "he");
        let chunks = super::split_text(&prepared, 200);
        let letters = |text: &str| -> String {
            text.chars().filter(|c| c.is_alphanumeric()).collect()
        };
        assert_eq!(letters(&chunks.join(" ")), letters(&prepared));
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 200));
    }

    #[test]
    fn returns_empty_for_empty_input() {
        assert!(split_phonemes("   ", Some(200)).is_empty());
    }

    #[test]
    fn split_text_combines_short_sentences_into_one_chunk() {
        // Two short sentences well under the limit stay together, so a short
        // input is a single clean chunk with no tiny trailing fragment.
        let chunks = super::split_text("First one. Second one.", 200);
        assert_eq!(chunks, ["First one. Second one."]);
    }

    #[test]
    fn split_text_breaks_when_over_the_limit() {
        let chunks = super::split_text("aaaa. bbbb. cccc.", 10);
        assert_eq!(chunks, ["aaaa.", "bbbb.", "cccc."]);
    }

    #[test]
    fn split_text_keeps_decimals_intact() {
        let chunks = super::split_text("Pi is 3.14 today.", 200);
        assert_eq!(chunks, ["Pi is 3.14 today."]);
    }

    #[test]
    fn split_text_hard_splits_a_single_overlong_sentence() {
        let chunks = super::split_text("alpha beta gamma delta", 11);
        assert_eq!(chunks, ["alpha beta", "gamma delta"]);
    }

    #[test]
    fn leaves_input_whole_without_a_limit() {
        assert_eq!(split_phonemes("a b c", None), ["a b c"]);
    }

    #[test]
    fn splits_at_sentence_boundaries_first() {
        assert_eq!(
            split_phonemes("first sentence. second sentence!", Some(200)),
            ["first sentence.", "second sentence!"]
        );
    }

    #[test]
    fn packs_whole_words_up_to_the_limit() {
        assert_eq!(
            split_phonemes("alpha beta gamma delta", Some(10)),
            ["alpha beta", "gamma", "delta"]
        );
    }

    #[test]
    fn never_splits_a_word_just_to_fill_a_chunk() {
        assert_eq!(split_phonemes("aaaa bbbbb", Some(8)), ["aaaa", "bbbbb"]);
    }

    #[test]
    fn preserves_language_tags_when_packing_words() {
        let chunks = split_phonemes("<en>həˈloʊ</en> <he>ʃaˈlom</he>", Some(18));
        assert_eq!(chunks, ["<en>həˈloʊ</en>", "<he>ʃaˈlom</he>"]);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.matches('<').count() == chunk.matches('>').count())
        );
    }

    #[test]
    fn hard_splits_an_overlong_word() {
        assert_eq!(
            split_phonemes("abcdefghij", Some(4)),
            ["abcd", "efgh", "ij"]
        );
    }

    #[test]
    fn hard_split_never_cuts_inside_a_tag_literal() {
        let chunks = split_phonemes("aa<en>tag</en>bbbb", Some(5));
        assert_eq!(chunks, ["aa", "<en>t", "ag", "</en>", "bbbb"]);
        assert!(
            chunks
                .iter()
                .all(|chunk| !chunk.contains("<e") || chunk.contains("<en>"))
        );
        assert!(
            chunks
                .iter()
                .all(|chunk| !chunk.contains("</e") || chunk.contains("</en>"))
        );
    }
}
