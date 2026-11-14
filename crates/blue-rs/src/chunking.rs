pub(crate) fn split_phonemes(input: &str, max_chars: Option<usize>) -> Vec<String> {
    let input = input.trim();
    if input.is_empty() {
        return Vec::new();
    }
    let Some(max_chars) = max_chars.filter(|max| *max > 0) else {
        return vec![input.to_string()];
    };

    let mut chunks = Vec::new();
    for sentence in split_phoneme_sentences(input) {
        pack_sentence(&sentence, max_chars, &mut chunks);
    }
    chunks
}

/// A sentence of fewer words than this ("Hi.", "Yes. Yes.") is too short to
/// stand alone as an utterance and joins the sentence after it.
const FRAGMENT_WORDS: usize = 3;

/// Split raw (pre-phonemization) text into one chunk a sentence.
///
/// A paragraph break always starts a new chunk. Within a paragraph each
/// sentence is its own chunk, so the model ends it the way it ends an
/// utterance; a fragment of fewer than [`FRAGMENT_WORDS`] words is held and
/// joined to the sentence that follows it. Chunking raw text rather than
/// phonemes keeps a short input one clean chunk. Whether a chunk fits the
/// model's token budget is only known after G2P, so the caller measures it
/// then and cuts an overlong sentence with [`hard_split_words`].
pub(crate) fn split_sentence_chunks(input: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    // A blank line, whatever the line endings and stray spaces on it.
    for paragraph in regex!(r"\n\s*\n").split(input) {
        let mut held = String::new();
        for sentence in split_sentences_text(paragraph) {
            if !held.is_empty() {
                held.push(' ');
            }
            held.push_str(sentence.trim());
            if held.split_whitespace().count() >= FRAGMENT_WORDS {
                chunks.push(std::mem::take(&mut held));
            }
        }
        if held.is_empty() {
            continue;
        }
        // A fragment that ends its paragraph has no next sentence to join.
        // Words are spoken on their own; stray punctuation goes back onto
        // the sentence before it.
        if held.chars().any(char::is_alphanumeric) || chunks.is_empty() {
            chunks.push(held);
        } else if let Some(last) = chunks.last_mut() {
            last.push_str(&held);
        }
    }
    chunks
}

pub(crate) fn append_silence(audio: &mut Vec<f32>, sample_rate: u32, seconds: f32) {
    let n = (seconds.max(0.0) * sample_rate as f32).round() as usize;
    audio.extend(std::iter::repeat_n(0.0, n));
}

/// Split phonemes after sentence-ending punctuation outside a `<…>` tag.
///
/// A run of marks (`...`, `?!`) is one boundary. Language routing tags stay
/// literal and balanced: a closing tag right after the mark stays with its
/// sentence, and a language span cut by a boundary is closed in one piece and
/// reopened in the next. A piece with nothing to say (punctuation alone, or a
/// tag alone) goes onto the sentence before it.
pub(crate) fn split_phoneme_sentences(input: &str) -> Vec<String> {
    let chars: Vec<char> = input.chars().collect();
    // The tag starting at `at`, if any: its text and the index after it.
    let tag_at = |at: usize| -> Option<(String, usize)> {
        if chars.get(at) != Some(&'<') {
            return None;
        }
        let close = at + 1 + chars[at + 1..].iter().position(|&c| c == '>' || c == '<')?;
        (chars[close] == '>').then(|| (chars[at..=close].iter().collect(), close + 1))
    };
    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut open: Option<String> = None;
    let mut reopen: Option<String> = None;
    let mut index = 0;
    while index < chars.len() {
        if let Some((tag, next)) = tag_at(index) {
            let closing = tag.starts_with("</");
            // A span closed at the boundary that now closes itself as well
            // leaves nothing to reopen.
            if !(closing && reopen.take().is_some()) {
                reopen = None;
                current.push_str(&tag);
                open = (!closing).then(|| tag[1..tag.len() - 1].to_owned());
            }
            index = next;
            continue;
        }
        let character = chars[index];
        if !character.is_whitespace()
            && let Some(name) = reopen.take()
        {
            current.push_str(&format!("<{name}>"));
            open = Some(name);
        }
        current.push(character);
        index += 1;
        if !is_sentence_boundary(character) {
            continue;
        }
        while let Some(&mark) = chars.get(index).filter(|&&c| is_sentence_boundary(c)) {
            current.push(mark);
            index += 1;
        }
        if let Some((tag, next)) = tag_at(index).filter(|(tag, _)| tag.starts_with("</")) {
            current.push_str(&tag);
            open = None;
            index = next;
        }
        if let Some(name) = open.take() {
            current.push_str(&format!("</{name}>"));
            reopen = Some(name);
        }
        push_trimmed(&mut pieces, &current);
        current.clear();
    }
    push_trimmed(&mut pieces, &current);
    merge_unspoken(pieces)
}

/// True when `text` has a letter or digit outside its `<…>` tags.
pub(crate) fn is_speakable(text: &str) -> bool {
    strip_tags(text).chars().any(char::is_alphanumeric)
}

fn strip_tags(text: &str) -> std::borrow::Cow<'_, str> {
    regex!(r"<[^<>]*>").replace_all(text, "")
}

/// Fold pieces with nothing to say into the speech next to them: onto the
/// piece before (inside its closing tags), or, at the start, onto the next.
fn merge_unspoken(pieces: Vec<String>) -> Vec<String> {
    let mut merged: Vec<String> = Vec::new();
    let mut lead = String::new();
    for piece in pieces {
        if is_speakable(&piece) {
            if lead.is_empty() {
                merged.push(piece);
            } else {
                merged.push(format!("{} {piece}", std::mem::take(&mut lead)));
            }
            continue;
        }
        let marks = strip_tags(&piece).split_whitespace().collect::<String>();
        if marks.is_empty() {
            continue;
        }
        match merged.last_mut() {
            Some(last) => {
                let mut at = last.len();
                while last[..at].ends_with('>') {
                    match last[..at].rfind('<') {
                        Some(start) if last[start..].starts_with("</") => at = start,
                        _ => break,
                    }
                }
                last.insert_str(at, &marks);
            }
            None => lead.push_str(&marks),
        }
    }
    if merged.is_empty() && !lead.is_empty() {
        merged.push(lead);
    }
    merged
}

/// Hold a phoneme sentence of fewer than [`FRAGMENT_WORDS`] words and join it
/// to the sentence after it, as [`split_sentence_chunks`] does with text.
pub(crate) fn join_phoneme_fragments(sentences: Vec<String>) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut held = String::new();
    for sentence in sentences {
        if !held.is_empty() {
            held.push(' ');
        }
        held.push_str(&sentence);
        if strip_tags(&held).split_whitespace().count() >= FRAGMENT_WORDS {
            chunks.push(std::mem::take(&mut held));
        }
    }
    if !held.is_empty() {
        chunks.push(held);
    }
    chunks
}

/// Greedily combine whole whitespace-delimited phoneme words. Length is
/// maintained incrementally, avoiding the previous repeated `chars().count()`.
pub(crate) fn pack_sentence(sentence: &str, max_chars: usize, chunks: &mut Vec<String>) {
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
pub(crate) fn split_sentences_text(input: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let closing_tag = regex!(r"^</[A-Za-z-]+>");
    for (index, character) in input.char_indices() {
        if matches!(character, '.' | '!' | '?' | '…') {
            let mark_end = index + character.len_utf8();
            let mut end = mark_end;
            // A language span that ends with the sentence ends with it.
            if let Some(tag) = closing_tag.find(&input[end..]) {
                end += tag.end();
            }
            let next_is_break = input[end..].chars().next().is_none_or(char::is_whitespace);
            if next_is_break && !ends_with_abbreviation(&input[start..mark_end]) {
                push_trimmed(&mut sentences, &input[start..end]);
                start = end;
            }
        }
    }
    push_trimmed(&mut sentences, &input[start..]);
    sentences
}

/// Abbreviations whose period does not end a sentence, as the release's
/// `chunk_text` lists them.
const ABBREVIATIONS: &[&str] = &[
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Ph.D.", "etc.", "e.g.", "i.e.", "vs.",
    "Inc.", "Ltd.", "Co.", "Corp.", "St.", "Ave.", "Blvd.",
];

/// True when the text so far ends in a listed abbreviation, a single capital
/// initial ("J. Smith") or a short ordinal number, so its period is not a
/// sentence end.
fn ends_with_abbreviation(text: &str) -> bool {
    if ABBREVIATIONS.iter().any(|item| text.ends_with(item)) {
        return true;
    }
    let Some(body) = text.strip_suffix('.') else {
        return false;
    };
    let word = body
        .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
        .next()
        .unwrap_or_default();
    let mut letters = word.chars();
    match (letters.next(), letters.next(), letters.next()) {
        (Some(initial), None, _) if initial.is_ascii_uppercase() => true,
        // A one or two digit number and a period is an ordinal or a list
        // number ("am 3. Oktober", "1. First"), which the G2P can only read
        // as one if the words after it stay in the same chunk. A longer
        // number, a year say, ends its sentence.
        (Some(first), second, None) => {
            first.is_ascii_digit() && second.is_none_or(|c| c.is_ascii_digit())
        }
        _ => false,
    }
}

/// Break an overlong sentence on word boundaries, only ever character-splitting
/// a single word that itself exceeds `max_chars`.
pub(crate) fn hard_split_words(sentence: &str, max_chars: usize) -> Vec<String> {
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
        let chunks = super::split_sentence_chunks(&prepared);
        let letters =
            |text: &str| -> String { text.chars().filter(|c| c.is_alphanumeric()).collect() };
        assert_eq!(letters(&chunks.join(" ")), letters(&prepared));
    }

    #[test]
    fn phoneme_sentences_have_no_punctuation_only_pieces() {
        use super::split_phoneme_sentences as split;
        assert_eq!(split("həlˈoʊ... wˈɜːld"), ["həlˈoʊ...", "wˈɜːld"]);
        assert_eq!(split("həlˈoʊ… wˈɜːld?!"), ["həlˈoʊ…", "wˈɜːld?!"]);
        assert_eq!(split("həlˈoʊ. . ! wˈɜːld"), ["həlˈoʊ..!", "wˈɜːld"]);
        assert_eq!(split("... həlˈoʊ"), ["... həlˈoʊ"]);
        assert_eq!(split("..."), ["..."]);
        assert_eq!(
            split("<en>həlˈoʊ.</en> <he>ʃalˈom.</he>"),
            ["<en>həlˈoʊ.</en>", "<he>ʃalˈom.</he>"]
        );
        // A span cut by a boundary is closed and reopened.
        assert_eq!(
            split("<he>ʃalˈom. ʔolˈam.</he>"),
            ["<he>ʃalˈom.</he>", "<he>ʔolˈam.</he>"]
        );
        assert_eq!(split("<en>həlˈoʊ.</en> ."), ["<en>həlˈoʊ..</en>"]);
        for chunk in split("<en>a. b... c</en> <he>d. e.</he> . <en>f</en>") {
            assert_eq!(
                chunk.matches("<en>").count(),
                chunk.matches("</en>").count(),
                "{chunk}"
            );
            assert_eq!(
                chunk.matches("<he>").count(),
                chunk.matches("</he>").count(),
                "{chunk}"
            );
            assert!(super::is_speakable(&chunk), "{chunk}");
        }
    }

    #[test]
    fn short_phoneme_sentences_join_the_next_one() {
        let sentences =
            super::split_phoneme_sentences("həlˈoʊ... wˈɜːld. aɪ æm hˈɪɹ nˈaʊ. ɡʊdbˈaɪ.");
        assert_eq!(
            super::join_phoneme_fragments(sentences),
            ["həlˈoʊ... wˈɜːld. aɪ æm hˈɪɹ nˈaʊ.", "ɡʊdbˈaɪ."]
        );
    }

    #[test]
    fn any_blank_line_is_a_paragraph_break() {
        for text in [
            "Title\r\n\r\nThe body starts here.",
            "Title\n  \n\nThe body starts here.",
        ] {
            assert_eq!(
                super::split_sentence_chunks(text),
                ["Title", "The body starts here."]
            );
        }
    }

    #[test]
    fn a_sentence_ends_after_its_closing_language_tag() {
        let chunks = super::split_sentence_chunks(
            "שלום לכולם כאן. <en>The meeting was long.</en> <en>Dr. Smith said hi there.</en> ביי לכולם עכשיו.",
        );
        assert_eq!(
            chunks,
            [
                "שלום לכולם כאן.",
                "<en>The meeting was long.</en>",
                "<en>Dr. Smith said hi there.</en>",
                "ביי לכולם עכשיו."
            ]
        );
    }

    #[test]
    fn returns_empty_for_empty_input() {
        assert!(split_phonemes("   ", Some(200)).is_empty());
    }

    #[test]
    fn each_sentence_is_its_own_chunk() {
        let chunks = super::split_sentence_chunks("The first one here. The second one here!");
        assert_eq!(chunks, ["The first one here.", "The second one here!"]);
    }

    #[test]
    fn a_short_fragment_joins_the_next_sentence() {
        let chunks = super::split_sentence_chunks("Hi. Yes. How are you today? Fine.");
        assert_eq!(chunks, ["Hi. Yes. How are you today?", "Fine."]);
    }

    #[test]
    fn a_paragraph_break_starts_a_new_chunk() {
        let chunks = super::split_sentence_chunks("Title\n\nThe body starts here.");
        assert_eq!(chunks, ["Title", "The body starts here."]);
    }

    #[test]
    fn trailing_punctuation_alone_stays_with_the_sentence_before_it() {
        let chunks = super::split_sentence_chunks("This is one sentence. ...");
        assert_eq!(chunks, ["This is one sentence...."]);
    }

    #[test]
    fn abbreviations_and_initials_do_not_end_a_sentence() {
        let chunks = super::split_sentence_chunks(
            "Dr. Smith met J. K. Rowling at St. Mary today. Then Mr. Jones left the room.",
        );
        assert_eq!(
            chunks,
            [
                "Dr. Smith met J. K. Rowling at St. Mary today.",
                "Then Mr. Jones left the room."
            ]
        );
    }

    #[test]
    fn an_ordinal_number_stays_with_the_words_after_it() {
        let chunks = super::split_sentence_chunks(
            "Die Regierung hat am 3. Oktober entschieden. Das war 1990. Dann kam der Winter.",
        );
        assert_eq!(
            chunks,
            [
                "Die Regierung hat am 3. Oktober entschieden.",
                "Das war 1990.",
                "Dann kam der Winter."
            ]
        );
    }

    #[test]
    fn decimals_are_not_sentence_ends() {
        let chunks = super::split_sentence_chunks("Pi is 3.14 today.");
        assert_eq!(chunks, ["Pi is 3.14 today."]);
    }

    #[test]
    fn an_overlong_sentence_is_cut_on_word_boundaries() {
        let chunks = super::hard_split_words("alpha beta gamma delta", 11);
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
