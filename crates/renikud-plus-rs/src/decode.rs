//! The cascade decode: exact MAP over `E(c, v, s)`, and the greedy fallback.

use std::collections::{HashMap, HashSet};

use crate::niqqud::{Constraint, Constraints};
use crate::pychars;
use crate::text::is_hebrew;

/// The energy an illegal reading is pushed to.
pub(crate) const NEG: f32 = -1e30;

/// Conditioning column blocks of the cascade heads, from the model metadata.
pub(crate) struct Cascade {
    /// `[C, V]`
    pub wv_c: Vec<Vec<f32>>,
    /// `[C, 2]`
    pub ws_c: Vec<Vec<f32>>,
    /// `[V, 2]`
    pub ws_v: Vec<Vec<f32>>,
    pub cond_softmax: bool,
    /// `[letter, consonant]`, true where the consonant class is illegal.
    pub forbidden: Vec<Vec<bool>>,
}

/// NumPy's pairwise summation, so a sum of f32 lands on the same bits it does.
fn pairwise_sum(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 8 {
        let mut res = 0.0f32;
        for &value in values {
            res += value;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&values[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, slot) in r.iter_mut().enumerate() {
                *slot += values[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += values[i];
            i += 1;
        }
        return res;
    }
    let half = (n / 2) - ((n / 2) % 8);
    pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
}

fn max_of(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// First index of the maximum, as `numpy.argmax`.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, &value) in values.iter().enumerate() {
        if value > values[best] {
            best = i;
        }
    }
    best
}

fn softmax_row(row: &[f32]) -> Vec<f32> {
    let max = max_of(row);
    let exponentiated: Vec<f32> = row.iter().map(|&value| (value - max).exp()).collect();
    let total = pairwise_sum(&exponentiated);
    exponentiated.iter().map(|&value| value / total).collect()
}

fn log_softmax_row(row: &[f32]) -> Vec<f32> {
    let max = max_of(row);
    let shifted: Vec<f32> = row.iter().map(|&value| value - max).collect();
    let total: Vec<f32> = shifted.iter().map(|&value| value.exp()).collect();
    let log_total = pairwise_sum(&total).ln();
    shifted.iter().map(|&value| value - log_total).collect()
}

fn onehot_row(row: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; row.len()];
    out[argmax(row)] = 1.0;
    out
}

/// Word spans over the window, as `re.finditer(r"\S+", text)` finds them.
fn word_spans(chars: &[char]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (i, &c) in chars.iter().enumerate() {
        if pychars::is_space(c) {
            if let Some(s) = start.take() {
                spans.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        spans.push((s, chars.len()));
    }
    spans
}

/// The labels one decode produced: a consonant id and a vowel id per token,
/// plus the stressed token of every word.
pub(crate) struct Decoded {
    pub consonants: Vec<usize>,
    pub vowels: Vec<usize>,
    pub stressed: HashSet<usize>,
}

/// One session run's logits: a row per token for each cascade head.
pub(crate) struct Heads<'a> {
    pub consonant: &'a [Vec<f32>],
    pub vowel: &'a [Vec<f32>],
    pub stress: &'a [Vec<f32>],
}

/// The label vocabularies the decode needs, by id and by name.
pub(crate) struct Labels<'a> {
    pub consonant_ids: &'a HashMap<String, usize>,
    pub vowel_ids: &'a HashMap<String, usize>,
}

/// Greedy argmax over each head, with the stress mark going to the
/// highest-margin vowel-bearing token of each word.
pub(crate) fn greedy(
    offsets: &[(usize, usize)],
    chars: &[char],
    heads: &Heads,
    vowel_vocab: &HashMap<usize, String>,
) -> Decoded {
    let stress_logits = heads.stress;
    let consonants: Vec<usize> = heads.consonant.iter().map(|row| argmax(row)).collect();
    let vowels: Vec<usize> = heads.vowel.iter().map(|row| argmax(row)).collect();

    let spans = word_spans(chars);
    let mut words: Vec<Vec<usize>> = vec![Vec::new(); spans.len()];
    for (tok_idx, &(start, end)) in offsets.iter().enumerate() {
        if end - start != 1 {
            continue;
        }
        if let Some(word_idx) = spans.iter().position(|&(ws, we)| ws <= start && start < we) {
            words[word_idx].push(tok_idx);
        }
    }
    let mut stressed = HashSet::new();
    for toks in words {
        // Stress must sit with a vowel (CˈV); never emit a trailing ˈ.
        let vowel_toks: Vec<usize> = toks
            .into_iter()
            .filter(|&t| {
                vowel_vocab
                    .get(&vowels[t])
                    .map(String::as_str)
                    .unwrap_or("∅")
                    != "∅"
            })
            .collect();
        // Margin (yes − no), not the raw yes-logit: raw logits differ in scale
        // across tokens.
        let margin = |t: usize| stress_logits[t][1] - stress_logits[t][0];
        if let Some(&best) = vowel_toks.iter().reduce(|best, candidate| {
            if margin(*candidate) > margin(*best) {
                candidate
            } else {
                best
            }
        }) {
            stressed.insert(best);
        }
    }
    Decoded {
        consonants,
        vowels,
        stressed,
    }
}

/// Closed-form MAP over the joint cascade energy.
///
/// ```text
/// E(c, v, s) = log P(c) + log P(v | c) + log P(s | c, v)
/// ```
///
/// The graph emits vowel/stress logits conditioned on the model's own upstream
/// softmax; subtracting that conditioning back out recovers the unconditioned
/// head logits, after which conditioning on ANY `(c, v)` is an exact column
/// addition. No early argmax anywhere, so the one-stress-per-word constraint is
/// free to flip both the vowel and the consonant.
pub(crate) fn exact_map(
    offsets: &[(usize, usize)],
    chars: &[char],
    heads: &Heads,
    cascade: &Cascade,
    constraints: Option<&Constraints>,
    labels: &Labels,
) -> Decoded {
    let (consonant_logits, vowel_logits, stress_logits) =
        (heads.consonant, heads.vowel, heads.stress);
    let (consonant_ids, vowel_ids) = (labels.consonant_ids, labels.vowel_ids);
    let seq = consonant_logits.len();
    let n_cons = cascade.wv_c.len();
    let n_vowels = cascade.ws_v.len();

    // energy[t][c][v][s], flattened.
    let stride_c = n_vowels * 2;
    let stride_t = n_cons * stride_c;
    let mut energy = vec![0.0f32; seq * stride_t];

    for t in 0..seq {
        let cond_c = if cascade.cond_softmax {
            softmax_row(&consonant_logits[t])
        } else {
            onehot_row(&consonant_logits[t])
        };
        let cond_v = if cascade.cond_softmax {
            softmax_row(&vowel_logits[t])
        } else {
            onehot_row(&vowel_logits[t])
        };
        // base_v = vowel_logits - cond_c @ wv_c
        let base_v: Vec<f32> = (0..n_vowels)
            .map(|v| {
                let mut sum = 0.0f32;
                for (c, weight) in cond_c.iter().enumerate() {
                    sum += weight * cascade.wv_c[c][v];
                }
                vowel_logits[t][v] - sum
            })
            .collect();
        // base_s = stress_logits - cond_c @ ws_c - cond_v @ ws_v
        let base_s: Vec<f32> = (0..2)
            .map(|k| {
                let mut from_c = 0.0f32;
                for (c, weight) in cond_c.iter().enumerate() {
                    from_c += weight * cascade.ws_c[c][k];
                }
                let mut from_v = 0.0f32;
                for (v, weight) in cond_v.iter().enumerate() {
                    from_v += weight * cascade.ws_v[v][k];
                }
                stress_logits[t][k] - from_c - from_v
            })
            .collect();

        let logc = log_softmax_row(&consonant_logits[t]);
        for (c, &log_consonant) in logc.iter().enumerate() {
            let logv = log_softmax_row(
                &base_v
                    .iter()
                    .zip(&cascade.wv_c[c])
                    .map(|(base, weight)| base + weight)
                    .collect::<Vec<_>>(),
            );
            for (v, &log_vowel) in logv.iter().enumerate() {
                let logs = log_softmax_row(&[
                    base_s[0] + cascade.ws_c[c][0] + cascade.ws_v[v][0],
                    base_s[1] + cascade.ws_c[c][1] + cascade.ws_v[v][1],
                ]);
                let base = t * stride_t + c * stride_c + v * 2;
                energy[base] = log_consonant + log_vowel + logs[0];
                energy[base + 1] = log_consonant + log_vowel + logs[1];
            }
        }
    }

    // Per-letter consonant legality, and "stress needs a vowel".
    for (t, &(start, end)) in offsets.iter().enumerate() {
        if end - start == 1 && is_hebrew(chars[start]) {
            let letter = chars[start] as usize - crate::text::ALEF as usize;
            for c in 0..n_cons {
                if cascade.forbidden[letter][c] {
                    for slot in &mut energy[t * stride_t + c * stride_c..][..stride_c] {
                        *slot = NEG;
                    }
                }
            }
        }
    }

    // Niqqud read off the input, as hard constraints. A constraint that
    // contradicts the letter's own legality would leave the token with no legal
    // reading at all, so it is dropped rather than allowed to pick garbage out
    // of an all-NEG row.
    let mut pinned_stress: HashMap<usize, Vec<usize>> = HashMap::new();
    if let Some(constraints) = constraints {
        let char_to_tok: HashMap<usize, usize> = offsets
            .iter()
            .enumerate()
            .filter(|&(_, &(start, end))| end - start == 1)
            .map(|(t, &(start, _))| (start, t))
            .collect();
        for (&idx, constraint) in constraints {
            let Some(&t) = char_to_tok.get(&idx) else {
                continue;
            };
            let Constraint {
                consonants,
                vowels,
                stressed,
            } = constraint;
            let saved: Vec<f32> = energy[t * stride_t..][..stride_t].to_vec();
            if let Some(consonants) = consonants {
                let keep: Vec<usize> = consonants
                    .iter()
                    .filter_map(|label| consonant_ids.get(*label).copied())
                    .collect();
                for c in 0..n_cons {
                    if !keep.contains(&c) {
                        for slot in &mut energy[t * stride_t + c * stride_c..][..stride_c] {
                            *slot = NEG;
                        }
                    }
                }
            }
            if let Some(vowels) = vowels {
                let keep: Vec<usize> = vowels
                    .iter()
                    .filter_map(|label| vowel_ids.get(*label).copied())
                    .collect();
                for c in 0..n_cons {
                    for v in 0..n_vowels {
                        if !keep.contains(&v) {
                            let base = t * stride_t + c * stride_c + v * 2;
                            energy[base] = NEG;
                            energy[base + 1] = NEG;
                        }
                    }
                }
            }
            if max_of(&energy[t * stride_t..][..stride_t]) < NEG / 2.0 {
                energy[t * stride_t..][..stride_t].copy_from_slice(&saved);
            }
            if *stressed {
                // The hatama: this letter takes its word's stress. Recorded per
                // word so the choice below can still fall back when the letter
                // turns out to have no legal stressed reading.
                if let Some(word) = word_spans(chars)
                    .into_iter()
                    .position(|(ws, we)| ws <= idx && idx < we)
                {
                    pinned_stress.entry(word).or_default().push(t);
                }
            }
        }
    }

    // Stress needs a vowel: vowel id 0 is ∅.
    for t in 0..seq {
        for c in 0..n_cons {
            energy[t * stride_t + c * stride_c + 1] = NEG;
        }
    }

    let mut arg_u = vec![0usize; seq];
    let mut arg_s = vec![0usize; seq];
    let mut best_u = vec![0.0f32; seq];
    let mut best_s = vec![0.0f32; seq];
    for t in 0..seq {
        let flat_u: Vec<f32> = (0..n_cons * n_vowels)
            .map(|i| energy[t * stride_t + (i / n_vowels) * stride_c + (i % n_vowels) * 2])
            .collect();
        let flat_s: Vec<f32> = (0..n_cons * n_vowels)
            .map(|i| energy[t * stride_t + (i / n_vowels) * stride_c + (i % n_vowels) * 2 + 1])
            .collect();
        arg_u[t] = argmax(&flat_u);
        arg_s[t] = argmax(&flat_s);
        best_u[t] = flat_u[arg_u[t]];
        best_s[t] = flat_s[arg_s[t]];
    }
    let gain: Vec<f32> = (0..seq).map(|t| best_s[t] - best_u[t]).collect();

    let char_tok: HashMap<usize, usize> = offsets
        .iter()
        .enumerate()
        .filter(|&(_, &(start, end))| end - start == 1 && is_hebrew(chars[start]))
        .map(|(t, &(start, _))| (start, t))
        .collect();
    let mut stressed = HashSet::new();
    for (word, (start, end)) in word_spans(chars).into_iter().enumerate() {
        let toks: Vec<usize> = (start..end)
            .filter_map(|i| char_tok.get(&i).copied())
            .collect();
        let viable = |t: &usize| best_s[*t] > NEG / 2.0;
        let pinned: Vec<usize> = pinned_stress
            .get(&word)
            .map(|toks| toks.iter().copied().filter(|t| viable(t)).collect())
            .unwrap_or_default();
        let cands: Vec<usize> = if pinned.is_empty() {
            toks.into_iter().filter(|t| viable(t)).collect()
        } else {
            pinned
        };
        if let Some(&best) = cands.iter().reduce(|best, candidate| {
            if gain[*candidate] > gain[*best] {
                candidate
            } else {
                best
            }
        }) {
            stressed.insert(best);
        }
    }

    let mut consonants = vec![0usize; seq];
    let mut vowels = vec![0usize; seq];
    for t in 0..seq {
        let flat = if stressed.contains(&t) {
            arg_s[t]
        } else {
            arg_u[t]
        };
        consonants[t] = flat / n_vowels;
        vowels[t] = flat % n_vowels;
    }
    Decoded {
        consonants,
        vowels,
        stressed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_sum_matches_a_plain_sum_on_exact_values() {
        let values: Vec<f32> = (0..25).map(|i| i as f32).collect();
        assert_eq!(pairwise_sum(&values), 300.0);
        assert_eq!(pairwise_sum(&values[..3]), 3.0);
        let long: Vec<f32> = (0..300).map(|_| 0.5f32).collect();
        assert_eq!(pairwise_sum(&long), 150.0);
    }

    #[test]
    fn argmax_takes_the_first_maximum() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn log_softmax_sums_to_one_in_probability_space() {
        let row = [0.5f32, -1.0, 2.0];
        let total: f32 = log_softmax_row(&row).iter().map(|v| v.exp()).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}
