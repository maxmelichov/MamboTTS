//! Reads the server's framed speech stream.
//!
//! Streamed chunks exist only so playback can start while inference runs.
//! They are handed straight to the webview in memory and never touch the
//! disk, so there is nothing to clean up after them. The only file this
//! writes is the finished recording, and that is removed again if the stream
//! does not complete: on a server error, a broken stream, or a cancel that
//! drops the read part-way.

use std::{
    fmt::Display,
    path::{Path, PathBuf},
};

use futures_util::{Stream, StreamExt};
use tokio::io::AsyncWriteExt;

/// Frames larger than this are treated as a corrupt stream.
const MAX_FRAME_BYTES: usize = 128 * 1024 * 1024;

/// Read frames until the finished recording is complete.
///
/// Every playable chunk goes to `on_chunk` as it arrives. Returns how many
/// chunks were delivered. The output file only survives a stream that ends
/// with the final frame; if this returns an error or the future is dropped,
/// whatever part of it was written is deleted.
pub async fn receive_speech_stream<S, B, E, F>(
    mut stream: S,
    output_path: &Path,
    mut on_chunk: F,
) -> Result<usize, String>
where
    S: Stream<Item = Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
    E: Display,
    F: FnMut(Vec<u8>) -> Result<(), String>,
{
    let mut output = PendingOutput::new(output_path);
    let mut pending = Vec::<u8>::new();
    let mut chunks = 0usize;
    let mut complete = false;

    while let Some(next) = stream.next().await {
        let bytes = next.map_err(|err| format!("failed to read speech stream: {err}"))?;
        pending.extend_from_slice(bytes.as_ref());

        while pending.len() >= 5 {
            let kind = pending[0];
            let length =
                u32::from_be_bytes([pending[1], pending[2], pending[3], pending[4]]) as usize;
            if length > MAX_FRAME_BYTES {
                return Err("received an invalidly large speech frame".to_string());
            }
            if pending.len() < 5 + length {
                break;
            }
            let payload = pending[5..5 + length].to_vec();
            pending.drain(..5 + length);

            match kind {
                1 => {
                    on_chunk(payload)?;
                    chunks += 1;
                }
                // The finished recording arrives in slices so neither side has
                // to hold a long one in memory as a single frame. Kind 4 is a
                // continuation; kind 2 is the last slice and the only thing
                // that marks the file complete.
                2 | 4 => {
                    output.write(&payload).await?;
                    complete = kind == 2;
                }
                3 => return Err(String::from_utf8_lossy(&payload).into_owned()),
                _ => return Err("received an unknown speech stream frame".to_string()),
            }
        }
    }

    if !pending.is_empty() {
        return Err("speech stream ended with an incomplete frame".to_string());
    }
    if !complete {
        return Err("speech stream ended before the final WAV was received".to_string());
    }
    output.keep();
    Ok(chunks)
}

/// The finished recording while it is being written. Dropping it without
/// calling [`PendingOutput::keep`] deletes the file, which covers errors,
/// early returns and a cancelled future alike.
struct PendingOutput {
    path: PathBuf,
    file: Option<tokio::fs::File>,
    keep: bool,
}

impl PendingOutput {
    fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            file: None,
            keep: false,
        }
    }

    async fn write(&mut self, payload: &[u8]) -> Result<(), String> {
        if self.file.is_none() {
            let file = tokio::fs::File::create(&self.path).await.map_err(|err| {
                format!("failed to open final audio {}: {err}", self.path.display())
            })?;
            self.file = Some(file);
        }
        let file = self.file.as_mut().expect("output file was just opened");
        // Flushing waits for tokio's background write to land, so the file is
        // complete when the stream reports success and has no write in flight
        // if it has to be deleted.
        let written = match file.write_all(payload).await {
            Ok(()) => file.flush().await,
            Err(err) => Err(err),
        };
        written.map_err(|err| format!("failed to write final audio {}: {err}", self.path.display()))
    }

    fn keep(mut self) {
        self.keep = true;
    }
}

impl Drop for PendingOutput {
    fn drop(&mut self) {
        // Close the handle first; Windows will not delete an open file.
        let opened = self.file.take().is_some();
        if opened && !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Delete chunk WAVs that older versions wrote beside each output file and
/// never removed (`mambotts-speech-<millis>-chunk-NNNN.wav`). They pile up in
/// the temp directory, which Windows does not sweep. Returns how many were
/// removed.
pub fn sweep_legacy_chunk_files(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| is_legacy_chunk_file(&entry.file_name().to_string_lossy()))
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
        .filter(|entry| std::fs::remove_file(entry.path()).is_ok())
        .count()
}

fn is_legacy_chunk_file(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("mambotts-speech-") else {
        return false;
    };
    let Some(stem) = rest.strip_suffix(".wav") else {
        return false;
    };
    let Some((millis, index)) = stem.split_once("-chunk-") else {
        return false;
    };
    !millis.is_empty()
        && millis.bytes().all(|byte| byte.is_ascii_digit())
        && index.len() == 4
        && index.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use std::{convert::Infallible, path::Path, time::Duration};

    use futures_util::{StreamExt, stream};

    use super::{is_legacy_chunk_file, receive_speech_stream, sweep_legacy_chunk_files};

    fn frame(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![kind];
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn files_in(dir: &Path) -> Vec<String> {
        let mut names: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    fn frames(
        items: Vec<Vec<u8>>,
    ) -> impl futures_util::Stream<Item = Result<Vec<u8>, Infallible>> + Unpin {
        stream::iter(items.into_iter().map(Ok))
    }

    #[tokio::test]
    async fn completed_generation_leaves_only_the_final_wav() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("mambotts-speech-1.wav");
        let mut delivered = Vec::new();
        // Split a frame across reads to exercise reassembly.
        let first = frame(1, b"chunk-a");
        let (head, tail) = first.split_at(3);
        let chunks = receive_speech_stream(
            frames(vec![
                head.to_vec(),
                tail.to_vec(),
                frame(1, b"chunk-b"),
                frame(4, b"final-"),
                frame(2, b"wav"),
            ]),
            &output,
            |chunk| {
                delivered.push(chunk);
                Ok(())
            },
        )
        .await
        .unwrap();

        assert_eq!(chunks, 2);
        assert_eq!(delivered, vec![b"chunk-a".to_vec(), b"chunk-b".to_vec()]);
        assert_eq!(std::fs::read(&output).unwrap(), b"final-wav");
        assert_eq!(files_in(dir.path()), vec!["mambotts-speech-1.wav"]);
    }

    #[tokio::test]
    async fn server_error_removes_the_partial_output() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("out.wav");
        let err = receive_speech_stream(
            frames(vec![
                frame(1, b"chunk"),
                frame(4, b"partial"),
                frame(3, b"engine exploded"),
            ]),
            &output,
            |_| Ok(()),
        )
        .await
        .unwrap_err();

        assert_eq!(err, "engine exploded");
        assert!(files_in(dir.path()).is_empty());
    }

    #[tokio::test]
    async fn truncated_stream_removes_the_partial_output() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("out.wav");
        let err = receive_speech_stream(
            frames(vec![frame(1, b"chunk"), frame(4, b"partial")]),
            &output,
            |_| Ok(()),
        )
        .await
        .unwrap_err();

        assert!(err.contains("before the final WAV"), "{err}");
        assert!(files_in(dir.path()).is_empty());
    }

    #[tokio::test]
    async fn chunk_delivery_failure_stops_and_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("out.wav");
        let err = receive_speech_stream(
            frames(vec![
                frame(4, b"partial"),
                frame(1, b"chunk"),
                frame(2, b"x"),
            ]),
            &output,
            |_| Err("webview went away".to_string()),
        )
        .await
        .unwrap_err();

        assert_eq!(err, "webview went away");
        assert!(files_in(dir.path()).is_empty());
    }

    #[tokio::test]
    async fn cancelling_mid_stream_removes_the_partial_output() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("out.wav");
        // The server has sent a chunk and part of the final recording, then
        // goes quiet; the reader is dropped the way a Stop drops it.
        let items = stream::iter(vec![
            Ok::<_, Infallible>(frame(1, b"chunk")),
            Ok(frame(4, b"partial")),
        ])
        .chain(stream::pending());
        let read = receive_speech_stream(items, &output, |_| Ok(()));
        let outcome = tokio::time::timeout(Duration::from_millis(100), read).await;

        assert!(
            outcome.is_err(),
            "the stream should still have been waiting"
        );
        assert!(files_in(dir.path()).is_empty());
    }

    /// End to end against a running server, which has to have a model loaded:
    /// `MAMBOTTS_TEST_SERVER=http://127.0.0.1:8791 cargo test -p mambotts --lib
    /// -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "needs MAMBOTTS_TEST_SERVER pointing at a server with a model loaded"]
    async fn a_real_generation_leaves_only_the_final_wav() {
        let base = std::env::var("MAMBOTTS_TEST_SERVER").expect("MAMBOTTS_TEST_SERVER");
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("mambotts-speech-1.wav");
        let client = reqwest::Client::new();

        let response = speech(&client, &base, "שלום עולם, זו בדיקה קצרה.").await;
        let mut sizes = Vec::new();
        let chunks = receive_speech_stream(response.bytes_stream(), &output, |chunk| {
            sizes.push(chunk.len());
            Ok(())
        })
        .await
        .unwrap();

        println!("chunks: {chunks} {sizes:?}, files: {:?}", files_in(dir.path()));
        assert!(chunks > 0);
        assert_eq!(files_in(dir.path()), vec!["mambotts-speech-1.wav"]);
        assert_eq!(&std::fs::read(&output).unwrap()[..4], b"RIFF");

        // Now stop one part-way, the way the Stop button does.
        let stopped_output = dir.path().join("mambotts-speech-2.wav");
        let long = "שלום לכם וברוכים הבאים לבדיקה של מנוע הדיבור. ".repeat(60);
        let response = speech(&client, &base, &long).await;
        let generation = response
            .headers()
            .get("x-mambotts-generation-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
            .expect("generation id header");
        let read = receive_speech_stream(response.bytes_stream(), &stopped_output, |_| Ok(()));
        let outcome = tokio::time::timeout(Duration::from_secs(45), read).await;
        assert!(outcome.is_err(), "the generation should still be running");
        client
            .post(format!("{base}/v1/audio/speech/cancel"))
            .json(&serde_json::json!({ "id": generation }))
            .send()
            .await
            .unwrap();
        println!("after the stop, files: {:?}", files_in(dir.path()));
        // The stopped take left nothing; only the finished one is still there.
        assert_eq!(files_in(dir.path()), vec!["mambotts-speech-1.wav"]);
    }

    #[cfg(test)]
    async fn speech(client: &reqwest::Client, base: &str, input: &str) -> reqwest::Response {
        let response = client
            .post(format!("{base}/v1/audio/speech"))
            .json(&serde_json::json!({
                "input": input,
                "voice": "Noa",
                "language": "he",
                "stream": true,
            }))
            .send()
            .await
            .expect("speech request");
        assert!(response.status().is_success(), "{:?}", response.status());
        response
    }

    #[test]
    fn sweep_removes_only_legacy_chunk_files() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "mambotts-speech-1726000000000-chunk-0000.wav",
            "mambotts-speech-1726000000000-chunk-0041.wav",
            "mambotts-speech-1726000000000.wav",
            "mambotts-speech-notes-chunk-0001.wav",
            "someone-else-chunk-0001.wav",
        ] {
            std::fs::write(dir.path().join(name), b"x").unwrap();
        }

        assert_eq!(sweep_legacy_chunk_files(dir.path()), 2);
        assert_eq!(
            files_in(dir.path()),
            vec![
                "mambotts-speech-1726000000000.wav",
                "mambotts-speech-notes-chunk-0001.wav",
                "someone-else-chunk-0001.wav",
            ]
        );
    }

    #[test]
    fn legacy_chunk_names() {
        assert!(is_legacy_chunk_file("mambotts-speech-12-chunk-0003.wav"));
        assert!(!is_legacy_chunk_file("mambotts-speech-12.wav"));
        assert!(!is_legacy_chunk_file("mambotts-speech--chunk-0003.wav"));
        assert!(!is_legacy_chunk_file("mambotts-speech-12-chunk-3.wav"));
    }
}
