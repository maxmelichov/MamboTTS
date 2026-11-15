//! Fetching one registry file: resumable, and verified before it gets its name.
//!
//! A file is written as `<name>.part`. A dropped connection or a closed app
//! leaves the part on disk, and the next attempt asks the server for the rest
//! of it (`Range: bytes=<have>-`) instead of starting over, which matters for
//! the 1.2 GB fp32 RenikudPlus. Once the part has the registry's exact size,
//! its SHA-256 is checked against the registry's, and only a match is renamed
//! to the real name. A file under its real name is therefore always complete.

use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use mambotts_registry::ModelFile;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

/// Where the bytes of `dest` collect until they are verified.
pub fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    name.push(".part");
    dest.with_file_name(name)
}

/// Whether `path` is the registry's file, by size alone (cheap enough to ask
/// at every start; the hash was checked when the file got its name).
pub fn has_expected_size(path: &Path, file: &ModelFile) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && metadata.len() == file.size)
}

/// Fetch `file` to `dest`, resuming a part left by an earlier attempt.
///
/// `on_progress` is called with the bytes of this file on disk so far
/// (resumed bytes included), so a caller can add it to the bundle's total.
pub async fn fetch_verified(
    client: &reqwest::Client,
    file: &ModelFile,
    dest: &Path,
    mut on_progress: impl FnMut(u64),
) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let part = part_path(dest);
    // Two passes at most: a part the server will not resume (or one that
    // turns out corrupt) is dropped and fetched once more from the start.
    for attempt in 0..2 {
        let have = match tokio::fs::metadata(&part).await {
            Ok(metadata) if metadata.len() <= file.size => metadata.len(),
            Ok(_) => {
                // Longer than the file can be: not a part of this revision.
                remove(&part).await?;
                0
            }
            Err(_) => 0,
        };
        on_progress(have);
        if have < file.size {
            download_rest(client, file, &part, have, &mut on_progress).await?;
        }
        let len = tokio::fs::metadata(&part)
            .await
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        if len != file.size {
            if attempt == 0 && len > file.size {
                remove(&part).await?;
                continue;
            }
            // Short: the stream ended early. The part stays for the next try.
            return Err(format!(
                "{} downloaded {len} of {} bytes; try again to resume",
                file.name, file.size
            ));
        }
        let digest = sha256_file(&part).await?;
        if digest == file.sha256 {
            return tokio::fs::rename(&part, dest).await.map_err(|err| {
                format!(
                    "failed to move {} to {}: {err}",
                    part.display(),
                    dest.display()
                )
            });
        }
        remove(&part).await?;
        if attempt == 1 {
            return Err(format!(
                "{} failed verification: SHA-256 {digest}, expected {}",
                file.name, file.sha256
            ));
        }
    }
    unreachable!("the loop returns on its second pass")
}

/// Append the bytes after `have` to `part`, or rewrite it from the start when
/// the server answers a range request with the whole file.
async fn download_rest(
    client: &reqwest::Client,
    file: &ModelFile,
    part: &Path,
    have: u64,
    on_progress: &mut impl FnMut(u64),
) -> Result<(), String> {
    let mut request = client.get(file.url);
    if have > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let response = request
        .send()
        .await
        .map_err(|err| format!("failed to download {}: {err}", file.url))?;
    let status = response.status();
    let resumed = have > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
    if !(status.is_success() || resumed) {
        return Err(format!("download of {} failed: {status}", file.url));
    }
    if resumed && content_range_start(&response) != Some(have) {
        return Err(format!(
            "{} answered a resume from byte {have} with another range",
            file.url
        ));
    }
    let mut written = if resumed { have } else { 0 };
    let mut out = if resumed {
        tokio::fs::OpenOptions::new().append(true).open(part).await
    } else {
        tokio::fs::File::create(part).await
    }
    .map_err(|err| format!("failed to open {}: {err}", part.display()))?;
    let mut stream = response.bytes_stream();
    let mut broke = None;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(err) => {
                broke = Some(format!(
                    "download of {} broke off: {err}; try again to resume",
                    file.url
                ));
                break;
            }
        };
        out.write_all(&chunk)
            .await
            .map_err(|err| format!("failed to write {}: {err}", part.display()))?;
        written += chunk.len() as u64;
        on_progress(written);
        if written > file.size {
            break;
        }
    }
    // Flushed on every exit, the broken one included: tokio writes in the
    // background, and a part whose last bytes land after the next attempt
    // measured it would be resumed from the wrong offset.
    out.flush()
        .await
        .map_err(|err| format!("failed to flush {}: {err}", part.display()))?;
    match broke {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

fn content_range_start(response: &reqwest::Response) -> Option<u64> {
    // "bytes 100-199/200"
    response
        .headers()
        .get(reqwest::header::CONTENT_RANGE)?
        .to_str()
        .ok()?
        .trim()
        .strip_prefix("bytes ")?
        .split('-')
        .next()?
        .parse()
        .ok()
}

/// Lowercase hex SHA-256 of a file, off the async threads.
pub async fn sha256_file(path: &Path) -> Result<String, String> {
    let shown = path.display().to_string();
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || {
        use std::io::Read;
        let mut file = std::fs::File::open(&path)
            .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    })
    .await
    .map_err(|err| format!("hashing {shown} stopped: {err}"))?
}

async fn remove(path: &Path) -> Result<(), String> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(format!("failed to remove {}: {err}", path.display())),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! Against a small HTTP server on localhost that serves one body, honours
    //! `Range`, and can be told to cut a response short or ignore ranges.

    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::net::TcpListener;

    #[derive(Clone, Default)]
    pub struct ServerOptions {
        /// Stop each 200/206 body after this many bytes (simulates a drop).
        pub cut_after: Option<usize>,
        /// Answer every request with the whole body and 200.
        pub ignore_range: bool,
    }

    pub struct TestServer {
        pub url: String,
        pub ranges: Arc<Mutex<Vec<Option<String>>>>,
        pub requests: Arc<AtomicUsize>,
        pub options: Arc<Mutex<ServerOptions>>,
    }

    pub async fn serve(body: Vec<u8>, options: ServerOptions) -> TestServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/model.onnx", listener.local_addr().unwrap());
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let requests = Arc::new(AtomicUsize::new(0));
        let options = Arc::new(Mutex::new(options));
        let body = Arc::new(body);
        let (r, q, o) = (ranges.clone(), requests.clone(), options.clone());
        tokio::spawn(async move {
            loop {
                let (socket, _) = listener.accept().await.unwrap();
                let (body, ranges, requests, options) =
                    (body.clone(), r.clone(), q.clone(), o.clone());
                tokio::spawn(async move {
                    let (read, mut write) = socket.into_split();
                    let mut reader = BufReader::new(read);
                    let mut range = None;
                    let mut line = String::new();
                    loop {
                        line.clear();
                        if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
                            return;
                        }
                        let header = line.trim_end();
                        if header.is_empty() {
                            break;
                        }
                        if let Some(value) = header
                            .to_ascii_lowercase()
                            .strip_prefix("range: bytes=")
                            .map(str::to_owned)
                        {
                            range = Some(value);
                        }
                    }
                    requests.fetch_add(1, Ordering::SeqCst);
                    ranges.lock().unwrap().push(range.clone());
                    let options = options.lock().unwrap().clone();
                    let start = range
                        .filter(|_| !options.ignore_range)
                        .and_then(|value| value.trim_end_matches('-').parse::<usize>().ok());
                    let (status, from) = match start {
                        Some(start) => (
                            format!(
                                "206 Partial Content\r\nContent-Range: bytes {start}-{}/{}",
                                body.len() - 1,
                                body.len()
                            ),
                            start,
                        ),
                        None => ("200 OK".to_owned(), 0),
                    };
                    let rest = &body[from..];
                    let head = format!(
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        rest.len()
                    );
                    let _ = write.write_all(head.as_bytes()).await;
                    let send = options
                        .cut_after
                        .map_or(rest.len(), |cut| cut.min(rest.len()));
                    let _ = write.write_all(&rest[..send]).await;
                    let _ = write.shutdown().await;
                });
            }
        });
        TestServer {
            url,
            ranges,
            requests,
            options,
        }
    }

    pub fn body(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 31 % 251) as u8).collect()
    }

    pub fn sha256_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    pub fn model_file(url: &str, bytes: &[u8]) -> ModelFile {
        ModelFile {
            name: "renikud-plus-fp32.onnx",
            url: Box::leak(url.to_owned().into_boxed_str()),
            size: bytes.len() as u64,
            sha256: Box::leak(sha256_hex(bytes).into_boxed_str()),
        }
    }

    #[tokio::test]
    async fn a_dropped_download_resumes_where_it_stopped() {
        let bytes = body(300_000);
        let server = serve(
            bytes.clone(),
            ServerOptions {
                cut_after: Some(100_000),
                ..Default::default()
            },
        )
        .await;
        let file = model_file(&server.url, &bytes);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(file.name);
        let client = reqwest::Client::new();

        // First attempt: the connection drops after 100 kB. The part stays,
        // the real name does not appear.
        let err = fetch_verified(&client, &file, &dest, |_| {})
            .await
            .unwrap_err();
        assert!(err.contains("resume"), "{err}");
        assert!(!dest.exists());
        // What arrived before the cut (at most what the server sent).
        let have = std::fs::metadata(part_path(&dest)).unwrap().len();
        assert!(have > 0 && have <= 100_000, "{have}");

        // Then the connection holds: the second request asks only for the rest.
        server.options.lock().unwrap().cut_after = None;
        let mut seen = Vec::new();
        fetch_verified(&client, &file, &dest, |n| seen.push(n))
            .await
            .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), bytes);
        assert!(!part_path(&dest).exists());
        assert_eq!(
            server.ranges.lock().unwrap().clone(),
            vec![None, Some(format!("{have}-"))]
        );
        assert_eq!(seen.first(), Some(&have));
        assert_eq!(seen.last(), Some(&300_000));
    }

    #[tokio::test]
    async fn a_server_that_ignores_ranges_restarts_the_file() {
        let bytes = body(50_000);
        let server = serve(
            bytes.clone(),
            ServerOptions {
                ignore_range: true,
                ..Default::default()
            },
        )
        .await;
        let file = model_file(&server.url, &bytes);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(file.name);
        std::fs::write(part_path(&dest), &bytes[..20_000]).unwrap();
        fetch_verified(&reqwest::Client::new(), &file, &dest, |_| {})
            .await
            .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), bytes);
    }

    #[tokio::test]
    async fn a_corrupt_file_never_gets_its_name() {
        let bytes = body(40_000);
        let mut wrong = bytes.clone();
        wrong[123] ^= 0xff;
        // The server serves the corrupt bytes; the registry knows the real hash.
        let server = serve(wrong, ServerOptions::default()).await;
        let file = model_file(&server.url, &bytes);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(file.name);
        let err = fetch_verified(&reqwest::Client::new(), &file, &dest, |_| {})
            .await
            .unwrap_err();
        assert!(err.contains("SHA-256"), "{err}");
        assert!(!dest.exists());
        assert!(!part_path(&dest).exists());
        // One retry from scratch, then it gives up.
        assert_eq!(server.requests.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_complete_part_is_verified_without_a_request() {
        let bytes = body(10_000);
        let server = serve(bytes.clone(), ServerOptions::default()).await;
        let file = model_file(&server.url, &bytes);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(file.name);
        std::fs::write(part_path(&dest), &bytes).unwrap();
        fetch_verified(&reqwest::Client::new(), &file, &dest, |_| {})
            .await
            .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), bytes);
        assert_eq!(server.requests.load(Ordering::SeqCst), 0);
    }
}
