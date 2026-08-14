#[cfg(unix)]
pub async fn watch_parent() {
    use std::time::Duration;

    let parent = std::os::unix::process::parent_id();
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if std::os::unix::process::parent_id() != parent {
            std::process::exit(0);
        }
    }
}

#[cfg(not(unix))]
pub async fn watch_parent() {}
