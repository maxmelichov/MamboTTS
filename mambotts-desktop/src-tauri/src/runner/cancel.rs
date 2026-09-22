use std::{
    collections::HashMap,
    sync::{Mutex, MutexGuard},
};

use tokio::sync::watch;

/// The error text a cancelled synthesis rejects with. The webview matches on
/// it to tell a Stop apart from a failure.
pub const SYNTHESIS_CANCELLED: &str = "synthesis cancelled";

/// Syntheses the webview can stop, keyed by an id the webview chooses.
///
/// The webview picks the id so it can ask to cancel before the `synthesize`
/// command has even registered: a cancel that arrives first is remembered and
/// the synthesis stops as soon as it starts.
#[derive(Default)]
pub struct SynthesisRegistry {
    active: Mutex<HashMap<String, watch::Sender<bool>>>,
}

impl SynthesisRegistry {
    fn active(&self) -> MutexGuard<'_, HashMap<String, watch::Sender<bool>>> {
        self.active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Start tracking `id` and return a receiver that turns true on cancel.
    pub fn register(&self, id: &str) -> watch::Receiver<bool> {
        self.active()
            .entry(id.to_owned())
            .or_insert_with(|| watch::channel(false).0)
            .subscribe()
    }

    pub fn cancel(&self, id: &str) {
        self.active()
            .entry(id.to_owned())
            .or_insert_with(|| watch::channel(false).0)
            .send_replace(true);
    }

    pub fn finish(&self, id: &str) {
        self.active().remove(id);
    }
}

#[cfg(test)]
mod tests {
    use super::SynthesisRegistry;

    #[test]
    fn cancel_reaches_a_running_synthesis() {
        let registry = SynthesisRegistry::default();
        let receiver = registry.register("a");
        let other = registry.register("b");
        registry.cancel("a");
        assert!(*receiver.borrow());
        assert!(!*other.borrow());
    }

    #[test]
    fn cancel_before_start_is_remembered() {
        let registry = SynthesisRegistry::default();
        registry.cancel("early");
        assert!(*registry.register("early").borrow());
        registry.finish("early");
        assert!(!*registry.register("early").borrow());
    }
}
