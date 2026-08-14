use std::path::Path;

pub fn first_non_empty(values: impl IntoIterator<Item = String>) -> String {
    values
        .into_iter()
        .find(|value| !value.is_empty())
        .unwrap_or_default()
}

pub fn first_non_zero(values: impl IntoIterator<Item = i32>) -> i32 {
    values
        .into_iter()
        .find(|value| *value != 0)
        .unwrap_or_default()
}

pub fn first_non_zero_float(values: impl IntoIterator<Item = f32>) -> f32 {
    values
        .into_iter()
        .find(|value| *value != 0.0)
        .unwrap_or_default()
}

pub fn empty_to_none(value: String) -> Option<String> {
    if value.is_empty() || value == "auto" {
        None
    } else {
        Some(value)
    }
}

pub fn path_option(value: &str) -> Option<&Path> {
    if value.is_empty() {
        None
    } else {
        Some(Path::new(value))
    }
}
