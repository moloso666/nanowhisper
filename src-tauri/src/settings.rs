use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const PROVIDER_OPENAI: &str = "openai";
pub const PROVIDER_GEMINI: &str = "gemini";
pub const PROVIDER_CUSTOM_OPENAI: &str = "custom_openai";
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_api_key")]
    pub api_key: String,
    #[serde(default)]
    pub gemini_api_key: String,
    #[serde(default)]
    pub custom_api_key: String,
    #[serde(default)]
    pub custom_base_url: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_language")]
    pub language: String,
    /// Free-form context sent with each transcription (OpenAI `prompt`
    /// field; appended to the Gemini instruction). Empty = not sent.
    #[serde(default)]
    pub prompt: String,
    #[serde(default = "default_proxy_mode")]
    pub proxy_mode: String,
    #[serde(default)]
    pub proxy_url: String,
    #[serde(default = "default_shortcut")]
    pub shortcut: String,
    #[serde(default = "default_sound_enabled")]
    pub sound_enabled: bool,
    /// Minimum macOS mic input gain (0–100) to restore at recording start.
    /// 0 disables the feature. See `mic_gain::ensure_min_gain`.
    #[serde(default = "default_mic_min_gain")]
    pub mic_min_gain: u8,
    #[serde(default)]
    pub overlay_rx: Option<f64>,
    #[serde(default)]
    pub overlay_ry: Option<f64>,
}

fn default_provider() -> String {
    PROVIDER_OPENAI.to_string()
}
fn default_api_key() -> String {
    String::new()
}
fn default_model() -> String {
    "gpt-4o-transcribe".to_string()
}
fn default_language() -> String {
    "auto".to_string()
}
fn default_proxy_mode() -> String {
    "system".to_string()
}
fn default_shortcut() -> String {
    String::new()
}
fn default_sound_enabled() -> bool {
    true
}
fn default_mic_min_gain() -> u8 {
    85
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            api_key: default_api_key(),
            gemini_api_key: String::new(),
            custom_api_key: String::new(),
            custom_base_url: String::new(),
            model: default_model(),
            language: default_language(),
            prompt: String::new(),
            proxy_mode: default_proxy_mode(),
            proxy_url: String::new(),
            shortcut: default_shortcut(),
            sound_enabled: default_sound_enabled(),
            mic_min_gain: default_mic_min_gain(),
            overlay_rx: None,
            overlay_ry: None,
        }
    }
}

impl AppSettings {
    pub fn is_gemini(&self) -> bool {
        self.provider == PROVIDER_GEMINI
    }

    pub fn is_custom_openai(&self) -> bool {
        self.provider == PROVIDER_CUSTOM_OPENAI
    }

    pub fn active_api_key(&self) -> &str {
        match self.provider.as_str() {
            PROVIDER_GEMINI => &self.gemini_api_key,
            PROVIDER_CUSTOM_OPENAI => &self.custom_api_key,
            _ => &self.api_key,
        }
    }

    /// Trimmed custom prompt, or None when empty.
    pub fn active_prompt(&self) -> Option<&str> {
        let p = self.prompt.trim();
        (!p.is_empty()).then_some(p)
    }

    pub fn openai_base_url(&self) -> &str {
        if self.is_custom_openai() {
            &self.custom_base_url
        } else {
            DEFAULT_OPENAI_BASE_URL
        }
    }
}

fn settings_path() -> PathBuf {
    crate::data_dir().join("settings.json")
}

pub fn get_settings() -> AppSettings {
    let path = settings_path();
    match std::fs::read_to_string(&path) {
        Ok(content) => serde_json::from_str::<AppSettings>(&content).unwrap_or_default(),
        Err(_) => AppSettings::default(),
    }
}

pub fn save_settings(settings: &AppSettings) {
    let dir = crate::data_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("settings.json");
    if let Ok(json) = serde_json::to_string_pretty(settings) {
        let _ = std::fs::write(&path, json);
    }
}

#[cfg(test)]
mod tests {
    use super::AppSettings;

    #[test]
    fn loads_legacy_settings_without_prompt() {
        let json = r#"{"provider":"openai","api_key":"k","model":"gpt-4o-transcribe","language":"auto"}"#;
        let s: AppSettings = serde_json::from_str(json).unwrap();
        assert_eq!(s.prompt, "");
        assert_eq!(s.active_prompt(), None);
    }

    #[test]
    fn active_prompt_trims_whitespace() {
        let s = AppSettings { prompt: "  hello  ".into(), ..Default::default() };
        assert_eq!(s.active_prompt(), Some("hello"));
    }
}
