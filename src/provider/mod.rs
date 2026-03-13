//! Data provider system for puzzle sources
//!
//! Providers supply puzzle data (pubkey, key range) from external sources.
//! Format: `provider:path` (e.g., `boha:b1000/135`)

#[cfg(feature = "boha")]
mod boha;

#[cfg(not(feature = "boha"))]
use anyhow::anyhow;
use anyhow::Result;

/// Result from resolving a provider reference
#[derive(Debug, Clone)]
pub struct ProviderResult {
    /// Puzzle identifier (e.g., "b1000/135")
    pub id: String,

    /// Compressed public key (33 bytes hex)
    pub pubkey: Option<String>,

    /// Start of key range (hex, without 0x prefix)
    pub start: Option<String>,

    /// End of key range (hex, without 0x prefix) - used for validation
    #[allow(dead_code)]
    pub end: Option<String>,

    /// Key range in bits
    pub range_bits: Option<u32>,
}

/// Resolve a provider reference to puzzle data
///
/// Format: `provider:path`
/// - `boha:b1000/135` - Bitcoin puzzle #135 from boha
///
/// Returns `Ok(None)` if input is not a provider reference.
/// Returns `Err` if provider is recognized but resolution fails.
pub fn resolve(input: &str) -> Result<Option<ProviderResult>> {
    let Some((provider, query)) = input.split_once(':') else {
        return Ok(None);
    };

    // Skip Windows drive paths (single char before colon)
    if provider.len() == 1 {
        return Ok(None);
    }

    match provider {
        #[cfg(feature = "boha")]
        "boha" => boha::resolve(query).map(Some),

        #[cfg(not(feature = "boha"))]
        "boha" => Err(anyhow!(
            "boha provider requires the 'boha' feature. Rebuild with: cargo build --features boha"
        )),

        _ => {
            let _ = query;
            Ok(None)
        }
    }
}

/// List available puzzles from all providers
///
/// Returns tuples of (provider, id, address, range_bits, has_pubkey)
#[cfg(feature = "boha")]
pub fn list_available() -> Vec<(String, String, String, Option<u32>, bool)> {
    boha::list_available()
}

#[cfg(not(feature = "boha"))]
pub fn list_available() -> Vec<(String, String, String, Option<u32>, bool)> {
    vec![]
}

/// Get list of supported provider names
pub fn supported_providers() -> Vec<&'static str> {
    #[cfg(feature = "boha")]
    {
        vec!["boha"]
    }
    #[cfg(not(feature = "boha"))]
    {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_not_provider() {
        let result = resolve("just-a-string").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_windows_path() {
        let result = resolve("C:\\Users\\test").unwrap();
        assert!(result.is_none());
    }
}
