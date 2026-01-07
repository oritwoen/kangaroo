//! Boha provider - crypto puzzles and bounties data

use super::ProviderResult;
use anyhow::{anyhow, Result};
use boha::Status;

pub fn resolve(path: &str) -> Result<ProviderResult> {
    let puzzle_id = path.replace(':', "/");

    let puzzle = boha::get(&puzzle_id).map_err(|e| anyhow!("{}", e))?;

    let pubkey: Option<String> = puzzle.pubkey_str().map(str::to_string);

    let (start, end, range_bits) = match puzzle.key_range_big() {
        Some((start_big, end_big)) => {
            let start_hex = format!("{:x}", start_big);
            let end_hex = format!("{:x}", end_big);
            let bits = puzzle.key.as_ref().and_then(|k| k.bits).map(|b| b as u32);
            (Some(start_hex), Some(end_hex), bits)
        }
        None => (None, None, None),
    };

    Ok(ProviderResult {
        id: puzzle_id,
        pubkey,
        start,
        end,
        range_bits,
    })
}

pub fn list_available() -> Vec<(String, String, String, Option<u32>, bool)> {
    boha::b1000::all()
        .filter(|p| p.status == Status::Unsolved)
        .map(|p| {
            let bits = p.key.as_ref().and_then(|k| k.bits).map(|b| b as u32);
            let has_pubkey = p.pubkey.is_some();
            (
                "boha".to_string(),
                p.id.to_string(),
                p.address.value.to_string(),
                bits,
                has_pubkey,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_valid_puzzle() {
        let result = resolve("b1000/66").unwrap();
        assert_eq!(result.id, "b1000/66");
        assert!(result.pubkey.is_some());
        assert!(result.start.is_some());
        assert!(result.end.is_some());
        assert_eq!(result.range_bits, Some(66));
    }

    #[test]
    fn test_resolve_colon_format() {
        let result = resolve("b1000:66").unwrap();
        assert_eq!(result.id, "b1000/66");
    }

    #[test]
    fn test_resolve_invalid_puzzle() {
        let result = resolve("b1000/9999");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_available_not_empty() {
        let list = list_available();
        assert!(!list.is_empty());

        for (provider, id, address, bits, _has_pubkey) in &list {
            assert_eq!(provider, "boha");
            assert!(id.starts_with("b1000/"));
            assert!(!address.is_empty());
            assert!(bits.is_some());
        }
    }
}
