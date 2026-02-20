use crate::{crypto, gpu_crypto, solver};
use anyhow::Result;
use std::fmt::Write as FmtWrite;
use std::path::Path;
use std::time::Instant;

struct Case {
    name: &'static str,
    pubkey: &'static str,
    start: &'static str,
    range_bits: u32,
}

const CASES: &[Case] = &[
    Case {
        name: "32-bit",
        pubkey: "03e9e661838a96a65331637e2a3e948dc0756e5009e7cb5c36664d9b72dd18c0a7",
        start: "80000000",
        range_bits: 32,
    },
    Case {
        name: "40-bit",
        pubkey: "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4",
        start: "8000000000",
        range_bits: 40,
    },
    Case {
        name: "48-bit",
        pubkey: "026864513503daca97ffae5d13d784192f932f304677b9a67a48a41af53f88ad19",
        start: "800000000000",
        range_bits: 48,
    },
];

struct CaseResult {
    name: &'static str,
    time_secs: f64,
    total_ops: u64,
    rate: f64,
}

pub fn run(gpu_index: u32, backend: gpu_crypto::GpuBackend, save_to_markdown: bool) -> Result<()> {
    println!("Kangaroo Benchmark Suite");
    println!("========================\n");

    let ctx = pollster::block_on(gpu_crypto::GpuContext::new(gpu_index, backend))?;
    let device_name = ctx.device_name().to_string();
    let compute_units = ctx.compute_units();

    println!("GPU: {device_name}");
    println!("Compute units: {compute_units}\n");

    println!(
        "{:<10} {:>12} {:>12} {:>14}",
        "Range", "Time", "Ops", "Rate"
    );
    println!("{}", "-".repeat(52));

    let num_k = ctx.optimal_kangaroos();
    let mut results = Vec::with_capacity(CASES.len());

    for case in CASES {
        let pubkey = crypto::parse_pubkey(case.pubkey)?;
        let start = crypto::parse_hex_u256(case.start)?;

        let dp_bits = (case.range_bits / 2)
            .saturating_sub((num_k as f64).log2() as u32 / 2)
            .clamp(8, 40);

        let mut solver = solver::KangarooSolver::new(
            ctx.clone(),
            pubkey,
            start,
            case.range_bits,
            dp_bits,
            num_k,
        )?;

        let t0 = Instant::now();
        loop {
            if solver.step()?.is_some() {
                break;
            }
        }
        let duration = t0.elapsed();
        let total_ops = solver.total_operations();
        let time_secs = duration.as_secs_f64();
        let rate = total_ops as f64 / time_secs;

        println!(
            "{:<10} {:>10.2}s {:>12} {:>12.2}M/s",
            case.name,
            time_secs,
            total_ops,
            rate / 1_000_000.0
        );

        results.push(CaseResult {
            name: case.name,
            time_secs,
            total_ops,
            rate,
        });
    }

    if save_to_markdown {
        let version = env!("CARGO_PKG_VERSION");
        save_to_file(&device_name, version, &results)?;
        println!("\nResults saved to BENCHMARKS.md");
    }

    Ok(())
}

fn save_to_file(device_name: &str, version: &str, results: &[CaseResult]) -> Result<()> {
    let benchmarks_path = Path::new("BENCHMARKS.md");
    let existing = if benchmarks_path.exists() {
        std::fs::read_to_string(benchmarks_path)?
    } else {
        String::new()
    };

    let new_content = generate_markdown(device_name, version, results, &existing);
    std::fs::write(benchmarks_path, new_content)?;
    Ok(())
}

fn generate_markdown(
    device_name: &str,
    version: &str,
    results: &[CaseResult],
    existing: &str,
) -> String {
    let section_header = format!("### {device_name}");
    let new_section = format_gpu_section(device_name, version, results);

    let mut content = if existing.is_empty() {
        format_fresh_file(&new_section)
    } else if let Some(range) = find_gpu_section(existing, &section_header) {
        let mut out = String::with_capacity(existing.len());
        out.push_str(&existing[..range.0]);
        out.push_str(&new_section);
        out.push_str(&existing[range.1..]);
        out
    } else if let Some(pos) = find_insert_position(existing) {
        let mut out = String::with_capacity(existing.len() + new_section.len());
        out.push_str(&existing[..pos]);
        out.push_str(&new_section);
        out.push('\n');
        out.push_str(&existing[pos..]);
        out
    } else {
        let mut out = existing.to_string();
        out.push_str("\n## Results\n\n");
        out.push_str(&new_section);
        out
    };

    let rate_48 = results
        .iter()
        .find(|r| r.name == "48-bit")
        .map(|r| r.rate / 1_000_000.0);

    if let Some(rate) = rate_48 {
        content = update_performance_history(&content, version, rate);
    }

    content
}

fn format_fresh_file(gpu_section: &str) -> String {
    let mut out = String::new();
    out.push_str("# Benchmark Results\n\n");
    out.push_str("Run benchmarks on your hardware:\n\n");
    out.push_str("```bash\nkangaroo --benchmark --save-benchmarks\n```\n\n");
    out.push_str("## Results\n\n");
    out.push_str(gpu_section);
    out.push_str("\n## Contributing\n\n");
    out.push_str(
        "Have different hardware? Run `kangaroo --benchmark --save-benchmarks` and submit a PR with your results!\n",
    );
    out
}

fn format_gpu_section(device_name: &str, version: &str, results: &[CaseResult]) -> String {
    let mut s = String::new();
    writeln!(s, "### {device_name}").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "| Range | Time | Ops | Rate |").unwrap();
    writeln!(s, "|-------|------|-----|------|").unwrap();
    for r in results {
        writeln!(
            s,
            "| {} | {:.2}s | {} | {:.2} M/s |",
            r.name,
            r.time_secs,
            format_ops(r.total_ops),
            r.rate / 1_000_000.0
        )
        .unwrap();
    }
    writeln!(s).unwrap();
    writeln!(s, "*Version: {version}*").unwrap();
    writeln!(s).unwrap();
    s
}

fn format_ops(ops: u64) -> String {
    let s = ops.to_string();
    let bytes = s.as_bytes();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(b as char);
    }
    result
}

fn update_performance_history(content: &str, version: &str, rate_m: f64) -> String {
    let version_tag = format!("v{version}");
    let section_header = "### Performance History";

    if let Some(range) = find_gpu_section(content, section_header) {
        let section = &content[range.0..range.1];
        let rows = parse_history_rows(section);
        let baseline = rows.first().map(|r| r.rate).unwrap_or(rate_m);
        let new_rows = upsert_history_row(rows, &version_tag, rate_m, baseline);
        let new_section = format_history_section(&new_rows, baseline);

        let mut out = String::with_capacity(content.len());
        out.push_str(&content[..range.0]);
        out.push_str(&new_section);
        out.push_str(&content[range.1..]);
        out
    } else if let Some(pos) = content.find("\n## Contributing") {
        let insert_at = pos + 1;
        let new_section = format_history_section(
            &[HistoryRow {
                version: version_tag,
                rate: rate_m,
            }],
            rate_m,
        );

        let mut out = String::with_capacity(content.len() + new_section.len());
        out.push_str(&content[..insert_at]);
        out.push_str(&new_section);
        out.push_str(&content[insert_at..]);
        out
    } else {
        let new_section = format_history_section(
            &[HistoryRow {
                version: version_tag,
                rate: rate_m,
            }],
            rate_m,
        );
        let mut out = content.to_string();
        out.push_str(&new_section);
        out
    }
}

struct HistoryRow {
    version: String,
    rate: f64,
}

fn parse_history_rows(section: &str) -> Vec<HistoryRow> {
    let mut rows = Vec::new();
    for line in section.lines() {
        if !line.starts_with('|') || line.starts_with("| Version") || line.starts_with("|---") {
            continue;
        }
        let cols: Vec<&str> = line.split('|').collect();
        if cols.len() < 4 {
            continue;
        }
        let version = cols[1].trim().to_string();
        let rate_str = cols[2].trim();
        if let Some(rate) = rate_str
            .strip_suffix(" M/s")
            .and_then(|s| s.trim().parse::<f64>().ok())
        {
            rows.push(HistoryRow { version, rate });
        }
    }
    rows
}

fn upsert_history_row(
    mut rows: Vec<HistoryRow>,
    version: &str,
    rate: f64,
    _baseline: f64,
) -> Vec<HistoryRow> {
    if let Some(existing) = rows.iter_mut().find(|r| r.version == version) {
        existing.rate = rate;
    } else {
        rows.push(HistoryRow {
            version: version.to_string(),
            rate,
        });
    }
    rows
}

fn format_history_section(rows: &[HistoryRow], baseline: f64) -> String {
    let mut s = String::new();
    writeln!(s, "### Performance History").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "| Version | 48-bit Rate | Improvement |").unwrap();
    writeln!(s, "|---------|-------------|-------------|").unwrap();
    for row in rows {
        let improvement = if (row.rate - baseline).abs() < 0.01 {
            "baseline".to_string()
        } else {
            let pct = ((row.rate - baseline) / baseline) * 100.0;
            if pct >= 0.0 {
                format!("+{pct:.0}%")
            } else {
                format!("{pct:.0}%")
            }
        };
        writeln!(
            s,
            "| {} | {:.2} M/s | {} |",
            row.version, row.rate, improvement
        )
        .unwrap();
    }
    writeln!(s).unwrap();
    s
}

/// Find the byte range of an existing GPU section (from ### header to next ### or ##)
fn find_gpu_section(content: &str, section_header: &str) -> Option<(usize, usize)> {
    let mut search_from = 0;
    while let Some(pos) = content[search_from..].find(section_header) {
        let start = search_from + pos;
        let after_header = start + section_header.len();

        let at_line_start = start == 0 || content.as_bytes()[start - 1] == b'\n';
        let at_line_end =
            after_header >= content.len() || content.as_bytes()[after_header] == b'\n';

        if at_line_start && at_line_end {
            let rest = &content[after_header..];
            let next_h3 = rest.find("\n### ");
            let next_h2 = rest.find("\n## ");
            let end = [next_h3, next_h2]
                .into_iter()
                .flatten()
                .min()
                .map(|p| after_header + p + 1)
                .unwrap_or(content.len());

            return Some((start, end));
        }

        search_from = after_header;
    }
    None
}

/// Find position after "## Results\n\n" to insert a new GPU section
fn find_insert_position(content: &str) -> Option<usize> {
    let marker = "## Results\n\n";
    content.find(marker).map(|pos| pos + marker.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_results() -> Vec<CaseResult> {
        vec![
            CaseResult {
                name: "32-bit",
                time_secs: 0.25,
                total_ops: 4_194_304,
                rate: 16_780_000.0,
            },
            CaseResult {
                name: "48-bit",
                time_secs: 13.0,
                total_ops: 222_298_112,
                rate: 17_100_000.0,
            },
        ]
    }

    #[test]
    fn generates_fresh_file() {
        let section = format_gpu_section("Test GPU", "0.6.0", &sample_results());
        let content = format_fresh_file(&section);
        assert!(content.contains("# Benchmark Results"));
        assert!(content.contains("### Test GPU"));
        assert!(content.contains("*Version: 0.6.0*"));
        assert!(content.contains("| 32-bit |"));
        assert!(content.contains("## Contributing"));
    }

    #[test]
    fn replaces_existing_gpu_section() {
        let existing = "\
# Benchmark Results

## Results

### Old GPU

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 32-bit | 1.00s | 100 | 1.00 M/s |

*Version: 0.1.0*

### Other GPU

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 32-bit | 2.00s | 200 | 2.00 M/s |

*Version: 0.1.0*

## Contributing

Submit your results!
";

        let content = generate_markdown("Old GPU", "0.6.0", &sample_results(), existing);
        assert!(content.contains("*Version: 0.6.0*"));
        assert!(content.contains("### Other GPU"));
        assert!(content.contains("2.00 M/s"));
        assert!(content.contains("## Contributing"));
        assert_eq!(content.matches("### Old GPU").count(), 1);
    }

    #[test]
    fn inserts_new_gpu_section() {
        let existing = "\
# Benchmark Results

## Results

### Existing GPU

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 32-bit | 1.00s | 100 | 1.00 M/s |

*Version: 0.1.0*

## Contributing

Submit your results!
";

        let content = generate_markdown("New GPU", "0.6.0", &sample_results(), existing);
        assert!(content.contains("### New GPU"));
        assert!(content.contains("### Existing GPU"));
        let new_pos = content.find("### New GPU").unwrap();
        let existing_pos = content.find("### Existing GPU").unwrap();
        assert!(new_pos < existing_pos);
    }

    #[test]
    fn updates_existing_performance_history() {
        let existing = "\
# Benchmark Results

## Results

### Test GPU

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 48-bit | 10.00s | 100,000,000 | 10.00 M/s |

*Version: 0.5.0*

### Performance History

| Version | 48-bit Rate | Improvement |
|---------|-------------|-------------|
| v0.2.0 | 3.70 M/s | baseline |
| v0.5.0 | 8.84 M/s | +139% |

## Contributing

Submit your results!
";

        let content = generate_markdown("Test GPU", "0.6.0", &sample_results(), existing);
        assert!(content.contains("v0.6.0"));
        assert!(content.contains("17.10 M/s"));
        assert_eq!(content.matches("v0.2.0").count(), 1);
        assert!(content.contains("baseline"));
    }

    #[test]
    fn upserts_same_version_in_history() {
        let existing = "\
# Benchmark Results

## Results

### Performance History

| Version | 48-bit Rate | Improvement |
|---------|-------------|-------------|
| v0.2.0 | 3.70 M/s | baseline |
| v0.5.0 | 8.84 M/s | +139% |

## Contributing

Submit your results!
";

        let content = generate_markdown("New GPU", "0.5.0", &sample_results(), existing);
        assert!(content.contains("*Version: 0.5.0*"));
        assert_eq!(content.matches("v0.5.0").count(), 1);
        assert!(content.contains("17.10 M/s"));
    }

    #[test]
    fn creates_performance_history_when_missing() {
        let existing = "\
# Benchmark Results

## Results

### Test GPU

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 48-bit | 10.00s | 100,000,000 | 10.00 M/s |

*Version: 0.4.0*

## Contributing

Submit your results!
";

        let content = generate_markdown("Test GPU", "0.5.0", &sample_results(), existing);
        assert!(content.contains("### Performance History"));
        assert!(content.contains("v0.5.0"));
        assert!(content.contains("baseline"));
    }

    #[test]
    fn does_not_match_substring_gpu_header() {
        let existing = "\
# Benchmark Results

## Results

### GPU ABC

| Range | Time | Ops | Rate |
|-------|------|-----|------|
| 48-bit | 1.00s | 100 | 1.00 M/s |

*Version: 0.1.0*

## Contributing

Submit your results!
";

        let content = generate_markdown("GPU A", "0.6.0", &sample_results(), existing);
        assert!(content.contains("### GPU A\n"));
        assert!(content.contains("### GPU ABC"));
        assert!(content.contains("1.00 M/s"));
    }

    #[test]
    fn formats_ops_with_separators() {
        assert_eq!(format_ops(0), "0");
        assert_eq!(format_ops(999), "999");
        assert_eq!(format_ops(1_000), "1,000");
        assert_eq!(format_ops(4_194_304), "4,194,304");
        assert_eq!(format_ops(222_298_112), "222,298,112");
        assert_eq!(format_ops(1_222_298_112), "1,222,298,112");
        assert_eq!(format_ops(10_000_000_000), "10,000,000,000");
    }
}
