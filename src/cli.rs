//! CLI utilities for progress bars and tracing

use indicatif::ProgressStyle;
use tracing_subscriber::{fmt, EnvFilter};

/// Initialize tracing with optional verbosity.
pub fn init_tracing(verbose: bool, quiet: bool) {
    if quiet {
        return;
    }

    let filter = if verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };

    let _ = fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

/// Returns the standard progress bar style.
pub fn default_progress_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .expect("Invalid progress bar template")
        .progress_chars("#>-")
}

/// Returns the standard progress bar style with a message field.
pub fn default_progress_style_with_msg() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .expect("Invalid progress bar template")
        .progress_chars("#>-")
}
