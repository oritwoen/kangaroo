//! GPU-accelerated Kangaroo solver
//!
//! Coordinates GPU compute with CPU collision detection.

use crate::cpu::init::{generate_jump_table, initialize_kangaroos};
use crate::cpu::DPTable;
use crate::crypto::{Point, U256};
use crate::gpu::{
    GpuBuffers, GpuConfig, GpuContext, GpuDistinguishedPoint, GpuKangaroo, KangarooPipeline,
};
use crate::math::create_dp_mask;
use anyhow::Result;
use std::time::Instant;
use tracing::info;

const MAX_DISTINGUISHED_POINTS: u32 = 65_536;
/// Target dispatch time in milliseconds (stay under TDR threshold)
const TARGET_DISPATCH_MS: u128 = 50;

/// Shared resources for batch mode (pipeline created once, reused)
#[allow(dead_code)]
pub struct SharedResources {
    pub ctx: GpuContext,
    pub pipeline: KangarooPipeline,
}

#[allow(dead_code)]
impl SharedResources {
    /// Create shared resources for batch mode
    pub fn new(ctx: GpuContext) -> Result<Self> {
        let pipeline = KangarooPipeline::new(&ctx)?;
        Ok(Self { ctx, pipeline })
    }
}

/// Main Kangaroo solver
pub struct KangarooSolver {
    ctx: GpuContext,
    pipeline: KangarooPipeline,
    buffers: GpuBuffers,
    dp_table: DPTable,
    total_ops: u64,
    num_kangaroos: u32,
    #[allow(dead_code)]
    steps_per_call: u32,
}

impl KangarooSolver {
    /// Create a new solver (with verbose logging)
    pub fn new(
        ctx: GpuContext,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
    ) -> Result<Self> {
        Self::new_internal(ctx, pubkey, start, range_bits, dp_bits, num_kangaroos, true)
    }

    #[allow(dead_code)]
    /// Create a solver from shared context (for batch mode - minimal logging)
    pub fn new_with_context(
        ctx: &GpuContext,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
    ) -> Result<Self> {
        // Clone is cheap - wgpu Device/Queue are Arc-wrapped
        Self::new_internal(
            ctx.clone(),
            pubkey,
            start,
            range_bits,
            dp_bits,
            num_kangaroos,
            false,
        )
    }

    #[allow(dead_code)]
    /// Create a solver with shared resources (pipeline reuse - fastest for batch mode)
    pub fn new_with_shared(
        shared: &SharedResources,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
    ) -> Result<Self> {
        Self::new_with_pipeline(
            &shared.ctx,
            &shared.pipeline,
            pubkey,
            start,
            range_bits,
            dp_bits,
            num_kangaroos,
        )
    }

    /// Select steps per GPU dispatch, respecting DP buffer capacity
    fn select_steps_per_call(
        optimal_steps: u32,
        num_kangaroos: u32,
        dp_bits: u32,
        max_dps: u32,
    ) -> u32 {
        if num_kangaroos == 0 || optimal_steps == 0 {
            return 0;
        }

        // 90% headroom keeps expected DP count below buffer capacity to avoid overflow
        let budgeted_dps = ((max_dps as u128) * 9 / 10).max(1);
        let dp_spacing = 1u128 << dp_bits;
        let num_k = num_kangaroos as u128;

        let allowed_steps = (budgeted_dps.saturating_mul(dp_spacing) / num_k).max(1);
        let capped_steps = allowed_steps.min(u128::from(u32::MAX)) as u32;
        capped_steps.min(optimal_steps)
    }

    #[allow(dead_code)]
    /// Create a solver with existing pipeline (no pipeline creation overhead)
    fn new_with_pipeline(
        ctx: &GpuContext,
        pipeline: &KangarooPipeline,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
    ) -> Result<Self> {
        let jump_table_size = 256u32;
        let (jump_points, jump_distances) = generate_jump_table(range_bits);

        // Create DP mask
        let dp_mask = create_dp_mask(dp_bits);

        // Config
        // Use optimal steps per call from context (calibrated for 2s limit)
        let steps_per_call = Self::select_steps_per_call(
            ctx.optimal_steps_per_call(),
            num_kangaroos,
            dp_bits,
            MAX_DISTINGUISHED_POINTS,
        );

        let config = GpuConfig {
            dp_mask_lo: [dp_mask[0], dp_mask[1], dp_mask[2], dp_mask[3]],
            dp_mask_hi: [dp_mask[4], dp_mask[5], dp_mask[6], dp_mask[7]],
            num_kangaroos,
            steps_per_call,
            jump_table_size,
            _padding: 0,
        };

        // Create buffers (reusing bind_group_layout from shared pipeline)
        let max_dps = MAX_DISTINGUISHED_POINTS;
        let buffers = GpuBuffers::new(
            ctx,
            pipeline,
            &config,
            &jump_points,
            &jump_distances,
            num_kangaroos,
            max_dps,
        )?;

        // Initialize kangaroos
        let kangaroos = initialize_kangaroos(&pubkey, &start, range_bits, num_kangaroos)?;
        upload_kangaroos(ctx, &buffers, &kangaroos)?;

        // Use start for key computation: k = start + tame_dist - wild_dist
        // Pass full 256-bit start to DPTable

        // Clone pipeline (wgpu types are Arc-wrapped, so this is cheap)
        let pipeline_clone = KangarooPipeline {
            pipeline: pipeline.pipeline.clone(),
            bind_group_layout: pipeline.bind_group_layout.clone(),
        };

        Ok(Self {
            ctx: ctx.clone(),
            pipeline: pipeline_clone,
            buffers,
            dp_table: DPTable::new(start),
            total_ops: 0,
            num_kangaroos,
            steps_per_call,
        })
    }

    fn new_internal(
        ctx: GpuContext,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
        verbose: bool,
    ) -> Result<Self> {
        if verbose {
            info!("Creating pipeline...");
        }
        let pipeline = KangarooPipeline::new(&ctx)?;
        if verbose {
            info!("Pipeline created");
        }

        if verbose {
            info!("Generating jump table...");
        }
        let jump_table_size = 256u32;
        let (jump_points, jump_distances) = generate_jump_table(range_bits);
        if verbose {
            info!("Jump table generated: {} entries", jump_table_size);
            for (i, dist) in jump_distances.iter().enumerate().take(4) {
                info!("Jump dist[{}] = 0x{:08x}", i, dist[0]);
            }
        }

        // Create DP mask
        let dp_mask = create_dp_mask(dp_bits);
        if verbose {
            info!("DP mask created");
        }

        // Config
        // Use optimal steps per call from context (calibrated for 2s limit)
        let steps_per_call = Self::select_steps_per_call(
            ctx.optimal_steps_per_call(),
            num_kangaroos,
            dp_bits,
            MAX_DISTINGUISHED_POINTS,
        );

        let config = GpuConfig {
            dp_mask_lo: [dp_mask[0], dp_mask[1], dp_mask[2], dp_mask[3]],
            dp_mask_hi: [dp_mask[4], dp_mask[5], dp_mask[6], dp_mask[7]],
            num_kangaroos,
            steps_per_call,
            jump_table_size,
            _padding: 0,
        };
        if verbose {
            info!("Config created: steps_per_call={}", steps_per_call);
        }

        // Create buffers
        if verbose {
            info!("Creating GPU buffers...");
        }
        let max_dps = MAX_DISTINGUISHED_POINTS;
        let buffers = GpuBuffers::new(
            &ctx,
            &pipeline,
            &config,
            &jump_points,
            &jump_distances,
            num_kangaroos,
            max_dps,
        )?;

        // Initialize kangaroos
        let kangaroos = initialize_kangaroos(&pubkey, &start, range_bits, num_kangaroos)?;
        upload_kangaroos(&ctx, &buffers, &kangaroos)?;

        // Create solver instance
        let mut solver = Self {
            ctx,
            pipeline,
            buffers,
            dp_table: DPTable::new(start),
            total_ops: 0,
            num_kangaroos,
            steps_per_call,
        };

        // Auto-calibrate steps_per_call
        solver.calibrate(dp_bits, verbose);

        // Update config buffer with calibrated value and correct DP mask
        let final_config = GpuConfig {
            dp_mask_lo: [dp_mask[0], dp_mask[1], dp_mask[2], dp_mask[3]],
            dp_mask_hi: [dp_mask[4], dp_mask[5], dp_mask[6], dp_mask[7]],
            num_kangaroos,
            steps_per_call: solver.steps_per_call,
            jump_table_size: 256,
            _padding: 0,
        };
        solver.ctx.queue.write_buffer(
            &solver.buffers.config_buffer,
            0,
            bytemuck::bytes_of(&final_config),
        );

        // Reset DP count after calibration warmup
        solver.reset_dp_count()?;

        Ok(solver)
    }

    /// Run one batch of GPU operations
    pub fn step(&mut self) -> Result<Option<Vec<u8>>> {
        // Create command encoder
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Kangaroo Encoder"),
            });

        // Dispatch compute
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Kangaroo Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.buffers.bind_group, &[]);

            let workgroups = self.num_kangaroos.div_ceil(64); // Workgroup size is 64
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy DP count for readback (first 4 bytes of staging)
        encoder.copy_buffer_to_buffer(
            &self.buffers.dp_count_buffer,
            0,
            &self.buffers.staging_buffer,
            0,
            4,
        );

        // Submit
        self.ctx.queue.submit(Some(encoder.finish()));

        // Update operation count
        self.total_ops += (self.num_kangaroos as u64) * (self.steps_per_call as u64);

        // Log progress every 10M ops (less verbose)
        if self.total_ops % 10_000_000 < (self.num_kangaroos as u64 * self.steps_per_call as u64) {
            let (tame, wild) = self.dp_table.count_by_type();
            tracing::info!(
                "Ops: {}M | DPs: {} ({} tame, {} wild)",
                self.total_ops / 1_000_000,
                self.dp_table.total_dps(),
                tame,
                wild
            );
        }

        // Read back DP count and process if any found
        // TODO: Optimization: Use double buffering for async readback.
        // Currently we block here waiting for GPU to finish execution and transfer data.
        // With double buffering, we could dispatch the next batch while waiting for
        // the previous one, keeping GPU fully occupied.
        // Needs:
        // 1. Two sets of buffers (or at least staging buffers)
        // 2. State machine to manage "Dispatch A -> Read B -> Dispatch B -> Read A"
        let dp_count = self.read_dp_count()?;
        if dp_count > 0 {
            // Copy DP buffer for readback
            let mut encoder2 =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("DP Readback"),
                    });

            let dp_size = std::mem::size_of::<GpuDistinguishedPoint>();
            let max_dps = MAX_DISTINGUISHED_POINTS as usize;
            let actual_count = (dp_count as usize).min(max_dps);
            let copy_size = (actual_count * dp_size) as u64;

            encoder2.copy_buffer_to_buffer(
                &self.buffers.dp_buffer,
                0,
                &self.buffers.staging_buffer,
                4,
                copy_size,
            );

            self.ctx.queue.submit(Some(encoder2.finish()));

            // Read back DPs and check for collision
            let dps = self.read_dps(actual_count as u32)?;
            for dp in dps {
                if let Some(key) = self.dp_table.insert_and_check(dp) {
                    return Ok(Some(key));
                }
            }

            self.reset_dp_count()?;
        }

        Ok(None)
    }

    /// Get total operations performed
    pub fn total_operations(&self) -> u64 {
        self.total_ops
    }

    fn read_dp_count(&self) -> Result<u32> {
        let slice = self.buffers.staging_buffer.slice(0..4);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv()??;

        let data = slice.get_mapped_range();
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        self.buffers.staging_buffer.unmap();

        Ok(count)
    }

    fn read_dps(&self, count: u32) -> Result<Vec<GpuDistinguishedPoint>> {
        let dp_size = std::mem::size_of::<GpuDistinguishedPoint>();
        let total_size = 4 + (count as usize * dp_size);

        let slice = self.buffers.staging_buffer.slice(0..total_size as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv()??;

        let data = slice.get_mapped_range();

        // Skip first 4 bytes (count), read DPs
        let dp_bytes = &data[4..];
        let dps: Vec<GpuDistinguishedPoint> = dp_bytes
            .chunks_exact(dp_size)
            .take(count as usize)
            .map(|chunk| *bytemuck::from_bytes::<GpuDistinguishedPoint>(chunk))
            .collect();

        drop(data);
        self.buffers.staging_buffer.unmap();

        Ok(dps)
    }

    fn reset_dp_count(&self) -> Result<()> {
        self.ctx
            .queue
            .write_buffer(&self.buffers.dp_count_buffer, 0, &[0u8; 4]);
        Ok(())
    }

    /// Calibrate steps_per_call by measuring actual GPU dispatch times
    fn calibrate(&mut self, dp_bits: u32, verbose: bool) {
        let candidates = [16u32, 32, 64, 128, 256, 512];
        let mut best_steps = candidates[0];

        if verbose {
            info!("Calibrating GPU performance...");
        }

        for &steps in &candidates {
            // Check DP buffer constraint first
            let max_steps = Self::select_steps_per_call(
                steps,
                self.num_kangaroos,
                dp_bits,
                MAX_DISTINGUISHED_POINTS,
            );
            if max_steps < steps {
                // Would overflow DP buffer, stop here
                break;
            }

            // Update config buffer with new steps_per_call
            let config = GpuConfig {
                dp_mask_lo: [0; 4], // Not used in timing test
                dp_mask_hi: [0; 4],
                num_kangaroos: self.num_kangaroos,
                steps_per_call: steps,
                jump_table_size: 256,
                _padding: 0,
            };
            self.ctx.queue.write_buffer(
                &self.buffers.config_buffer,
                0,
                bytemuck::bytes_of(&config),
            );

            // Warm up dispatch
            self.dispatch_once();

            // Timed dispatch
            let start = Instant::now();
            self.dispatch_once();
            let elapsed_ms = start.elapsed().as_millis();

            if verbose {
                info!("  steps_per_call={}: {}ms", steps, elapsed_ms);
            }

            if elapsed_ms <= TARGET_DISPATCH_MS {
                best_steps = steps;
            } else {
                // Too slow, stop searching
                break;
            }
        }

        // Apply the best value
        self.steps_per_call = best_steps;

        if verbose {
            info!("Calibrated: steps_per_call={}", best_steps);
        }
    }

    /// Single GPU dispatch without readback (for calibration)
    fn dispatch_once(&self) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Calibration Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Calibration Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.buffers.bind_group, &[]);
            let workgroups = self.num_kangaroos.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }
}

fn upload_kangaroos(
    ctx: &GpuContext,
    buffers: &GpuBuffers,
    kangaroos: &[GpuKangaroo],
) -> Result<()> {
    ctx.queue.write_buffer(
        &buffers.kangaroos_buffer,
        0,
        bytemuck::cast_slice(kangaroos),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{KangarooSolver, MAX_DISTINGUISHED_POINTS};

    #[test]
    fn caps_steps_when_dp_buffer_would_overflow() {
        // With dense DPs (8 bits) and many kangaroos, a large steps_per_call would overflow the DP buffer.
        let steps =
            KangarooSolver::select_steps_per_call(4_096, 16_384, 8, MAX_DISTINGUISHED_POINTS);
        assert_eq!(steps, 921);
    }

    #[test]
    fn keeps_optimal_when_within_budget() {
        // Higher DP bits reduce DP density; we should keep the GPU-optimal step count.
        let steps =
            KangarooSolver::select_steps_per_call(4_096, 4_096, 16, MAX_DISTINGUISHED_POINTS);
        assert_eq!(steps, 4_096);
    }
}
