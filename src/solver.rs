//! GPU-accelerated Kangaroo solver
//!
//! Coordinates GPU compute with CPU collision detection.

use crate::cpu::init::{generate_jump_tables, initialize_kangaroos};
use crate::cpu::DPTable;
use crate::crypto::{Point, U256};
use crate::gpu::{
    GpuBuffers, GpuConfig, GpuContext, GpuDistinguishedPoint, GpuKangaroo, JumpTableData,
    KangarooPipeline, WorkgroupVariant,
};
use anyhow::{anyhow, ensure, Result};
use k256::ProjectivePoint;
use std::time::{Duration, Instant};
use tracing::info;

/// Maximum time to wait for a single GPU poll before treating the device as unresponsive.
/// Generous enough for slow GPUs with high steps_per_call, short enough to unblock
/// multi-GPU shutdown and avoid indefinite hangs on wedged drivers.
const GPU_POLL_TIMEOUT: Duration = Duration::from_secs(5);

const MAX_DISTINGUISHED_POINTS: u32 = 65_536;
const JUMP_TABLE_SIZE: u32 = 256;
/// Target dispatch time in milliseconds for calibration.
///
/// This is still comfortably below the multi-second GPU watchdog budgets on the
/// supported backends, but high enough to avoid overpaying host round-trip cost
/// on benchmark-sized solves.
const TARGET_DISPATCH_MS: u128 = 120;

struct JumpTableRefs<'a> {
    jump_points: &'a [crate::gpu::GpuAffinePoint],
    jump_distances: &'a [[u32; 8]],
}

/// Main Kangaroo solver
pub struct KangarooSolver {
    ctx: GpuContext,
    pipeline: KangarooPipeline,
    buffers: GpuBuffers,
    dp_table: Option<DPTable>,
    total_ops: u64,
    num_kangaroos: u32,
    steps_per_call: u32,
    workgroup_size: u32,
    current_slot: usize,
    prev_submission: Option<wgpu::SubmissionIndex>,
}

impl KangarooSolver {
    fn dp_meta(dp_bits: u32) -> [u32; 4] {
        let full_limbs = (dp_bits / 32).min(8);
        let rem = dp_bits % 32;
        let partial_mask = if rem == 0 { 0 } else { (1u32 << rem) - 1 };
        [full_limbs, partial_mask, 0, 0]
    }

    fn cycle_cap_for(dp_bits: u32) -> u32 {
        let exp = (dp_bits / 2).min(31);
        512u32.max(2u32.pow(exp))
    }

    pub fn new(
        ctx: GpuContext,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
    ) -> Result<Self> {
        Self::new_internal(
            ctx,
            pubkey,
            start,
            range_bits,
            dp_bits,
            num_kangaroos,
            num_kangaroos,
            true,
            true,
            ProjectivePoint::GENERATOR,
            0,
        )
    }

    /// Create a solver with a custom base point.
    ///
    /// Used for modular constraint search where H = M*G replaces G.
    /// The pubkey and start should already be the transformed values (Q, j_start).
    pub fn new_with_base(
        ctx: GpuContext,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
        base_point: ProjectivePoint,
    ) -> Result<Self> {
        Self::new_internal(
            ctx,
            pubkey,
            start,
            range_bits,
            dp_bits,
            num_kangaroos,
            num_kangaroos,
            true,
            true,
            base_point,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_base_no_dp_table(
        ctx: GpuContext,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
        global_kangaroo_count: u32,
        base_point: ProjectivePoint,
        kangaroo_offset: u32,
    ) -> Result<Self> {
        Self::new_internal(
            ctx,
            pubkey,
            start,
            range_bits,
            dp_bits,
            num_kangaroos,
            global_kangaroo_count,
            false,
            false,
            base_point,
            kangaroo_offset,
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

    fn benchmark_variant(
        ctx: &GpuContext,
        variant: WorkgroupVariant,
        config: &GpuConfig,
        table_refs: &JumpTableRefs<'_>,
        kangaroos: &[GpuKangaroo],
        num_kangaroos: u32,
    ) -> Result<(KangarooPipeline, u128)> {
        let pipeline = KangarooPipeline::new(ctx, variant)?;
        let buffers = GpuBuffers::new(
            ctx,
            &pipeline,
            config,
            JumpTableData {
                jump_points: table_refs.jump_points,
                jump_distances: table_refs.jump_distances,
            },
            num_kangaroos,
            MAX_DISTINGUISHED_POINTS,
        )?;
        upload_kangaroos(ctx, &buffers, kangaroos)?;

        // warmup
        Self::dispatch_once_raw(ctx, &pipeline, &buffers, num_kangaroos, variant.size())?;

        ctx.queue
            .write_buffer(buffers.dp_count_buffer(0), 0, &[0u8; 4]);

        let start = Instant::now();
        Self::dispatch_once_raw(ctx, &pipeline, &buffers, num_kangaroos, variant.size())?;
        let elapsed = start.elapsed().as_millis();

        Ok((pipeline, elapsed))
    }

    fn select_best_variant(
        ctx: &GpuContext,
        config: &GpuConfig,
        table_refs: &JumpTableRefs<'_>,
        kangaroos: &[GpuKangaroo],
        num_kangaroos: u32,
        verbose: bool,
    ) -> Result<KangarooPipeline> {
        let variant = if ctx.max_workgroup_size() >= 128 && num_kangaroos >= 65_536 {
            WorkgroupVariant::Wg128
        } else {
            WorkgroupVariant::Wg64
        };

        if variant == WorkgroupVariant::Wg64 {
            if verbose {
                info!("Skipping kernel probe, using workgroup=64");
            }
            return KangarooPipeline::new(ctx, variant);
        }

        let (pipeline, elapsed) =
            Self::benchmark_variant(ctx, variant, config, table_refs, kangaroos, num_kangaroos)?;

        if verbose {
            info!(
                "Kernel warmup: workgroup={} dispatch={}ms",
                variant.size(),
                elapsed
            );
        }

        Ok(pipeline)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_internal(
        ctx: GpuContext,
        pubkey: Point,
        start: U256,
        range_bits: u32,
        dp_bits: u32,
        num_kangaroos: u32,
        global_kangaroo_count: u32,
        verbose: bool,
        with_dp_table: bool,
        base_point: ProjectivePoint,
        kangaroo_offset: u32,
    ) -> Result<Self> {
        if verbose {
            info!("Generating jump table...");
        }
        let jump_table_size = JUMP_TABLE_SIZE;
        let (jump_points, jump_distances) = generate_jump_tables(range_bits, &base_point);
        ensure!(
            jump_points.len() == JUMP_TABLE_SIZE as usize
                && jump_distances.len() == JUMP_TABLE_SIZE as usize,
            "jump tables must have {} entries",
            JUMP_TABLE_SIZE
        );
        if verbose {
            info!("Jump table generated: {} entries", jump_table_size);
            for (i, dist) in jump_distances.iter().enumerate().take(4) {
                info!("Jump dist[{}] = 0x{:08x}", i, dist[0]);
            }
        }

        let dp_meta = Self::dp_meta(dp_bits);
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

        let cycle_cap = Self::cycle_cap_for(dp_bits);
        let config = GpuConfig {
            dp_meta,
            num_kangaroos,
            steps_per_call,
            jump_table_size,
            cycle_cap,
        };
        if verbose {
            info!("Config created: steps_per_call={}", steps_per_call);
        }

        let kangaroos = initialize_kangaroos(
            &pubkey,
            &start,
            range_bits,
            num_kangaroos,
            &base_point,
            kangaroo_offset,
            global_kangaroo_count,
        )?;

        if verbose {
            info!("Probing kernel variants (64/128)...");
        }
        let table_refs = JumpTableRefs {
            jump_points: &jump_points,
            jump_distances: &jump_distances,
        };
        let pipeline = Self::select_best_variant(
            &ctx,
            &config,
            &table_refs,
            &kangaroos,
            num_kangaroos,
            verbose,
        )?;
        if verbose {
            info!("Selected kernel workgroup={}", pipeline.variant.size());
        }
        let workgroup_size = pipeline.variant.size();

        // Create buffers
        if verbose {
            info!("Creating GPU buffers...");
        }
        let max_dps = MAX_DISTINGUISHED_POINTS;
        let buffers = GpuBuffers::new(
            &ctx,
            &pipeline,
            &config,
            JumpTableData {
                jump_points: &jump_points,
                jump_distances: &jump_distances,
            },
            num_kangaroos,
            max_dps,
        )?;

        upload_kangaroos(&ctx, &buffers, &kangaroos)?;

        // Create solver instance
        let mut solver = Self {
            ctx,
            pipeline,
            buffers,
            dp_table: if with_dp_table {
                Some(DPTable::new(start, pubkey, base_point))
            } else {
                None
            },
            total_ops: 0,
            num_kangaroos,
            steps_per_call,
            workgroup_size,
            current_slot: 0,
            prev_submission: None,
        };

        // Auto-calibrate steps_per_call
        solver.calibrate(dp_bits, verbose)?;

        // Update config buffer with calibrated value and correct DP mask
        let cycle_cap = Self::cycle_cap_for(dp_bits);
        let final_config = GpuConfig {
            dp_meta,
            num_kangaroos,
            steps_per_call: solver.steps_per_call,
            jump_table_size: JUMP_TABLE_SIZE,
            cycle_cap,
        };
        solver.ctx.queue.write_buffer(
            &solver.buffers.config_buffer,
            0,
            bytemuck::bytes_of(&final_config),
        );

        solver.reset_dp_count(0)?;
        solver.reset_dp_count(1)?;

        Ok(solver)
    }

    /// Run one batch of GPU operations.
    ///
    /// Pipelines dispatch and readback across double-buffered slots:
    /// dispatches new work on the current slot while reading back results
    /// from the previous slot. The first call returns no DPs (nothing
    /// pending yet); steady-state calls overlap GPU compute with CPU
    /// readback.
    pub fn step_collect(&mut self) -> Result<(Vec<GpuDistinguishedPoint>, u64)> {
        let write_slot = self.current_slot;
        let read_slot = 1 - write_slot;

        // Queue new compute work on write_slot.
        // Safe despite shared kangaroos_buffer: wgpu executes submissions
        // in order within a single queue, so the previous dispatch finishes
        // writing kangaroo positions before this dispatch reads them.
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Kangaroo Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Kangaroo Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, self.buffers.bind_group(write_slot), &[]);
            let workgroups = self.num_kangaroos.div_ceil(self.workgroup_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            self.buffers.dp_count_buffer(write_slot),
            0,
            self.buffers.staging_buffer(write_slot),
            0,
            4,
        );

        let new_sub = self.ctx.queue.submit(Some(encoder.finish()));

        let ops_delta = (self.num_kangaroos as u64) * (self.steps_per_call as u64);
        self.total_ops += ops_delta;

        // Read back from the previous slot while the GPU works on write_slot
        let dps = if let Some(prev_sub) = self.prev_submission.take() {
            self.read_pending(read_slot, prev_sub)?
        } else {
            Vec::new()
        };

        self.prev_submission = Some(new_sub);
        self.current_slot = 1 - write_slot;
        Ok((dps, ops_delta))
    }

    /// Read back DPs from a completed slot and reset its counter.
    fn read_pending(
        &self,
        slot: usize,
        submission: wgpu::SubmissionIndex,
    ) -> Result<Vec<GpuDistinguishedPoint>> {
        let count = self.read_slot_dp_count(slot, submission)?;
        let dps = if count == 0 {
            Vec::new()
        } else {
            let mut copy_encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Kangaroo DP Copy Encoder"),
                    });
            let dp_size = std::mem::size_of::<GpuDistinguishedPoint>() as u64;
            let clamped = count.min(MAX_DISTINGUISHED_POINTS);
            copy_encoder.copy_buffer_to_buffer(
                self.buffers.dp_buffer(slot),
                0,
                self.buffers.staging_buffer(slot),
                0,
                (clamped as u64) * dp_size,
            );
            let copy_sub = self.ctx.queue.submit(Some(copy_encoder.finish()));
            self.read_slot_dps(slot, clamped, copy_sub)?
        };
        self.reset_dp_count(slot)?;
        Ok(dps)
    }

    /// Drain the pipeline: read back DPs from the last dispatched batch.
    ///
    /// Call after the solve loop exits to collect any remaining results.
    pub fn flush_pending(&mut self) -> Result<Vec<GpuDistinguishedPoint>> {
        let Some(prev_sub) = self.prev_submission.take() else {
            return Ok(Vec::new());
        };
        let pending_slot = 1 - self.current_slot;
        self.read_pending(pending_slot, prev_sub)
    }

    /// Run one batch of GPU operations.
    pub fn step(&mut self) -> Result<Option<Vec<u8>>> {
        let (dps, ops_delta) = self.step_collect()?;

        if self.total_ops % 10_000_000 < ops_delta {
            if let Some(dp_table) = self.dp_table.as_ref() {
                let (tame, w1, w2) = dp_table.count_by_type();
                tracing::info!(
                    "Ops: {}M | DPs: {} ({} tame, {} wild1, {} wild2)",
                    self.total_ops / 1_000_000,
                    dp_table.total_dps(),
                    tame,
                    w1,
                    w2
                );
            }
        }

        if let Some(dp_table) = self.dp_table.as_mut() {
            for dp in dps {
                if let Some(key) = dp_table.insert_and_check(dp) {
                    return Ok(Some(key));
                }
            }
        }

        Ok(None)
    }

    /// Get total operations performed
    pub fn total_operations(&self) -> u64 {
        self.total_ops
    }

    fn read_slot_dp_count(&self, slot: usize, submission: wgpu::SubmissionIndex) -> Result<u32> {
        let staging = self.buffers.staging_buffer(slot);
        let slice = staging.slice(0..4);
        let result = (|| -> Result<u32> {
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            self.ctx
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(GPU_POLL_TIMEOUT),
                })
                .map_err(|e| anyhow!("GPU poll timed out or failed reading DP count: {e:?}"))?;

            let map_result = rx
                .recv_timeout(GPU_POLL_TIMEOUT)
                .map_err(|e| anyhow!("DP count map callback not received within timeout: {e}"))?;
            map_result.map_err(|e| anyhow!("Failed to map DP count buffer: {e:?}"))?;

            let data = slice.get_mapped_range();
            let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            drop(data);
            Ok(count)
        })();
        staging.unmap();
        result
    }

    fn read_slot_dps(
        &self,
        slot: usize,
        count: u32,
        submission: wgpu::SubmissionIndex,
    ) -> Result<Vec<GpuDistinguishedPoint>> {
        let dp_size = std::mem::size_of::<GpuDistinguishedPoint>();
        let actual_count = (count as usize).min(MAX_DISTINGUISHED_POINTS as usize);
        let total_size = actual_count * dp_size;

        let staging = self.buffers.staging_buffer(slot);
        let slice = staging.slice(0..total_size as u64);
        let result = (|| -> Result<Vec<GpuDistinguishedPoint>> {
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            self.ctx
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(GPU_POLL_TIMEOUT),
                })
                .map_err(|e| anyhow!("GPU poll timed out or failed reading DP payload: {e:?}"))?;

            let map_result = rx
                .recv_timeout(GPU_POLL_TIMEOUT)
                .map_err(|e| anyhow!("DP payload map callback not received within timeout: {e}"))?;
            map_result.map_err(|e| anyhow!("Failed to map DP payload buffer: {e:?}"))?;

            let data = slice.get_mapped_range();
            let dps: Vec<GpuDistinguishedPoint> = data
                .chunks_exact(dp_size)
                .take(actual_count)
                .map(|chunk| *bytemuck::from_bytes::<GpuDistinguishedPoint>(chunk))
                .collect();

            drop(data);
            Ok(dps)
        })();
        staging.unmap();
        result
    }

    fn reset_dp_count(&self, slot: usize) -> Result<()> {
        self.ctx
            .queue
            .write_buffer(self.buffers.dp_count_buffer(slot), 0, &[0u8; 4]);
        Ok(())
    }

    /// Calibrate steps_per_call by measuring actual GPU dispatch times
    fn calibrate(&mut self, dp_bits: u32, verbose: bool) -> Result<()> {
        // Benchmark-sized solves hit a sharp cliff between 24 and 32 steps on the
        // tested GPUs, so probe a few low-end values before jumping back to the
        // usual power-of-two sweep. Keep the list short - calibration dispatches
        // still burn startup work even though their DP output gets dropped.
        let candidates = [16u32, 17, 18, 24, 64, 128, 256, 512];
        let mut best_steps = candidates[0];
        let dp_meta = Self::dp_meta(dp_bits);

        if self.workgroup_size == 64 && self.num_kangaroos <= 65_536 && self.steps_per_call == 16 {
            if verbose {
                info!("Skipping calibration for small-herd Wg64 path; using steps_per_call=16");
            }
            return Ok(());
        }

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
                dp_meta,
                num_kangaroos: self.num_kangaroos,
                steps_per_call: steps,
                jump_table_size: JUMP_TABLE_SIZE,
                cycle_cap: 512, // Default floor for calibration
            };
            self.ctx.queue.write_buffer(
                &self.buffers.config_buffer,
                0,
                bytemuck::bytes_of(&config),
            );

            let calibration_slot = 0;

            // Warm up dispatch
            self.reset_dp_count(calibration_slot)?;
            if self.dispatch_once().is_err() {
                break; // GPU too slow for this candidate, keep last good value
            }

            // Timed dispatch
            self.reset_dp_count(calibration_slot)?;
            let start = Instant::now();
            if self.dispatch_once().is_err() {
                break; // GPU too slow for this candidate, keep last good value
            }
            let elapsed_ms = start.elapsed().as_millis();

            if verbose {
                info!("  steps_per_call={}: {}ms", steps, elapsed_ms);
            }

            if elapsed_ms <= TARGET_DISPATCH_MS {
                best_steps = steps;

                // Candidates double each time. If we're already above half the budget,
                // the next probe is overwhelmingly likely to miss and just burn startup time.
                if elapsed_ms.saturating_mul(2) > TARGET_DISPATCH_MS {
                    break;
                }
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

        Ok(())
    }

    /// Single GPU dispatch without readback (for calibration)
    fn dispatch_once(&self) -> Result<()> {
        Self::dispatch_once_raw(
            &self.ctx,
            &self.pipeline,
            &self.buffers,
            self.num_kangaroos,
            self.workgroup_size,
        )
    }

    fn dispatch_once_raw(
        ctx: &GpuContext,
        pipeline: &KangarooPipeline,
        buffers: &GpuBuffers,
        num_kangaroos: u32,
        workgroup_size: u32,
    ) -> Result<()> {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Calibration Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Calibration Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, buffers.bind_group(0), &[]);
            let workgroups = num_kangaroos.div_ceil(workgroup_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(GPU_POLL_TIMEOUT),
            })
            .map_err(|e| anyhow!("GPU poll timed out or failed during dispatch: {e:?}"))?;

        Ok(())
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
