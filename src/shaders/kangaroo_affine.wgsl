// =============================================================================
// Pollard's Kangaroo Algorithm - GPU Kernel (Affine Coordinates)
// =============================================================================
// Uses affine coordinates with batch inversion for point addition
// More efficient than Jacobian: fewer field operations, no Z coordinate

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

struct Config {
    dp_mask_lo: vec4<u32>,
    dp_mask_hi: vec4<u32>,
    num_kangaroos: u32,
    steps_per_call: u32,
    jump_table_size: u32,
    _padding: u32
}

// Must match Rust GpuKangaroo struct layout!
struct Kangaroo {
    x: array<u32, 8>,
    y: array<u32, 8>,
    dist: array<u32, 8>,
    ktype: u32,
    is_active: u32,
    _padding: array<u32, 6>
}

struct DistinguishedPoint {
    x: array<u32, 8>,
    dist: array<u32, 8>,
    ktype: u32,
    kangaroo_id: u32,
    _padding: array<u32, 6>
}

// -----------------------------------------------------------------------------
// Buffers
// -----------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> jump_points: array<AffinePoint, 256>;
@group(0) @binding(2) var<storage, read> jump_distances: array<array<u32, 8>, 256>;
@group(0) @binding(3) var<storage, read_write> kangaroos: array<Kangaroo>;
@group(0) @binding(4) var<storage, read_write> dp_buffer: array<DistinguishedPoint>;
@group(0) @binding(5) var<storage, read_write> dp_count: atomic<u32>;

// Shared memory for batch inversion (tree-based Montgomery's trick)
// Product tree + saved right-child products for inverse propagation
var<workgroup> shared_prod: array<array<u32, 8>, 64>;    // Product tree / individual inverses
var<workgroup> shared_save: array<array<u32, 8>, 64>;    // Saved right-child products (63 entries used)

// -----------------------------------------------------------------------------
// Store distinguished point
// -----------------------------------------------------------------------------

fn store_dp(k: Kangaroo, kangaroo_id: u32) {
    let idx = atomicAdd(&dp_count, 1u);

    if (idx < 65536u) {
        var dp: DistinguishedPoint;
        dp.x = k.x;
        dp.dist = k.dist;
        dp.ktype = k.ktype;
        dp.kangaroo_id = kangaroo_id;
        dp._padding = array<u32, 6>(0u, 0u, 0u, 0u, 0u, 0u);
        dp_buffer[idx] = dp;
    }
}

// -----------------------------------------------------------------------------
// Affine point addition: R = P + Q (both affine)
// Returns (x3, y3) given (x1, y1), (x2, y2), and precomputed inv = 1/(x2-x1)
// 
// Formula:
//   λ = (y2 - y1) * inv
//   x3 = λ² - x1 - x2
//   y3 = λ * (x1 - x3) - y1
//
// Cost: 2M + 1S (with precomputed inverse)
// Compare to Jacobian mixed add: 8M + 4S
// -----------------------------------------------------------------------------

fn affine_add_with_inv(
    x1: array<u32, 8>,
    y1: array<u32, 8>,
    x2: array<u32, 8>,
    y2: array<u32, 8>,
    dx_inv: array<u32, 8>
) -> AffinePoint {
    // λ = (y2 - y1) / (x2 - x1) = (y2 - y1) * dx_inv
    let dy = fe_sub(y2, y1);
    let lambda = fe_mul(dy, dx_inv);
    
    // x3 = λ² - x1 - x2
    let lambda_sq = fe_square(lambda);
    let x3 = fe_sub(fe_sub(lambda_sq, x1), x2);
    
    // y3 = λ * (x1 - x3) - y1
    let x1_minus_x3 = fe_sub(x1, x3);
    let y3 = fe_sub(fe_mul(lambda, x1_minus_x3), y1);
    
    var result: AffinePoint;
    result.x = x3;
    result.y = y3;
    return result;
}

fn is_distinguished(px: array<u32, 8>) -> bool {
    return ((px[0] & config.dp_mask_lo.x) == 0u)
        && ((px[1] & config.dp_mask_lo.y) == 0u)
        && ((px[2] & config.dp_mask_lo.z) == 0u)
        && ((px[3] & config.dp_mask_lo.w) == 0u)
        && ((px[4] & config.dp_mask_hi.x) == 0u)
        && ((px[5] & config.dp_mask_hi.y) == 0u)
        && ((px[6] & config.dp_mask_hi.z) == 0u)
        && ((px[7] & config.dp_mask_hi.w) == 0u);
}

// -----------------------------------------------------------------------------
// Main compute shader
// -----------------------------------------------------------------------------

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id_vec: vec3<u32>) {
    let kid = global_id.x;
    let lid = local_id_vec.x;

    // Load kangaroo state (if valid)
    var k: Kangaroo;
    var valid = false;
    if (kid < config.num_kangaroos) {
        k = kangaroos[kid];
        if (k.is_active != 0u) {
            valid = true;
        }
    }

    // Current point in affine coordinates
    var px: array<u32, 8>;
    var py: array<u32, 8>;
    
    if (valid) {
        px = k.x;
        py = k.y;
    } else {
        // Dummy point for inactive threads
        px = fe_one();
        py = fe_one();
    }

    // Track if we already stored a DP this batch
    var dp_stored = false;

    // Perform jumps
    for (var step = 0u; step < config.steps_per_call; step++) {
        // Select jump based on x coordinate
        let jump_idx = px[0] & 0xFFu;
        let jump_point = jump_points[jump_idx];
        let jump_dist = jump_distances[jump_idx];
        
        // =====================================================================
        // BATCH INVERSION (Montgomery's trick for dx = x_jump - x_point)
        // =====================================================================
        
        // 1. Compute dx = x_jump - x_point and store in shared memory
        //    If dx=0 (point equals jump point), use 1 to avoid poisoning the batch,
        //    but track it to skip the affine add later (astronomically unlikely: 1/2^256).
        var dx = fe_sub(jump_point.x, px);
        var dx_was_zero = fe_is_zero(dx);
        if (dx_was_zero) {
            dx = fe_one();
        }
        shared_prod[lid] = dx;
        workgroupBarrier();

        // 2. Tree-based batch inversion (Montgomery's trick with parallel tree)
        //    Up-sweep builds product tree while saving right children.
        //    Single fe_inv of root, then down-sweep propagates individual inverses.
        //    Eliminates suffix scan: ~14 barriers vs ~30, ~18 fe_mul rounds vs ~26.

        // ===== UP-SWEEP: build product tree, save right children =====
        // Layout of shared_save: [0..31]=level0, [32..47]=level1, [48..55]=level2,
        //                        [56..59]=level3, [60..61]=level4, [62]=level5

        // Level 0 (stride 1): 32 threads merge pairs
        if ((lid & 1u) == 1u) {
            shared_save[lid >> 1u] = shared_prod[lid];
            shared_prod[lid] = fe_mul(shared_prod[lid - 1u], shared_prod[lid]);
        }
        workgroupBarrier();

        // Level 1 (stride 2): 16 threads merge quads
        if ((lid & 3u) == 3u) {
            shared_save[32u + (lid >> 2u)] = shared_prod[lid];
            shared_prod[lid] = fe_mul(shared_prod[lid - 2u], shared_prod[lid]);
        }
        workgroupBarrier();

        // Level 2 (stride 4): 8 threads merge octets
        if ((lid & 7u) == 7u) {
            shared_save[48u + (lid >> 3u)] = shared_prod[lid];
            shared_prod[lid] = fe_mul(shared_prod[lid - 4u], shared_prod[lid]);
        }
        workgroupBarrier();

        // Level 3 (stride 8): 4 threads merge 16-element groups
        if ((lid & 15u) == 15u) {
            shared_save[56u + (lid >> 4u)] = shared_prod[lid];
            shared_prod[lid] = fe_mul(shared_prod[lid - 8u], shared_prod[lid]);
        }
        workgroupBarrier();

        // Level 4 (stride 16): 2 threads merge 32-element halves
        if ((lid & 31u) == 31u) {
            shared_save[60u + (lid >> 5u)] = shared_prod[lid];
            shared_prod[lid] = fe_mul(shared_prod[lid - 16u], shared_prod[lid]);
        }
        workgroupBarrier();

        // Level 5 (stride 32): 1 thread merges full 64-element product
        if (lid == 63u) {
            shared_save[62u] = shared_prod[63u];
            shared_prod[63u] = fe_mul(shared_prod[31u], shared_prod[63u]);
        }
        workgroupBarrier();

        // ===== INVERT root (total product of all dx values) =====
        if (lid == 0u) {
            shared_prod[63u] = fe_inv(shared_prod[63u]);
        }
        workgroupBarrier();

        // ===== DOWN-SWEEP: propagate inverses through tree =====
        // At each node: inv(left) = inv(parent) * right_saved
        //               inv(right) = inv(parent) * left_preserved

        // Level 5 (stride 32): 1 thread splits root inverse
        if (lid == 63u) {
            let inv_p = shared_prod[63u];
            let left = shared_prod[31u];
            let right = shared_save[62u];
            shared_prod[31u] = fe_mul(inv_p, right);
            shared_prod[63u] = fe_mul(inv_p, left);
        }
        workgroupBarrier();

        // Level 4 (stride 16): 2 threads
        if ((lid & 31u) == 31u) {
            let inv_p = shared_prod[lid];
            let left = shared_prod[lid - 16u];
            let right = shared_save[60u + (lid >> 5u)];
            shared_prod[lid - 16u] = fe_mul(inv_p, right);
            shared_prod[lid] = fe_mul(inv_p, left);
        }
        workgroupBarrier();

        // Level 3 (stride 8): 4 threads
        if ((lid & 15u) == 15u) {
            let inv_p = shared_prod[lid];
            let left = shared_prod[lid - 8u];
            let right = shared_save[56u + (lid >> 4u)];
            shared_prod[lid - 8u] = fe_mul(inv_p, right);
            shared_prod[lid] = fe_mul(inv_p, left);
        }
        workgroupBarrier();

        // Level 2 (stride 4): 8 threads
        if ((lid & 7u) == 7u) {
            let inv_p = shared_prod[lid];
            let left = shared_prod[lid - 4u];
            let right = shared_save[48u + (lid >> 3u)];
            shared_prod[lid - 4u] = fe_mul(inv_p, right);
            shared_prod[lid] = fe_mul(inv_p, left);
        }
        workgroupBarrier();

        // Level 1 (stride 2): 16 threads
        if ((lid & 3u) == 3u) {
            let inv_p = shared_prod[lid];
            let left = shared_prod[lid - 2u];
            let right = shared_save[32u + (lid >> 2u)];
            shared_prod[lid - 2u] = fe_mul(inv_p, right);
            shared_prod[lid] = fe_mul(inv_p, left);
        }
        workgroupBarrier();

        // Level 0 (stride 1): 32 threads produce individual inverses
        if ((lid & 1u) == 1u) {
            let inv_p = shared_prod[lid];
            let left = shared_prod[lid - 1u];
            let right = shared_save[lid >> 1u];
            shared_prod[lid - 1u] = fe_mul(inv_p, right);
            shared_prod[lid] = fe_mul(inv_p, left);
        }
        workgroupBarrier();

        // Now: shared_prod[lid] = 1/dx[lid] for all threads
        let dx_inv = shared_prod[lid];

        // =====================================================================
        // POINT ADDITION AND DP CHECK
        // =====================================================================

        if (valid) {
            // Check for DP before the jump (on current position)
            if (!dp_stored) {
                if (is_distinguished(px)) {
                    k.x = px;
                    k.y = py;
                    store_dp(k, kid);
                    dp_stored = true;
                }
            }

            // Perform affine addition: P = P + jump_point
            // Skip if dx was zero (point collision - astronomically unlikely)
            if (!dx_was_zero) {
                let result = affine_add_with_inv(px, py, jump_point.x, jump_point.y, dx_inv);
                px = result.x;
                py = result.y;

                // Update distance
                k.dist = scalar_add_256(k.dist, jump_dist);
            }
        }
    }

    // Write back updated state
    if (valid) {
        k.x = px;
        k.y = py;
        kangaroos[kid] = k;
    }
}
