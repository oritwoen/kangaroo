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

// Shared memory for batch inversion (Montgomery's trick)
// We batch invert (x_jump - x_point) for all threads
var<workgroup> shared_dx: array<array<u32, 8>, 64>;      // Delta X values
var<workgroup> shared_prod: array<array<u32, 8>, 64>;    // Prefix products

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
        //    If dx=0 (point equals jump point), use 1 to avoid division by zero.
        //    This is astronomically unlikely (1/2^256) and would just skip one jump.
        var dx = fe_sub(jump_point.x, px);
        if (fe_is_zero(dx)) {
            dx = fe_one();
        }
        shared_dx[lid] = dx;
        workgroupBarrier();

        // 2. Thread 0 computes prefix products and batch inverse
        if (lid == 0u) {
            var prod = fe_one();
            
            // Compute prefix products: shared_prod[i] = dx_0 * dx_1 * ... * dx_i
            for (var i = 0u; i < 64u; i++) {
                prod = fe_mul(prod, shared_dx[i]);
                shared_prod[i] = prod;
            }

            // Invert total product
            var inv = fe_inv(prod);

            // Compute individual inverses backwards
            // inv_dx[i] = inv_total * (dx_0 * ... * dx_{i-1}) * (dx_{i+1} * ... * dx_{n-1})
            var inv_acc = inv;
            for (var i = 63u; i > 0u; i--) {
                let prev_prod = shared_prod[i - 1u];
                let val_inv = fe_mul(inv_acc, prev_prod);
                let val_dx = shared_dx[i];
                
                // Store result in shared_prod
                shared_prod[i] = val_inv;
                
                inv_acc = fe_mul(inv_acc, val_dx);
            }
            shared_prod[0] = inv_acc;
        }
        workgroupBarrier();

        // 3. Read inverse dx from shared memory
        let dx_inv = shared_prod[lid];

        // =====================================================================
        // POINT ADDITION AND DP CHECK
        // =====================================================================

        if (valid) {
            // Check for DP before the jump (on current position)
            if (!dp_stored) {
                if ((px[0] & config.dp_mask_lo.x) == 0u) {
                    k.x = px;
                    k.y = py;
                    store_dp(k, kid);
                    dp_stored = true;
                }
            }

            // Perform affine addition: P = P + jump_point
            let result = affine_add_with_inv(px, py, jump_point.x, jump_point.y, dx_inv);
            px = result.x;
            py = result.y;

            // Update distance
            k.dist = scalar_add_256(k.dist, jump_dist);
        }
    }

    // Write back updated state
    if (valid) {
        k.x = px;
        k.y = py;
        kangaroos[kid] = k;
    }
}
