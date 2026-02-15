// =============================================================================
// secp256k1 Field Arithmetic (256-bit using u32 limbs)
// =============================================================================
// Field prime: p = 2^256 - 2^32 - 977
// Representation: 8 x u32 limbs, little-endian

// Prime p in limbs (little-endian): 2^256 - 2^32 - 977
const P0: u32 = 0xFFFFFC2Fu;  // = -977 mod 2^32 = 4294966319
const P1: u32 = 0xFFFFFFFEu;  // = -2 mod 2^32 (from -2^32)
const P2: u32 = 0xFFFFFFFFu;
const P3: u32 = 0xFFFFFFFFu;
const P4: u32 = 0xFFFFFFFFu;
const P5: u32 = 0xFFFFFFFFu;
const P6: u32 = 0xFFFFFFFFu;
const P7: u32 = 0xFFFFFFFFu;

// -----------------------------------------------------------------------------
// Addition: c = a + b (mod p)
// Returns result without full reduction (may be >= p but < 2p)
// -----------------------------------------------------------------------------
fn fe_add(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var c: array<u32, 8>;
    var carry: u32 = 0u;
    var sum: u32;

    sum = a[0] + b[0]; carry = select(0u, 1u, sum < a[0]); c[0] = sum;
    sum = a[1] + b[1] + carry; carry = select(0u, 1u, sum < a[1] || (carry == 1u && sum == a[1])); c[1] = sum;
    sum = a[2] + b[2] + carry; carry = select(0u, 1u, sum < a[2] || (carry == 1u && sum == a[2])); c[2] = sum;
    sum = a[3] + b[3] + carry; carry = select(0u, 1u, sum < a[3] || (carry == 1u && sum == a[3])); c[3] = sum;
    sum = a[4] + b[4] + carry; carry = select(0u, 1u, sum < a[4] || (carry == 1u && sum == a[4])); c[4] = sum;
    sum = a[5] + b[5] + carry; carry = select(0u, 1u, sum < a[5] || (carry == 1u && sum == a[5])); c[5] = sum;
    sum = a[6] + b[6] + carry; carry = select(0u, 1u, sum < a[6] || (carry == 1u && sum == a[6])); c[6] = sum;
    sum = a[7] + b[7] + carry; carry = select(0u, 1u, sum < a[7] || (carry == 1u && sum == a[7])); c[7] = sum;

    // If overflow (result >= 2^256), reduce by adding 2^32 + 977 (because 2^256 ≡ 2^32 + 977 mod P)
    if (carry == 1u) {
        var old: u32;
        old = c[0]; sum = c[0] + 977u; carry = select(0u, 1u, sum < old); c[0] = sum;
        old = c[1]; sum = c[1] + 1u + carry; carry = select(0u, 1u, sum < old || (carry == 1u && sum == old)); c[1] = sum;
        old = c[2]; sum = c[2] + carry; carry = select(0u, 1u, sum < old); c[2] = sum;
        old = c[3]; sum = c[3] + carry; carry = select(0u, 1u, sum < old); c[3] = sum;
        old = c[4]; sum = c[4] + carry; carry = select(0u, 1u, sum < old); c[4] = sum;
        old = c[5]; sum = c[5] + carry; carry = select(0u, 1u, sum < old); c[5] = sum;
        old = c[6]; sum = c[6] + carry; carry = select(0u, 1u, sum < old); c[6] = sum;
        c[7] = c[7] + carry;
    }

    return c;
}

// -----------------------------------------------------------------------------
// Subtraction: c = a - b (mod p)
// If a < b, adds p to result
// -----------------------------------------------------------------------------
fn fe_sub(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var c: array<u32, 8>;
    var borrow: u32 = 0u;
    var diff: u32;

    diff = a[0] - b[0]; borrow = select(0u, 1u, a[0] < b[0]); c[0] = diff;
    diff = a[1] - b[1] - borrow; borrow = select(0u, 1u, a[1] < b[1] + borrow || (borrow == 1u && b[1] == 0xFFFFFFFFu)); c[1] = diff;
    diff = a[2] - b[2] - borrow; borrow = select(0u, 1u, a[2] < b[2] + borrow || (borrow == 1u && b[2] == 0xFFFFFFFFu)); c[2] = diff;
    diff = a[3] - b[3] - borrow; borrow = select(0u, 1u, a[3] < b[3] + borrow || (borrow == 1u && b[3] == 0xFFFFFFFFu)); c[3] = diff;
    diff = a[4] - b[4] - borrow; borrow = select(0u, 1u, a[4] < b[4] + borrow || (borrow == 1u && b[4] == 0xFFFFFFFFu)); c[4] = diff;
    diff = a[5] - b[5] - borrow; borrow = select(0u, 1u, a[5] < b[5] + borrow || (borrow == 1u && b[5] == 0xFFFFFFFFu)); c[5] = diff;
    diff = a[6] - b[6] - borrow; borrow = select(0u, 1u, a[6] < b[6] + borrow || (borrow == 1u && b[6] == 0xFFFFFFFFu)); c[6] = diff;
    diff = a[7] - b[7] - borrow; c[7] = diff; borrow = select(0u, 1u, a[7] < b[7] + borrow || (borrow == 1u && b[7] == 0xFFFFFFFFu));

    // If borrow, add p back
    if (borrow == 1u) {
        var carry2: u32 = 0u;
        var sum2: u32;

        sum2 = c[0] + P0; carry2 = select(0u, 1u, sum2 < c[0]); c[0] = sum2;
        sum2 = c[1] + P1 + carry2; carry2 = select(0u, 1u, sum2 < c[1] || (carry2 == 1u && sum2 == c[1])); c[1] = sum2;
        sum2 = c[2] + P2 + carry2; carry2 = select(0u, 1u, sum2 < c[2] || (carry2 == 1u && sum2 == c[2])); c[2] = sum2;
        sum2 = c[3] + P3 + carry2; carry2 = select(0u, 1u, sum2 < c[3] || (carry2 == 1u && sum2 == c[3])); c[3] = sum2;
        sum2 = c[4] + P4 + carry2; carry2 = select(0u, 1u, sum2 < c[4] || (carry2 == 1u && sum2 == c[4])); c[4] = sum2;
        sum2 = c[5] + P5 + carry2; carry2 = select(0u, 1u, sum2 < c[5] || (carry2 == 1u && sum2 == c[5])); c[5] = sum2;
        sum2 = c[6] + P6 + carry2; carry2 = select(0u, 1u, sum2 < c[6] || (carry2 == 1u && sum2 == c[6])); c[6] = sum2;
        sum2 = c[7] + P7 + carry2; c[7] = sum2;
    }

    return c;
}

// -----------------------------------------------------------------------------
// 32x32 -> 64 bit multiplication using 16-bit halves
// Returns (lo, hi) as vec2<u32>
// -----------------------------------------------------------------------------
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    let mid = p1 + p2;
    let mid_overflow = select(0u, 1u, mid < p1);

    let lo1 = p0 + (mid << 16u);
    let lo_carry1 = select(0u, 1u, lo1 < p0);

    let hi1 = p3 + (mid >> 16u) + (mid_overflow << 16u) + lo_carry1;

    return vec2<u32>(lo1, hi1);
}

// -----------------------------------------------------------------------------
// Double a field element: c = 2*a (mod p)
// -----------------------------------------------------------------------------
fn fe_double(a: array<u32, 8>) -> array<u32, 8> {
    return fe_add(a, a);
}

// -----------------------------------------------------------------------------
// Field multiplication: c = a * b (mod p)
// Uses schoolbook multiplication with secp256k1 reduction
// secp256k1: p = 2^256 - 2^32 - 977
// For reduction: 2^256 ≡ 2^32 + 977 (mod p)
//
// Approach: Classic schoolbook with row-based accumulation
// Process one row at a time, immediately propagate carry through the row
// This ensures no accumulator exceeds 32 bits
// -----------------------------------------------------------------------------

fn fe_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    // =========================================================================
    // STEP 1: Full schoolbook multiplication to get 16 limbs (512-bit product)
    // Using individual variables to avoid potential RADV array indexing issues
    // =========================================================================
    var p0: u32 = 0u; var p1: u32 = 0u; var p2: u32 = 0u; var p3: u32 = 0u;
    var p4: u32 = 0u; var p5: u32 = 0u; var p6: u32 = 0u; var p7: u32 = 0u;
    var p8: u32 = 0u; var p9: u32 = 0u; var p10: u32 = 0u; var p11: u32 = 0u;
    var p12: u32 = 0u; var p13: u32 = 0u; var p14: u32 = 0u; var p15: u32 = 0u;

    var t: vec2<u32>;
    var carry: u32;
    var s: u32;
    var hi: u32;

    // Row 0: a[0] * b[j]
    carry = 0u;
    t = mul32(a[0], b[0]); s = p0 + t.x; carry = select(0u, 1u, s < p0) + t.y; p0 = s;
    t = mul32(a[0], b[1]); s = p1 + t.x; hi = select(0u, 1u, s < p1); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p1 = s;
    t = mul32(a[0], b[2]); s = p2 + t.x; hi = select(0u, 1u, s < p2); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p2 = s;
    t = mul32(a[0], b[3]); s = p3 + t.x; hi = select(0u, 1u, s < p3); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p3 = s;
    t = mul32(a[0], b[4]); s = p4 + t.x; hi = select(0u, 1u, s < p4); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p4 = s;
    t = mul32(a[0], b[5]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[0], b[6]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[0], b[7]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    p8 = carry;

    // Row 1
    carry = 0u;
    t = mul32(a[1], b[0]); s = p1 + t.x; hi = select(0u, 1u, s < p1); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p1 = s;
    t = mul32(a[1], b[1]); s = p2 + t.x; hi = select(0u, 1u, s < p2); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p2 = s;
    t = mul32(a[1], b[2]); s = p3 + t.x; hi = select(0u, 1u, s < p3); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p3 = s;
    t = mul32(a[1], b[3]); s = p4 + t.x; hi = select(0u, 1u, s < p4); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p4 = s;
    t = mul32(a[1], b[4]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[1], b[5]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[1], b[6]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[1], b[7]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    p9 = carry;

    // Row 2
    carry = 0u;
    t = mul32(a[2], b[0]); s = p2 + t.x; hi = select(0u, 1u, s < p2); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p2 = s;
    t = mul32(a[2], b[1]); s = p3 + t.x; hi = select(0u, 1u, s < p3); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p3 = s;
    t = mul32(a[2], b[2]); s = p4 + t.x; hi = select(0u, 1u, s < p4); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p4 = s;
    t = mul32(a[2], b[3]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[2], b[4]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[2], b[5]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[2], b[6]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[2], b[7]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    p10 = carry;

    // Row 3
    carry = 0u;
    t = mul32(a[3], b[0]); s = p3 + t.x; hi = select(0u, 1u, s < p3); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p3 = s;
    t = mul32(a[3], b[1]); s = p4 + t.x; hi = select(0u, 1u, s < p4); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p4 = s;
    t = mul32(a[3], b[2]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[3], b[3]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[3], b[4]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[3], b[5]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[3], b[6]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    t = mul32(a[3], b[7]); s = p10 + t.x; hi = select(0u, 1u, s < p10); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p10 = s;
    p11 = carry;

    // Row 4
    carry = 0u;
    t = mul32(a[4], b[0]); s = p4 + t.x; hi = select(0u, 1u, s < p4); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p4 = s;
    t = mul32(a[4], b[1]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[4], b[2]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[4], b[3]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[4], b[4]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[4], b[5]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    t = mul32(a[4], b[6]); s = p10 + t.x; hi = select(0u, 1u, s < p10); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p10 = s;
    t = mul32(a[4], b[7]); s = p11 + t.x; hi = select(0u, 1u, s < p11); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p11 = s;
    p12 = carry;

    // Row 5
    carry = 0u;
    t = mul32(a[5], b[0]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[5], b[1]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[5], b[2]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[5], b[3]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[5], b[4]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    t = mul32(a[5], b[5]); s = p10 + t.x; hi = select(0u, 1u, s < p10); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p10 = s;
    t = mul32(a[5], b[6]); s = p11 + t.x; hi = select(0u, 1u, s < p11); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p11 = s;
    t = mul32(a[5], b[7]); s = p12 + t.x; hi = select(0u, 1u, s < p12); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p12 = s;
    p13 = carry;

    // Row 6
    carry = 0u;
    t = mul32(a[6], b[0]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[6], b[1]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[6], b[2]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[6], b[3]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    t = mul32(a[6], b[4]); s = p10 + t.x; hi = select(0u, 1u, s < p10); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p10 = s;
    t = mul32(a[6], b[5]); s = p11 + t.x; hi = select(0u, 1u, s < p11); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p11 = s;
    t = mul32(a[6], b[6]); s = p12 + t.x; hi = select(0u, 1u, s < p12); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p12 = s;
    t = mul32(a[6], b[7]); s = p13 + t.x; hi = select(0u, 1u, s < p13); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p13 = s;
    p14 = carry;

    // Row 7
    carry = 0u;
    t = mul32(a[7], b[0]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[7], b[1]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[7], b[2]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    t = mul32(a[7], b[3]); s = p10 + t.x; hi = select(0u, 1u, s < p10); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p10 = s;
    t = mul32(a[7], b[4]); s = p11 + t.x; hi = select(0u, 1u, s < p11); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p11 = s;
    t = mul32(a[7], b[5]); s = p12 + t.x; hi = select(0u, 1u, s < p12); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p12 = s;
    t = mul32(a[7], b[6]); s = p13 + t.x; hi = select(0u, 1u, s < p13); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p13 = s;
    t = mul32(a[7], b[7]); s = p14 + t.x; hi = select(0u, 1u, s < p14); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p14 = s;
    p15 = carry;

    // =========================================================================
    // STEP 2: Reduction using 2^256 ≡ 2^32 + 977 (mod p)
    // FIXED: Complete carry propagation through ALL remaining limbs
    // =========================================================================
    var h: u32;
    var old: u32;
    var c: u32;

    // Reduce p15
    h = p15; p15 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p7; p7 = p7 + t.x; c = select(0u, 1u, p7 < old);
        old = p8; p8 = p8 + t.y + c; c = select(0u, 1u, p8 < old || (c == 1u && t.y == 0u && p8 == old));
        old = p8; p8 = p8 + h; c = c + select(0u, 1u, p8 < old);
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
        if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
        if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
        if (c > 0u) { old = p13; p13 = p13 + c; c = select(0u, 1u, p13 < old); }
        if (c > 0u) { old = p14; p14 = p14 + c; c = select(0u, 1u, p14 < old); }
        if (c > 0u) { p15 = p15 + c; }
    }

    // Reduce p14
    h = p14; p14 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p6; p6 = p6 + t.x; c = select(0u, 1u, p6 < old);
        old = p7; p7 = p7 + t.y + c; c = select(0u, 1u, p7 < old || (c == 1u && t.y == 0u && p7 == old));
        old = p7; p7 = p7 + h; c = c + select(0u, 1u, p7 < old);
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
        if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
        if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
        if (c > 0u) { old = p13; p13 = p13 + c; c = select(0u, 1u, p13 < old); }
        if (c > 0u) { p14 = p14 + c; }
    }

    // Reduce p13
    h = p13; p13 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p5; p5 = p5 + t.x; c = select(0u, 1u, p5 < old);
        old = p6; p6 = p6 + t.y + c; c = select(0u, 1u, p6 < old || (c == 1u && t.y == 0u && p6 == old));
        old = p6; p6 = p6 + h; c = c + select(0u, 1u, p6 < old);
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
        if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
        if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
        if (c > 0u) { p13 = p13 + c; }
    }

    // Reduce p12
    h = p12; p12 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p4; p4 = p4 + t.x; c = select(0u, 1u, p4 < old);
        old = p5; p5 = p5 + t.y + c; c = select(0u, 1u, p5 < old || (c == 1u && t.y == 0u && p5 == old));
        old = p5; p5 = p5 + h; c = c + select(0u, 1u, p5 < old);
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
        if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
        if (c > 0u) { p12 = p12 + c; }
    }

    // Reduce p11
    h = p11; p11 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p3; p3 = p3 + t.x; c = select(0u, 1u, p3 < old);
        old = p4; p4 = p4 + t.y + c; c = select(0u, 1u, p4 < old || (c == 1u && t.y == 0u && p4 == old));
        old = p4; p4 = p4 + h; c = c + select(0u, 1u, p4 < old);
        if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
        if (c > 0u) { p11 = p11 + c; }
    }

    // Reduce p10
    h = p10; p10 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p2; p2 = p2 + t.x; c = select(0u, 1u, p2 < old);
        old = p3; p3 = p3 + t.y + c; c = select(0u, 1u, p3 < old || (c == 1u && t.y == 0u && p3 == old));
        old = p3; p3 = p3 + h; c = c + select(0u, 1u, p3 < old);
        if (c > 0u) { old = p4; p4 = p4 + c; c = select(0u, 1u, p4 < old); }
        if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { p10 = p10 + c; }
    }

    // Reduce p9
    h = p9; p9 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p1; p1 = p1 + t.x; c = select(0u, 1u, p1 < old);
        old = p2; p2 = p2 + t.y + c; c = select(0u, 1u, p2 < old || (c == 1u && t.y == 0u && p2 == old));
        old = p2; p2 = p2 + h; c = c + select(0u, 1u, p2 < old);
        if (c > 0u) { old = p3; p3 = p3 + c; c = select(0u, 1u, p3 < old); }
        if (c > 0u) { old = p4; p4 = p4 + c; c = select(0u, 1u, p4 < old); }
        if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { p9 = p9 + c; }
    }

    // Reduce p8 (first pass)
    h = p8; p8 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p0; p0 = p0 + t.x; c = select(0u, 1u, p0 < old);
        old = p1; p1 = p1 + t.y + c; c = select(0u, 1u, p1 < old || (c == 1u && t.y == 0u && p1 == old));
        old = p1; p1 = p1 + h; c = c + select(0u, 1u, p1 < old);
        if (c > 0u) { old = p2; p2 = p2 + c; c = select(0u, 1u, p2 < old); }
        if (c > 0u) { old = p3; p3 = p3 + c; c = select(0u, 1u, p3 < old); }
        if (c > 0u) { old = p4; p4 = p4 + c; c = select(0u, 1u, p4 < old); }
        if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { p8 = p8 + c; }
    }

    // Reduce p8 (second pass if needed)
    h = p8; p8 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p0; p0 = p0 + t.x; c = select(0u, 1u, p0 < old);
        old = p1; p1 = p1 + t.y + c; c = select(0u, 1u, p1 < old || (c == 1u && t.y == 0u && p1 == old));
        old = p1; p1 = p1 + h; c = c + select(0u, 1u, p1 < old);
        if (c > 0u) { old = p2; p2 = p2 + c; c = select(0u, 1u, p2 < old); }
        if (c > 0u) { old = p3; p3 = p3 + c; c = select(0u, 1u, p3 < old); }
        if (c > 0u) { old = p4; p4 = p4 + c; c = select(0u, 1u, p4 < old); }
        if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; }
    }

    var result: array<u32, 8>;
    result[0] = p0; result[1] = p1; result[2] = p2; result[3] = p3;
    result[4] = p4; result[5] = p5; result[6] = p6; result[7] = p7;
    return result;
}

// -----------------------------------------------------------------------------
// Field squaring: c = a² (mod p)
// Optimized: 36 mul32 calls instead of 64 (exploits a[i]*a[j] == a[j]*a[i])
// Method: cross products (28) + double + diagonal products (8) + reduction
// -----------------------------------------------------------------------------
fn fe_square(a: array<u32, 8>) -> array<u32, 8> {
    var p0: u32 = 0u; var p1: u32 = 0u; var p2: u32 = 0u; var p3: u32 = 0u;
    var p4: u32 = 0u; var p5: u32 = 0u; var p6: u32 = 0u; var p7: u32 = 0u;
    var p8: u32 = 0u; var p9: u32 = 0u; var p10: u32 = 0u; var p11: u32 = 0u;
    var p12: u32 = 0u; var p13: u32 = 0u; var p14: u32 = 0u; var p15: u32 = 0u;

    var t: vec2<u32>;
    var carry: u32;
    var s: u32;
    var hi: u32;

    // =========================================================================
    // STEP 1: Cross products (upper triangle only: a[i]*a[j] for i < j)
    // 28 mul32 calls instead of 64 for full multiply
    // =========================================================================

    // Cross Row 0: a[0] * a[1..7]
    carry = 0u;
    t = mul32(a[0], a[1]); s = p1 + t.x; carry = select(0u, 1u, s < p1) + t.y; p1 = s;
    t = mul32(a[0], a[2]); s = p2 + t.x; hi = select(0u, 1u, s < p2); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p2 = s;
    t = mul32(a[0], a[3]); s = p3 + t.x; hi = select(0u, 1u, s < p3); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p3 = s;
    t = mul32(a[0], a[4]); s = p4 + t.x; hi = select(0u, 1u, s < p4); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p4 = s;
    t = mul32(a[0], a[5]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[0], a[6]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[0], a[7]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    p8 = carry;

    // Cross Row 1: a[1] * a[2..7]
    carry = 0u;
    t = mul32(a[1], a[2]); s = p3 + t.x; carry = select(0u, 1u, s < p3) + t.y; p3 = s;
    t = mul32(a[1], a[3]); s = p4 + t.x; hi = select(0u, 1u, s < p4); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p4 = s;
    t = mul32(a[1], a[4]); s = p5 + t.x; hi = select(0u, 1u, s < p5); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p5 = s;
    t = mul32(a[1], a[5]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[1], a[6]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[1], a[7]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    p9 = carry;

    // Cross Row 2: a[2] * a[3..7]
    carry = 0u;
    t = mul32(a[2], a[3]); s = p5 + t.x; carry = select(0u, 1u, s < p5) + t.y; p5 = s;
    t = mul32(a[2], a[4]); s = p6 + t.x; hi = select(0u, 1u, s < p6); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p6 = s;
    t = mul32(a[2], a[5]); s = p7 + t.x; hi = select(0u, 1u, s < p7); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p7 = s;
    t = mul32(a[2], a[6]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[2], a[7]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    p10 = carry;

    // Cross Row 3: a[3] * a[4..7]
    carry = 0u;
    t = mul32(a[3], a[4]); s = p7 + t.x; carry = select(0u, 1u, s < p7) + t.y; p7 = s;
    t = mul32(a[3], a[5]); s = p8 + t.x; hi = select(0u, 1u, s < p8); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p8 = s;
    t = mul32(a[3], a[6]); s = p9 + t.x; hi = select(0u, 1u, s < p9); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p9 = s;
    t = mul32(a[3], a[7]); s = p10 + t.x; hi = select(0u, 1u, s < p10); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p10 = s;
    p11 = carry;

    // Cross Row 4: a[4] * a[5..7]
    carry = 0u;
    t = mul32(a[4], a[5]); s = p9 + t.x; carry = select(0u, 1u, s < p9) + t.y; p9 = s;
    t = mul32(a[4], a[6]); s = p10 + t.x; hi = select(0u, 1u, s < p10); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p10 = s;
    t = mul32(a[4], a[7]); s = p11 + t.x; hi = select(0u, 1u, s < p11); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p11 = s;
    p12 = carry;

    // Cross Row 5: a[5] * a[6..7]
    carry = 0u;
    t = mul32(a[5], a[6]); s = p11 + t.x; carry = select(0u, 1u, s < p11) + t.y; p11 = s;
    t = mul32(a[5], a[7]); s = p12 + t.x; hi = select(0u, 1u, s < p12); s = s + carry; hi = hi + select(0u, 1u, s < carry); carry = hi + t.y; p12 = s;
    p13 = carry;

    // Cross Row 6: a[6] * a[7]
    t = mul32(a[6], a[7]); s = p13 + t.x; carry = select(0u, 1u, s < p13) + t.y; p13 = s;
    p14 = carry;

    // =========================================================================
    // STEP 2: Double all cross products (512-bit left shift by 1)
    // =========================================================================
    p15 = p14 >> 31u;
    p14 = (p14 << 1u) | (p13 >> 31u);
    p13 = (p13 << 1u) | (p12 >> 31u);
    p12 = (p12 << 1u) | (p11 >> 31u);
    p11 = (p11 << 1u) | (p10 >> 31u);
    p10 = (p10 << 1u) | (p9 >> 31u);
    p9 = (p9 << 1u) | (p8 >> 31u);
    p8 = (p8 << 1u) | (p7 >> 31u);
    p7 = (p7 << 1u) | (p6 >> 31u);
    p6 = (p6 << 1u) | (p5 >> 31u);
    p5 = (p5 << 1u) | (p4 >> 31u);
    p4 = (p4 << 1u) | (p3 >> 31u);
    p3 = (p3 << 1u) | (p2 >> 31u);
    p2 = (p2 << 1u) | (p1 >> 31u);
    p1 = p1 << 1u;
    // p0 stays 0 (no cross terms contribute to it)

    // =========================================================================
    // STEP 3: Add diagonal products a[i]² to p[2i], p[2i+1]
    // =========================================================================
    var c: u32;
    var old: u32;

    // a[0]²
    t = mul32(a[0], a[0]);
    p0 = t.x;
    old = p1; p1 = p1 + t.y; c = select(0u, 1u, p1 < old);
    if (c > 0u) { old = p2; p2 = p2 + c; c = select(0u, 1u, p2 < old); }
    if (c > 0u) { old = p3; p3 = p3 + c; c = select(0u, 1u, p3 < old); }
    if (c > 0u) { old = p4; p4 = p4 + c; c = select(0u, 1u, p4 < old); }
    if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
    if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
    if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
    if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
    if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
    if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
    if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
    if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
    if (c > 0u) { old = p13; p13 = p13 + c; c = select(0u, 1u, p13 < old); }
    if (c > 0u) { old = p14; p14 = p14 + c; c = select(0u, 1u, p14 < old); }
    if (c > 0u) { p15 = p15 + c; }

    // a[1]²
    t = mul32(a[1], a[1]);
    old = p2; p2 = p2 + t.x; c = select(0u, 1u, p2 < old);
    s = p3 + t.y; hi = select(0u, 1u, s < p3); s = s + c; hi = hi + select(0u, 1u, s < c); p3 = s; c = hi;
    if (c > 0u) { old = p4; p4 = p4 + c; c = select(0u, 1u, p4 < old); }
    if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
    if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
    if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
    if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
    if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
    if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
    if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
    if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
    if (c > 0u) { old = p13; p13 = p13 + c; c = select(0u, 1u, p13 < old); }
    if (c > 0u) { old = p14; p14 = p14 + c; c = select(0u, 1u, p14 < old); }
    if (c > 0u) { p15 = p15 + c; }

    // a[2]²
    t = mul32(a[2], a[2]);
    old = p4; p4 = p4 + t.x; c = select(0u, 1u, p4 < old);
    s = p5 + t.y; hi = select(0u, 1u, s < p5); s = s + c; hi = hi + select(0u, 1u, s < c); p5 = s; c = hi;
    if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
    if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
    if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
    if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
    if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
    if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
    if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
    if (c > 0u) { old = p13; p13 = p13 + c; c = select(0u, 1u, p13 < old); }
    if (c > 0u) { old = p14; p14 = p14 + c; c = select(0u, 1u, p14 < old); }
    if (c > 0u) { p15 = p15 + c; }

    // a[3]²
    t = mul32(a[3], a[3]);
    old = p6; p6 = p6 + t.x; c = select(0u, 1u, p6 < old);
    s = p7 + t.y; hi = select(0u, 1u, s < p7); s = s + c; hi = hi + select(0u, 1u, s < c); p7 = s; c = hi;
    if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
    if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
    if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
    if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
    if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
    if (c > 0u) { old = p13; p13 = p13 + c; c = select(0u, 1u, p13 < old); }
    if (c > 0u) { old = p14; p14 = p14 + c; c = select(0u, 1u, p14 < old); }
    if (c > 0u) { p15 = p15 + c; }

    // a[4]²
    t = mul32(a[4], a[4]);
    old = p8; p8 = p8 + t.x; c = select(0u, 1u, p8 < old);
    s = p9 + t.y; hi = select(0u, 1u, s < p9); s = s + c; hi = hi + select(0u, 1u, s < c); p9 = s; c = hi;
    if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
    if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
    if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
    if (c > 0u) { old = p13; p13 = p13 + c; c = select(0u, 1u, p13 < old); }
    if (c > 0u) { old = p14; p14 = p14 + c; c = select(0u, 1u, p14 < old); }
    if (c > 0u) { p15 = p15 + c; }

    // a[5]²
    t = mul32(a[5], a[5]);
    old = p10; p10 = p10 + t.x; c = select(0u, 1u, p10 < old);
    s = p11 + t.y; hi = select(0u, 1u, s < p11); s = s + c; hi = hi + select(0u, 1u, s < c); p11 = s; c = hi;
    if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
    if (c > 0u) { old = p13; p13 = p13 + c; c = select(0u, 1u, p13 < old); }
    if (c > 0u) { old = p14; p14 = p14 + c; c = select(0u, 1u, p14 < old); }
    if (c > 0u) { p15 = p15 + c; }

    // a[6]²
    t = mul32(a[6], a[6]);
    old = p12; p12 = p12 + t.x; c = select(0u, 1u, p12 < old);
    s = p13 + t.y; hi = select(0u, 1u, s < p13); s = s + c; hi = hi + select(0u, 1u, s < c); p13 = s; c = hi;
    if (c > 0u) { old = p14; p14 = p14 + c; c = select(0u, 1u, p14 < old); }
    if (c > 0u) { p15 = p15 + c; }

    // a[7]²
    t = mul32(a[7], a[7]);
    old = p14; p14 = p14 + t.x; c = select(0u, 1u, p14 < old);
    p15 = p15 + t.y + c;

    // =========================================================================
    // STEP 4: Reduction using 2^256 ≡ 2^32 + 977 (mod p)
    // Identical to fe_mul reduction
    // =========================================================================
    var h: u32;

    // Reduce p15
    h = p15; p15 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p7; p7 = p7 + t.x; c = select(0u, 1u, p7 < old);
        old = p8; p8 = p8 + t.y + c; c = select(0u, 1u, p8 < old || (c == 1u && t.y == 0u && p8 == old));
        old = p8; p8 = p8 + h; c = c + select(0u, 1u, p8 < old);
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
        if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
        if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
        if (c > 0u) { old = p13; p13 = p13 + c; c = select(0u, 1u, p13 < old); }
        if (c > 0u) { old = p14; p14 = p14 + c; c = select(0u, 1u, p14 < old); }
        if (c > 0u) { p15 = p15 + c; }
    }

    // Reduce p14
    h = p14; p14 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p6; p6 = p6 + t.x; c = select(0u, 1u, p6 < old);
        old = p7; p7 = p7 + t.y + c; c = select(0u, 1u, p7 < old || (c == 1u && t.y == 0u && p7 == old));
        old = p7; p7 = p7 + h; c = c + select(0u, 1u, p7 < old);
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
        if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
        if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
        if (c > 0u) { old = p13; p13 = p13 + c; c = select(0u, 1u, p13 < old); }
        if (c > 0u) { p14 = p14 + c; }
    }

    // Reduce p13
    h = p13; p13 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p5; p5 = p5 + t.x; c = select(0u, 1u, p5 < old);
        old = p6; p6 = p6 + t.y + c; c = select(0u, 1u, p6 < old || (c == 1u && t.y == 0u && p6 == old));
        old = p6; p6 = p6 + h; c = c + select(0u, 1u, p6 < old);
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
        if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
        if (c > 0u) { old = p12; p12 = p12 + c; c = select(0u, 1u, p12 < old); }
        if (c > 0u) { p13 = p13 + c; }
    }

    // Reduce p12
    h = p12; p12 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p4; p4 = p4 + t.x; c = select(0u, 1u, p4 < old);
        old = p5; p5 = p5 + t.y + c; c = select(0u, 1u, p5 < old || (c == 1u && t.y == 0u && p5 == old));
        old = p5; p5 = p5 + h; c = c + select(0u, 1u, p5 < old);
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
        if (c > 0u) { old = p11; p11 = p11 + c; c = select(0u, 1u, p11 < old); }
        if (c > 0u) { p12 = p12 + c; }
    }

    // Reduce p11
    h = p11; p11 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p3; p3 = p3 + t.x; c = select(0u, 1u, p3 < old);
        old = p4; p4 = p4 + t.y + c; c = select(0u, 1u, p4 < old || (c == 1u && t.y == 0u && p4 == old));
        old = p4; p4 = p4 + h; c = c + select(0u, 1u, p4 < old);
        if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { old = p10; p10 = p10 + c; c = select(0u, 1u, p10 < old); }
        if (c > 0u) { p11 = p11 + c; }
    }

    // Reduce p10
    h = p10; p10 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p2; p2 = p2 + t.x; c = select(0u, 1u, p2 < old);
        old = p3; p3 = p3 + t.y + c; c = select(0u, 1u, p3 < old || (c == 1u && t.y == 0u && p3 == old));
        old = p3; p3 = p3 + h; c = c + select(0u, 1u, p3 < old);
        if (c > 0u) { old = p4; p4 = p4 + c; c = select(0u, 1u, p4 < old); }
        if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { old = p9; p9 = p9 + c; c = select(0u, 1u, p9 < old); }
        if (c > 0u) { p10 = p10 + c; }
    }

    // Reduce p9
    h = p9; p9 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p1; p1 = p1 + t.x; c = select(0u, 1u, p1 < old);
        old = p2; p2 = p2 + t.y + c; c = select(0u, 1u, p2 < old || (c == 1u && t.y == 0u && p2 == old));
        old = p2; p2 = p2 + h; c = c + select(0u, 1u, p2 < old);
        if (c > 0u) { old = p3; p3 = p3 + c; c = select(0u, 1u, p3 < old); }
        if (c > 0u) { old = p4; p4 = p4 + c; c = select(0u, 1u, p4 < old); }
        if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { old = p8; p8 = p8 + c; c = select(0u, 1u, p8 < old); }
        if (c > 0u) { p9 = p9 + c; }
    }

    // Reduce p8 (first pass)
    h = p8; p8 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p0; p0 = p0 + t.x; c = select(0u, 1u, p0 < old);
        old = p1; p1 = p1 + t.y + c; c = select(0u, 1u, p1 < old || (c == 1u && t.y == 0u && p1 == old));
        old = p1; p1 = p1 + h; c = c + select(0u, 1u, p1 < old);
        if (c > 0u) { old = p2; p2 = p2 + c; c = select(0u, 1u, p2 < old); }
        if (c > 0u) { old = p3; p3 = p3 + c; c = select(0u, 1u, p3 < old); }
        if (c > 0u) { old = p4; p4 = p4 + c; c = select(0u, 1u, p4 < old); }
        if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; c = select(0u, 1u, p7 < old); }
        if (c > 0u) { p8 = p8 + c; }
    }

    // Reduce p8 (second pass if needed)
    h = p8; p8 = 0u;
    if (h != 0u) {
        t = mul32(h, 977u);
        old = p0; p0 = p0 + t.x; c = select(0u, 1u, p0 < old);
        old = p1; p1 = p1 + t.y + c; c = select(0u, 1u, p1 < old || (c == 1u && t.y == 0u && p1 == old));
        old = p1; p1 = p1 + h; c = c + select(0u, 1u, p1 < old);
        if (c > 0u) { old = p2; p2 = p2 + c; c = select(0u, 1u, p2 < old); }
        if (c > 0u) { old = p3; p3 = p3 + c; c = select(0u, 1u, p3 < old); }
        if (c > 0u) { old = p4; p4 = p4 + c; c = select(0u, 1u, p4 < old); }
        if (c > 0u) { old = p5; p5 = p5 + c; c = select(0u, 1u, p5 < old); }
        if (c > 0u) { old = p6; p6 = p6 + c; c = select(0u, 1u, p6 < old); }
        if (c > 0u) { old = p7; p7 = p7 + c; }
    }

    var result: array<u32, 8>;
    result[0] = p0; result[1] = p1; result[2] = p2; result[3] = p3;
    result[4] = p4; result[5] = p5; result[6] = p6; result[7] = p7;
    return result;
}

// -----------------------------------------------------------------------------
// Field inversion: c = a^(-1) (mod p)
// Uses Fermat's little theorem: a^(-1) = a^(p-2) mod p
//
// Optimized addition chain: 255 squarings + 15 multiplications
// Based on Peter Dettman's chain for libsecp256k1 (2014)
// Reference: https://briansmith.org/ecc-inversion-addition-chains-01
//
// p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
// Bit structure: [223 ones][0][22 ones][0000][10][11][01]
// -----------------------------------------------------------------------------
fn fe_inv(a: array<u32, 8>) -> array<u32, 8> {
    // ===== BUILD: compute a^(2^n - 1) blocks =====

    let a2 = fe_square(a);                     // a^2 (reused in tail)
    let x2 = fe_mul(a2, a);                    // a^(2^2-1) = a^3

    var t = fe_square(x2);
    let x3 = fe_mul(t, a);                     // a^(2^3-1) = a^7

    t = x3;
    for (var i = 0u; i < 3u; i = i + 1u) { t = fe_square(t); }
    t = fe_mul(t, x3);                         // a^(2^6-1)
    for (var i = 0u; i < 3u; i = i + 1u) { t = fe_square(t); }
    t = fe_mul(t, x3);                         // a^(2^9-1)
    for (var i = 0u; i < 2u; i = i + 1u) { t = fe_square(t); }
    let x11 = fe_mul(t, x2);                   // a^(2^11-1)

    t = x11;
    for (var i = 0u; i < 11u; i = i + 1u) { t = fe_square(t); }
    let x22 = fe_mul(t, x11);                  // a^(2^22-1)

    t = x22;
    for (var i = 0u; i < 22u; i = i + 1u) { t = fe_square(t); }
    let x44 = fe_mul(t, x22);                  // a^(2^44-1)

    t = x44;
    for (var i = 0u; i < 44u; i = i + 1u) { t = fe_square(t); }
    let x88 = fe_mul(t, x44);                  // a^(2^88-1)

    // ===== ASSEMBLY: combine blocks for p-2 =====
    // Build a^(2^223-1) = x223

    t = x88;
    for (var i = 0u; i < 88u; i = i + 1u) { t = fe_square(t); }
    t = fe_mul(t, x88);                        // a^(2^176-1)
    for (var i = 0u; i < 44u; i = i + 1u) { t = fe_square(t); }
    t = fe_mul(t, x44);                        // a^(2^220-1)
    for (var i = 0u; i < 3u; i = i + 1u) { t = fe_square(t); }
    t = fe_mul(t, x3);                         // a^(2^223-1)

    // bit 32 = 0, bits 31-10 = 22 ones
    for (var i = 0u; i < 23u; i = i + 1u) { t = fe_square(t); }
    t = fe_mul(t, x22);

    // bits 9-0 = 0000_10_11_01 (2-bit windows)
    t = fe_square(fe_square(fe_square(fe_square(t))));  // 4 zeros
    t = fe_square(fe_square(t));
    t = fe_mul(t, a2);                         // window 10 = a^2
    t = fe_square(fe_square(t));
    t = fe_mul(t, x2);                         // window 11 = a^3
    t = fe_square(fe_square(t));
    t = fe_mul(t, a);                          // window 01 = a^1

    return t;
}
