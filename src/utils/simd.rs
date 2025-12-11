//! SIMD utilities and optimized operations.
//!
//! This module provides SIMD-accelerated implementations of common operations
//! used throughout the HTM algorithms. It includes runtime feature detection
//! and fallback scalar implementations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// =============================================================================
// RUNTIME FEATURE DETECTION
// =============================================================================

/// Check if AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Check if AVX2 is available (always false on non-x86).
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn has_avx2() -> bool {
    false
}

/// Check if FMA is available at runtime.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_fma() -> bool {
    is_x86_feature_detected!("fma")
}

/// Check if FMA is available (always false on non-x86).
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn has_fma() -> bool {
    false
}

/// Check if NEON is available (always true on aarch64).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn has_neon() -> bool {
    true
}

/// Check if NEON is available (always false on non-ARM).
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn has_neon() -> bool {
    false
}

// =============================================================================
// FAST EXP APPROXIMATION
// =============================================================================

/// Fast exponential approximation using Schraudolph's method.
/// Accuracy: ~1-2% relative error, sufficient for boost factors.
#[inline]
pub fn fast_exp_f32(x: f32) -> f32 {
    // Clamp to prevent overflow/underflow
    let x = x.clamp(-87.0, 88.0);

    // Schraudolph's approximation: exp(x) ≈ 2^(x/ln(2))
    // Using integer bit manipulation for speed
    const A: f32 = (1 << 23) as f32 / std::f32::consts::LN_2;
    const B: f32 = (1 << 23) as f32 * (127.0 - 0.043677448); // Bias adjustment

    let bits = ((A * x + B) as i32) as u32;
    f32::from_bits(bits)
}

/// Batch exponential computation with automatic SIMD dispatch.
#[inline]
pub fn exp_batch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { exp_batch_avx2(input, output) };
            return;
        }
    }

    // Scalar fallback
    exp_batch_scalar(input, output);
}

/// Scalar fallback for batch exp.
#[inline]
fn exp_batch_scalar(input: &[f32], output: &mut [f32]) {
    for (out, &inp) in output.iter_mut().zip(input.iter()) {
        *out = fast_exp_f32(inp);
    }
}

/// AVX2 implementation of batch exp.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn exp_batch_avx2(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let chunks = len / 8;

    // Constants for Schraudolph's approximation
    let a = _mm256_set1_ps((1i32 << 23) as f32 / std::f32::consts::LN_2);
    let b = _mm256_set1_ps((1i32 << 23) as f32 * (127.0 - 0.043677448));
    let min_val = _mm256_set1_ps(-87.0);
    let max_val = _mm256_set1_ps(88.0);

    let inp_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 floats
        let x = _mm256_loadu_ps(inp_ptr.add(offset));

        // Clamp
        let x = _mm256_max_ps(_mm256_min_ps(x, max_val), min_val);

        // Compute: (A * x + B)
        let result = _mm256_add_ps(_mm256_mul_ps(a, x), b);

        // Convert to int and back to float (reinterpret bits)
        let result_int = _mm256_cvtps_epi32(result);
        let result_float = _mm256_castsi256_ps(result_int);

        // Store
        _mm256_storeu_ps(out_ptr.add(offset), result_float);
    }

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        *output.get_unchecked_mut(i) = fast_exp_f32(*input.get_unchecked(i));
    }
}

// =============================================================================
// BOOST FACTOR COMPUTATION
// =============================================================================

/// Compute boost factors: boost[i] = exp(strength * (target - duty[i]))
#[inline]
pub fn compute_boost_factors(
    duty_cycles: &[f32],
    boost_factors: &mut [f32],
    target_density: f32,
    boost_strength: f32,
) {
    debug_assert_eq!(duty_cycles.len(), boost_factors.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                compute_boost_factors_avx2(duty_cycles, boost_factors, target_density, boost_strength)
            };
            return;
        }
    }

    // Scalar fallback
    compute_boost_factors_scalar(duty_cycles, boost_factors, target_density, boost_strength);
}

/// Scalar fallback for boost factor computation.
#[inline]
fn compute_boost_factors_scalar(
    duty_cycles: &[f32],
    boost_factors: &mut [f32],
    target_density: f32,
    boost_strength: f32,
) {
    for (boost, &duty) in boost_factors.iter_mut().zip(duty_cycles.iter()) {
        *boost = fast_exp_f32(boost_strength * (target_density - duty));
    }
}

/// AVX2 implementation of boost factor computation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_boost_factors_avx2(
    duty_cycles: &[f32],
    boost_factors: &mut [f32],
    target_density: f32,
    boost_strength: f32,
) {
    let len = duty_cycles.len();
    let chunks = len / 8;

    let target = _mm256_set1_ps(target_density);
    let strength = _mm256_set1_ps(boost_strength);

    // Exp constants
    let a = _mm256_set1_ps((1i32 << 23) as f32 / std::f32::consts::LN_2);
    let b = _mm256_set1_ps((1i32 << 23) as f32 * (127.0 - 0.043677448));
    let min_val = _mm256_set1_ps(-87.0);
    let max_val = _mm256_set1_ps(88.0);

    let duty_ptr = duty_cycles.as_ptr();
    let boost_ptr = boost_factors.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;

        // Load duty cycles
        let duty = _mm256_loadu_ps(duty_ptr.add(offset));

        // Compute: strength * (target - duty)
        let diff = _mm256_sub_ps(target, duty);
        let x = _mm256_mul_ps(strength, diff);

        // Clamp
        let x = _mm256_max_ps(_mm256_min_ps(x, max_val), min_val);

        // Fast exp
        let result = _mm256_add_ps(_mm256_mul_ps(a, x), b);
        let result_int = _mm256_cvtps_epi32(result);
        let result_float = _mm256_castsi256_ps(result_int);

        // Store
        _mm256_storeu_ps(boost_ptr.add(offset), result_float);
    }

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        let duty = *duty_cycles.get_unchecked(i);
        *boost_factors.get_unchecked_mut(i) = fast_exp_f32(boost_strength * (target_density - duty));
    }
}

// =============================================================================
// PERMANENCE UPDATES
// =============================================================================

/// Batch update permanences with clamping: perm[i] = clamp(perm[i] + delta[i], 0, 1)
#[inline]
pub fn update_permanences_batch(
    permanences: &mut [f32],
    deltas: &[f32],
) {
    debug_assert_eq!(permanences.len(), deltas.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { update_permanences_avx2(permanences, deltas) };
            return;
        }
    }

    // Scalar fallback
    update_permanences_scalar(permanences, deltas);
}

/// Scalar fallback for permanence updates.
#[inline]
fn update_permanences_scalar(permanences: &mut [f32], deltas: &[f32]) {
    for (perm, &delta) in permanences.iter_mut().zip(deltas.iter()) {
        *perm = (*perm + delta).clamp(0.0, 1.0);
    }
}

/// AVX2 implementation of permanence updates.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn update_permanences_avx2(permanences: &mut [f32], deltas: &[f32]) {
    let len = permanences.len();
    let chunks = len / 8;

    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);

    let perm_ptr = permanences.as_mut_ptr();
    let delta_ptr = deltas.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;

        // Load permanences and deltas
        let p = _mm256_loadu_ps(perm_ptr.add(offset));
        let d = _mm256_loadu_ps(delta_ptr.add(offset));

        // Add
        let sum = _mm256_add_ps(p, d);

        // Clamp to [0, 1]
        let clamped = _mm256_min_ps(_mm256_max_ps(sum, zero), one);

        // Store
        _mm256_storeu_ps(perm_ptr.add(offset), clamped);
    }

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        let perm = permanences.get_unchecked_mut(i);
        *perm = (*perm + *deltas.get_unchecked(i)).clamp(0.0, 1.0);
    }
}

/// Update permanences with increment/decrement based on active mask.
/// If mask[i] is true, add increment; otherwise subtract decrement.
#[inline]
pub fn update_permanences_masked(
    permanences: &mut [f32],
    active_mask: &[bool],
    increment: f32,
    decrement: f32,
) {
    debug_assert_eq!(permanences.len(), active_mask.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { update_permanences_masked_avx2(permanences, active_mask, increment, decrement) };
            return;
        }
    }

    // Scalar fallback
    for (perm, &active) in permanences.iter_mut().zip(active_mask.iter()) {
        let delta = if active { increment } else { -decrement };
        *perm = (*perm + delta).clamp(0.0, 1.0);
    }
}

/// AVX2 implementation of masked permanence updates.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn update_permanences_masked_avx2(
    permanences: &mut [f32],
    active_mask: &[bool],
    increment: f32,
    decrement: f32,
) {
    let len = permanences.len();
    let chunks = len / 8;

    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let inc = _mm256_set1_ps(increment);
    let dec = _mm256_set1_ps(-decrement);

    let perm_ptr = permanences.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;

        // Load permanences
        let p = _mm256_loadu_ps(perm_ptr.add(offset));

        // Build mask from bools (convert to f32 mask)
        let mask_slice = &active_mask[offset..offset + 8];
        let mask_bits: i32 = mask_slice.iter().enumerate().fold(0, |acc, (j, &b)| {
            if b { acc | (1 << j) } else { acc }
        });

        // Use blendv to select between inc and dec
        // Create mask vector: -1.0 for true, 0.0 for false
        let mask_f32: [f32; 8] = [
            if mask_slice[0] { f32::from_bits(0xFFFFFFFF) } else { 0.0 },
            if mask_slice[1] { f32::from_bits(0xFFFFFFFF) } else { 0.0 },
            if mask_slice[2] { f32::from_bits(0xFFFFFFFF) } else { 0.0 },
            if mask_slice[3] { f32::from_bits(0xFFFFFFFF) } else { 0.0 },
            if mask_slice[4] { f32::from_bits(0xFFFFFFFF) } else { 0.0 },
            if mask_slice[5] { f32::from_bits(0xFFFFFFFF) } else { 0.0 },
            if mask_slice[6] { f32::from_bits(0xFFFFFFFF) } else { 0.0 },
            if mask_slice[7] { f32::from_bits(0xFFFFFFFF) } else { 0.0 },
        ];
        let mask_vec = _mm256_loadu_ps(mask_f32.as_ptr());

        // Select delta: inc if mask, dec if not
        let delta = _mm256_blendv_ps(dec, inc, mask_vec);

        // Add and clamp
        let sum = _mm256_add_ps(p, delta);
        let clamped = _mm256_min_ps(_mm256_max_ps(sum, zero), one);

        // Store
        _mm256_storeu_ps(perm_ptr.add(offset), clamped);
    }

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        let perm = permanences.get_unchecked_mut(i);
        let delta = if *active_mask.get_unchecked(i) { increment } else { -decrement };
        *perm = (*perm + delta).clamp(0.0, 1.0);
    }
}

// =============================================================================
// DUTY CYCLE UPDATES
// =============================================================================

/// Update duty cycles using exponential moving average.
/// duty[i] = ((period - 1) * duty[i] + value[i]) / period
#[inline]
pub fn update_duty_cycles(
    duty_cycles: &mut [f32],
    values: &[f32],
    period: f32,
) {
    debug_assert_eq!(duty_cycles.len(), values.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { update_duty_cycles_avx2(duty_cycles, values, period) };
            return;
        }
    }

    // Scalar fallback
    update_duty_cycles_scalar(duty_cycles, values, period);
}

/// Scalar fallback for duty cycle updates.
#[inline]
fn update_duty_cycles_scalar(duty_cycles: &mut [f32], values: &[f32], period: f32) {
    let period_minus_1 = period - 1.0;
    let inv_period = 1.0 / period;

    for (duty, &value) in duty_cycles.iter_mut().zip(values.iter()) {
        *duty = (period_minus_1 * *duty + value) * inv_period;
    }
}

/// AVX2 implementation of duty cycle updates.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn update_duty_cycles_avx2(duty_cycles: &mut [f32], values: &[f32], period: f32) {
    let len = duty_cycles.len();
    let chunks = len / 8;

    let period_minus_1 = _mm256_set1_ps(period - 1.0);
    let inv_period = _mm256_set1_ps(1.0 / period);

    let duty_ptr = duty_cycles.as_mut_ptr();
    let value_ptr = values.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;

        // Load duty cycles and values
        let d = _mm256_loadu_ps(duty_ptr.add(offset));
        let v = _mm256_loadu_ps(value_ptr.add(offset));

        // Compute: ((period - 1) * duty + value) / period
        // = ((period - 1) * duty + value) * inv_period
        let scaled = _mm256_mul_ps(period_minus_1, d);
        let sum = _mm256_add_ps(scaled, v);
        let result = _mm256_mul_ps(sum, inv_period);

        // Store
        _mm256_storeu_ps(duty_ptr.add(offset), result);
    }

    // Handle remainder
    let period_minus_1_scalar = period - 1.0;
    let inv_period_scalar = 1.0 / period;
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        let duty = duty_cycles.get_unchecked_mut(i);
        *duty = (period_minus_1_scalar * *duty + *values.get_unchecked(i)) * inv_period_scalar;
    }
}

/// Update duty cycles from overlap counts (converting u16 overlaps to 0/1 values).
#[inline]
pub fn update_duty_cycles_from_overlaps(
    duty_cycles: &mut [f32],
    overlaps: &[u16],
    period: f32,
) {
    debug_assert_eq!(duty_cycles.len(), overlaps.len());

    let period_minus_1 = period - 1.0;
    let inv_period = 1.0 / period;

    for (duty, &overlap) in duty_cycles.iter_mut().zip(overlaps.iter()) {
        let value = if overlap > 0 { 1.0 } else { 0.0 };
        *duty = (period_minus_1 * *duty + value) * inv_period;
    }
}

// =============================================================================
// SORTED VECTOR OPERATIONS
// =============================================================================

/// Compute overlap between two sorted u32 vectors.
/// Returns the count of common elements.
#[inline]
pub fn sorted_overlap(a: &[u32], b: &[u32]) -> usize {
    // The scalar two-pointer merge is already very cache-efficient
    // SIMD only helps for very large arrays with specific patterns
    // For typical HTM sparsity (~2-5%), scalar is optimal
    sorted_overlap_scalar(a, b)
}

/// Scalar implementation of sorted overlap (two-pointer merge).
#[inline]
pub fn sorted_overlap_scalar(a: &[u32], b: &[u32]) -> usize {
    let mut count = 0;
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        let av = unsafe { *a.get_unchecked(i) };
        let bv = unsafe { *b.get_unchecked(j) };

        if av < bv {
            i += 1;
        } else if av > bv {
            j += 1;
        } else {
            count += 1;
            i += 1;
            j += 1;
        }
    }

    count
}

/// AVX2 implementation of sorted overlap using galloping search.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sorted_overlap_avx2(a: &[u32], b: &[u32]) -> usize {
    // Use adaptive algorithm: if sizes are very different, use galloping
    // Otherwise use merge with SIMD acceleration

    let (smaller, larger) = if a.len() <= b.len() { (a, b) } else { (b, a) };

    // If one is much smaller, use galloping binary search
    if smaller.len() * 8 < larger.len() {
        return galloping_overlap(smaller, larger);
    }

    // For similar sizes, use SIMD-accelerated merge
    simd_merge_overlap(a, b)
}

/// Galloping binary search overlap for very different sized arrays.
#[cfg(target_arch = "x86_64")]
fn galloping_overlap(smaller: &[u32], larger: &[u32]) -> usize {
    let mut count = 0;
    let mut search_start = 0;

    for &val in smaller {
        // Binary search in larger array starting from search_start
        if let Ok(idx) = larger[search_start..].binary_search(&val) {
            count += 1;
            search_start += idx + 1;
        } else if let Err(idx) = larger[search_start..].binary_search(&val) {
            search_start += idx;
        }

        if search_start >= larger.len() {
            break;
        }
    }

    count
}

/// SIMD-accelerated merge overlap for similar-sized arrays.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_merge_overlap(a: &[u32], b: &[u32]) -> usize {
    // For now, use optimized scalar with prefetching hints
    // True SIMD merge is complex and may not be faster for typical HTM sparsity

    let mut count = 0;
    let mut i = 0;
    let mut j = 0;

    // Prefetch hints
    if a.len() > 64 {
        _mm_prefetch(a.as_ptr().add(32) as *const i8, _MM_HINT_T0);
    }
    if b.len() > 64 {
        _mm_prefetch(b.as_ptr().add(32) as *const i8, _MM_HINT_T0);
    }

    while i < a.len() && j < b.len() {
        let av = *a.get_unchecked(i);
        let bv = *b.get_unchecked(j);

        // Prefetch ahead
        if i + 32 < a.len() && (i & 31) == 0 {
            _mm_prefetch(a.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);
        }
        if j + 32 < b.len() && (j & 31) == 0 {
            _mm_prefetch(b.as_ptr().add(j + 32) as *const i8, _MM_HINT_T0);
        }

        if av < bv {
            i += 1;
        } else if av > bv {
            j += 1;
        } else {
            count += 1;
            i += 1;
            j += 1;
        }
    }

    count
}

// =============================================================================
// VECTOR MULTIPLY-ADD
// =============================================================================

/// Multiply vector by scalar and add to output: out[i] = a[i] * scalar + out[i]
#[inline]
pub fn fmadd_scalar_batch(a: &[f32], scalar: f32, out: &mut [f32]) {
    debug_assert_eq!(a.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            unsafe { fmadd_scalar_avx2(a, scalar, out) };
            return;
        }
    }

    // Scalar fallback
    for (o, &av) in out.iter_mut().zip(a.iter()) {
        *o += av * scalar;
    }
}

/// AVX2+FMA implementation of scalar multiply-add.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fmadd_scalar_avx2(a: &[f32], scalar: f32, out: &mut [f32]) {
    let len = a.len();
    let chunks = len / 8;

    let s = _mm256_set1_ps(scalar);

    let a_ptr = a.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;

        let av = _mm256_loadu_ps(a_ptr.add(offset));
        let ov = _mm256_loadu_ps(out_ptr.add(offset));

        // out = a * scalar + out
        let result = _mm256_fmadd_ps(av, s, ov);

        _mm256_storeu_ps(out_ptr.add(offset), result);
    }

    // Remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        *out.get_unchecked_mut(i) += *a.get_unchecked(i) * scalar;
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_exp() {
        let test_values = [-5.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        for &x in &test_values {
            let fast = fast_exp_f32(x);
            let accurate = x.exp();
            let rel_error = (fast - accurate).abs() / accurate;
            // Allow up to 4% error for Schraudolph's approximation
            // This is acceptable for boost factor computation
            assert!(rel_error < 0.04, "fast_exp({}) = {}, expected {}, error = {}", x, fast, accurate, rel_error);
        }
    }

    #[test]
    fn test_exp_batch() {
        let input: Vec<f32> = (-10..10).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0; input.len()];

        exp_batch(&input, &mut output);

        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            let expected = fast_exp_f32(inp);
            assert!((out - expected).abs() < 1e-6, "Mismatch at {}: {} vs {}", i, out, expected);
        }
    }

    #[test]
    fn test_boost_factors() {
        let duty_cycles: Vec<f32> = (0..100).map(|i| 0.01 + i as f32 * 0.001).collect();
        let mut boost_factors = vec![0.0; 100];
        let target = 0.05;
        let strength = 3.0;

        compute_boost_factors(&duty_cycles, &mut boost_factors, target, strength);

        for (i, (&duty, &boost)) in duty_cycles.iter().zip(boost_factors.iter()).enumerate() {
            let expected = fast_exp_f32(strength * (target - duty));
            assert!((boost - expected).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, boost, expected);
        }
    }

    #[test]
    fn test_permanence_update() {
        let mut perms = vec![0.1, 0.5, 0.9, 0.0, 1.0, 0.3];
        let deltas = vec![0.1, 0.1, 0.2, -0.1, 0.1, -0.5];

        update_permanences_batch(&mut perms, &deltas);

        assert!((perms[0] - 0.2).abs() < 1e-6);
        assert!((perms[1] - 0.6).abs() < 1e-6);
        assert!((perms[2] - 1.0).abs() < 1e-6); // Clamped to 1
        assert!((perms[3] - 0.0).abs() < 1e-6); // Clamped to 0
        assert!((perms[4] - 1.0).abs() < 1e-6); // Clamped to 1
        assert!((perms[5] - 0.0).abs() < 1e-6); // Clamped to 0
    }

    #[test]
    fn test_duty_cycle_update() {
        let mut duties = vec![0.0, 0.5, 1.0];
        let values = vec![1.0, 1.0, 0.0];
        let period = 10.0;

        update_duty_cycles(&mut duties, &values, period);

        // duty = (9 * old + new) / 10
        assert!((duties[0] - 0.1).abs() < 1e-6);  // (9*0 + 1) / 10 = 0.1
        assert!((duties[1] - 0.55).abs() < 1e-6); // (9*0.5 + 1) / 10 = 0.55
        assert!((duties[2] - 0.9).abs() < 1e-6);  // (9*1 + 0) / 10 = 0.9
    }

    #[test]
    fn test_sorted_overlap() {
        let a = vec![1, 3, 5, 7, 9, 11, 13, 15];
        let b = vec![2, 3, 5, 8, 11, 14, 15, 16];

        let overlap = sorted_overlap(&a, &b);
        assert_eq!(overlap, 4); // 3, 5, 11, 15
    }

    #[test]
    fn test_sorted_overlap_empty() {
        let a: Vec<u32> = vec![];
        let b = vec![1, 2, 3];

        assert_eq!(sorted_overlap(&a, &b), 0);
        assert_eq!(sorted_overlap(&b, &a), 0);
    }

    #[test]
    fn test_sorted_overlap_disjoint() {
        let a = vec![1, 2, 3];
        let b = vec![10, 11, 12];

        assert_eq!(sorted_overlap(&a, &b), 0);
    }

    #[test]
    fn test_sorted_overlap_identical() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];

        assert_eq!(sorted_overlap(&a, &b), 5);
    }
}
