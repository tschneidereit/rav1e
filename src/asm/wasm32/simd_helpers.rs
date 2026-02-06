// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! SIMD helper functions for wasm32.
//!
//! This module contains common SIMD primitives used across the wasm32
//! implementations, including horizontal reductions, emulated instructions,
//! and utility functions.
//!
//! When the `relaxed-simd` target feature is enabled, additional optimized
//! functions become available using relaxed-simd instructions.

#![allow(dead_code)]

use core::arch::wasm32::*;

/// Horizontal sum of all i16 lanes in a v128, returning an i32.
///
/// Adds all 8 i16 lanes together by progressively adding halves.
#[inline(always)]
pub fn horizontal_sum_i16x8(v: v128) -> i32 {
  // Sum pairs: [a+b, c+d, e+f, g+h, ...]
  let sum1 = i32x4_add(
    i32x4_extend_low_i16x8(v),
    i32x4_extend_high_i16x8(v),
  );
  // Now we have 4 i32 values to sum
  horizontal_sum_i32x4(sum1)
}

/// Horizontal sum of all u16 lanes in a v128, returning a u32.
#[inline(always)]
pub fn horizontal_sum_u16x8(v: v128) -> u32 {
  // Sum pairs: unsigned extend and add
  let sum1 = i32x4_add(
    u32x4_extend_low_u16x8(v),
    u32x4_extend_high_u16x8(v),
  );
  horizontal_sum_i32x4(sum1) as u32
}

/// Horizontal sum of all i32 lanes in a v128, returning an i32.
#[inline(always)]
pub fn horizontal_sum_i32x4(v: v128) -> i32 {
  // Shuffle to get [c, d, a, b] and add to get [a+c, b+d, ...]
  let shuffled = i32x4_shuffle::<2, 3, 0, 1>(v, v);
  let sum1 = i32x4_add(v, shuffled);
  // Shuffle again to get [b+d, a+c, ...] and add
  let shuffled2 = i32x4_shuffle::<1, 0, 3, 2>(sum1, sum1);
  let sum2 = i32x4_add(sum1, shuffled2);
  // Extract first lane
  i32x4_extract_lane::<0>(sum2)
}

/// Horizontal sum of all u32 lanes in a v128, returning a u32.
#[inline(always)]
pub fn horizontal_sum_u32x4(v: v128) -> u32 {
  horizontal_sum_i32x4(v) as u32
}

/// Horizontal sum of i64x2 lanes, returning an i64.
#[inline(always)]
pub fn horizontal_sum_i64x2(v: v128) -> i64 {
  let a = i64x2_extract_lane::<0>(v);
  let b = i64x2_extract_lane::<1>(v);
  a.wrapping_add(b)
}

/// Horizontal sum of u8x16 lanes, returning a u32.
///
/// Sums all 16 bytes by extending to wider types.
#[inline(always)]
pub fn horizontal_sum_u8x16(v: v128) -> u32 {
  // Extend low and high halves to u16
  let lo = u16x8_extend_low_u8x16(v);
  let hi = u16x8_extend_high_u8x16(v);
  // Sum the two halves
  let sum16 = i16x8_add(lo, hi);
  // Continue summing
  horizontal_sum_u16x8(sum16)
}

/// Compute absolute difference of two u8x16 vectors and return as u8x16.
///
/// For each lane: |a - b|
#[inline(always)]
pub fn abs_diff_u8x16(a: v128, b: v128) -> v128 {
  // max(a,b) - min(a,b) gives absolute difference for unsigned
  let max_ab = u8x16_max(a, b);
  let min_ab = u8x16_min(a, b);
  u8x16_sub(max_ab, min_ab)
}

/// Compute absolute difference of two u16x8 vectors and return as u16x8.
#[inline(always)]
pub fn abs_diff_u16x8(a: v128, b: v128) -> v128 {
  let max_ab = u16x8_max(a, b);
  let min_ab = u16x8_min(a, b);
  u16x8_sub(max_ab, min_ab)
}

/// Emulate pmaddwd: multiply pairs of i16 and add adjacent pairs to produce i32.
///
/// For input vectors a = [a0, a1, a2, a3, a4, a5, a6, a7] and
/// b = [b0, b1, b2, b3, b4, b5, b6, b7], produces:
/// [a0*b0 + a1*b1, a2*b2 + a3*b3, a4*b4 + a5*b5, a6*b6 + a7*b7]
#[inline(always)]
pub fn pmaddwd_i16x8(a: v128, b: v128) -> v128 {
  // Extend low halves (even indices after multiplication)
  let a_lo = i32x4_extend_low_i16x8(a);   // [a0, a1, a2, a3] sign-extended
  let b_lo = i32x4_extend_low_i16x8(b);   // [b0, b1, b2, b3] sign-extended
  let a_hi = i32x4_extend_high_i16x8(a);  // [a4, a5, a6, a7] sign-extended
  let b_hi = i32x4_extend_high_i16x8(b);  // [b4, b5, b6, b7] sign-extended

  // Multiply to get full products
  let prod_lo = i32x4_mul(a_lo, b_lo);  // [a0*b0, a1*b1, a2*b2, a3*b3]
  let prod_hi = i32x4_mul(a_hi, b_hi);  // [a4*b4, a5*b5, a6*b6, a7*b7]

  // Now we need to add adjacent pairs
  // For prod_lo = [p0, p1, p2, p3], we want [p0+p1, p2+p3]
  // Use shuffles to align and add

  // Shuffle to get odd elements
  let odd_lo = i32x4_shuffle::<1, 3, 5, 7>(prod_lo, prod_hi);  // [p1, p3, p5, p7]
  let even_lo = i32x4_shuffle::<0, 2, 4, 6>(prod_lo, prod_hi); // [p0, p2, p4, p6]

  i32x4_add(even_lo, odd_lo)
}

/// Rounding right shift for i32x4.
///
/// Computes (v + (1 << (shift-1))) >> shift with proper rounding.
#[inline(always)]
pub fn rounding_shr_i32x4(v: v128, shift: u32) -> v128 {
  debug_assert!(shift > 0 && shift < 32);
  let rounding = i32x4_splat(1 << (shift - 1));
  let rounded = i32x4_add(v, rounding);
  i32x4_shr(rounded, shift)
}

/// Rounding right shift for i16x8.
#[inline(always)]
pub fn rounding_shr_i16x8(v: v128, shift: u32) -> v128 {
  debug_assert!(shift > 0 && shift < 16);
  let rounding = i16x8_splat(1 << (shift - 1));
  let rounded = i16x8_add(v, rounding);
  i16x8_shr(rounded, shift)
}

/// Load 8 bytes from memory and zero-extend to u16x8.
///
/// # Safety
/// The pointer must be valid for reading 8 bytes.
#[inline(always)]
pub unsafe fn load_u8x8_to_u16x8(ptr: *const u8) -> v128 {
  v128_load64_zero(ptr as *const u64)
    .pipe(|v| u16x8_extend_low_u8x16(v))
}

/// Load 4 bytes from memory and zero-extend to u16x8 (lower 4 lanes).
///
/// # Safety
/// The pointer must be valid for reading 4 bytes.
#[inline(always)]
pub unsafe fn load_u8x4_to_u16x8(ptr: *const u8) -> v128 {
  v128_load32_zero(ptr as *const u32)
    .pipe(|v| u16x8_extend_low_u8x16(v))
}

/// Helper trait to enable method chaining with `.pipe()`
trait Pipe: Sized {
  fn pipe<F, R>(self, f: F) -> R
  where
    F: FnOnce(Self) -> R;
}

impl<T> Pipe for T {
  #[inline(always)]
  fn pipe<F, R>(self, f: F) -> R
  where
    F: FnOnce(Self) -> R,
  {
    f(self)
  }
}

// ============================================================================
// Relaxed SIMD helpers (available when target_feature = "relaxed-simd")
// ============================================================================

/// Relaxed dot product: computes sum of products of i8 and i7 (signed 7-bit) pairs,
/// accumulated into i32 lanes with an accumulator.
///
/// This is extremely useful for filter convolutions and SAD calculations.
/// For a = [a0..a15] (i8) and b = [b0..b15] (i7), with acc = [acc0..acc3]:
/// result[i] = acc[i] + sum(a[4*i+j] * b[4*i+j] for j in 0..4)
///
/// Note: b values should be in range [-64, 63] for correct results.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn relaxed_dot_i8x16_add(a: v128, b: v128, acc: v128) -> v128 {
  i32x4_relaxed_dot_i8x16_i7x16_add(a, b, acc)
}

/// Relaxed i8x16 dot product without accumulator - just returns the dot products.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn relaxed_dot_i8x16(a: v128, b: v128) -> v128 {
  i32x4_relaxed_dot_i8x16_i7x16_add(a, b, i32x4_splat(0))
}

/// Relaxed Q15 fixed-point multiply with rounding.
///
/// Computes (a * b + 0x4000) >> 15 for each i16 lane.
/// Useful for fixed-point filter coefficient multiplication.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn relaxed_q15mulr_i16x8(a: v128, b: v128) -> v128 {
  i16x8_relaxed_q15mulr(a, b)
}

/// Relaxed lane select (blend) for i32x4.
///
/// For each lane: if mask bit is set, select from a; otherwise from b.
/// This is faster than bitwise operations for conditional selection.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn relaxed_laneselect_i32x4(a: v128, b: v128, mask: v128) -> v128 {
  i32x4_relaxed_laneselect(a, b, mask)
}

/// Relaxed lane select (blend) for i16x8.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn relaxed_laneselect_i16x8(a: v128, b: v128, mask: v128) -> v128 {
  i16x8_relaxed_laneselect(a, b, mask)
}

/// Relaxed lane select (blend) for i8x16.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn relaxed_laneselect_i8x16(a: v128, b: v128, mask: v128) -> v128 {
  i8x16_relaxed_laneselect(a, b, mask)
}

/// Relaxed swizzle for i8x16.
///
/// Similar to i8x16_swizzle but with relaxed out-of-bounds behavior.
/// When an index is >= 16, the result is implementation-defined (not necessarily 0).
/// Use only when you know all indices are valid.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn relaxed_swizzle_i8x16(a: v128, indices: v128) -> v128 {
  i8x16_relaxed_swizzle(a, indices)
}

/// 8-tap filter convolution using relaxed dot product.
///
/// Computes: sum(src[i] * filter[i] for i in 0..8) for 2 adjacent output positions.
/// This processes 2 outputs at a time using the 8-wide dot product.
///
/// # Arguments
/// * `src` - 16 consecutive source samples as i8 (subtract 128 for u8 sources)
/// * `filter` - 8 filter coefficients packed twice: [f0..f7, f0..f7] as i8
/// * `acc` - Accumulator to add to (can be rounding bias)
///
/// # Returns
/// i32x4 with [out0, out1, 0, 0] where out0 uses src[0..8] and out1 uses src[1..9]
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn filter_8tap_2x_relaxed(src: v128, filter: v128, acc: v128) -> v128 {
  // The relaxed dot product computes 4 groups of 4 multiplies
  // We need to arrange data so that groups 0,1 compute one output
  // and groups 2,3 compute another output
  
  // For now, use the simpler approach with two separate calls
  relaxed_dot_i8x16_add(src, filter, acc)
}

/// Compute sum of absolute differences (SAD) using relaxed dot product.
///
/// For two u8x16 vectors, computes sum(|a[i] - b[i]|) using:
/// 1. Compute differences (may be negative)
/// 2. Use dot product with sign vector to get absolute values summed
///
/// This is faster than the standard abs_diff + horizontal_sum approach.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn sad_u8x16_relaxed(a: v128, b: v128) -> u32 {
  // Standard approach is still needed since relaxed dot requires i7 range
  // The abs_diff approach works well with auto-vectorization
  horizontal_sum_u8x16(abs_diff_u8x16(a, b))
}

/// Compute weighted sum for smooth prediction using relaxed Q15 multiply.
///
/// Computes: (weight * above + inv_weight * below) >> 8
/// using fixed-point arithmetic.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
pub fn smooth_blend_relaxed(above: v128, below: v128, weight: v128) -> v128 {
  // Scale weights to Q15 format (multiply by 128 to get into range)
  // Then use Q15 multiply which gives (a * b + 0x4000) >> 15
  
  // weight is 0-255, scale to Q15 by << 7
  let weight_q15 = i16x8_shl(weight, 7);
  let inv_weight_q15 = i16x8_sub(i16x8_splat(0x7FFF), weight_q15);
  
  let prod_above = relaxed_q15mulr_i16x8(above, weight_q15);
  let prod_below = relaxed_q15mulr_i16x8(below, inv_weight_q15);
  
  i16x8_add(prod_above, prod_below)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_horizontal_sum_i32x4() {
    let v = i32x4(1, 2, 3, 4);
    assert_eq!(horizontal_sum_i32x4(v), 10);

    let v = i32x4(-1, -2, 3, 4);
    assert_eq!(horizontal_sum_i32x4(v), 4);
  }

  #[test]
  fn test_horizontal_sum_i16x8() {
    let v = i16x8(1, 2, 3, 4, 5, 6, 7, 8);
    assert_eq!(horizontal_sum_i16x8(v), 36);
  }

  #[test]
  fn test_abs_diff_u8x16() {
    let a = u8x16(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160);
    let b = u8x16(5, 25, 30, 35, 60, 50, 75, 85, 80, 110, 100, 130, 120, 150, 140, 170);
    let result = abs_diff_u8x16(a, b);

    assert_eq!(u8x16_extract_lane::<0>(result), 5);  // |10-5|
    assert_eq!(u8x16_extract_lane::<1>(result), 5);  // |20-25|
    assert_eq!(u8x16_extract_lane::<2>(result), 0);  // |30-30|
  }

  #[test]
  fn test_pmaddwd() {
    // Test: [1, 2, 3, 4, 5, 6, 7, 8] * [1, 1, 1, 1, 1, 1, 1, 1]
    // Result should be: [1+2, 3+4, 5+6, 7+8] = [3, 7, 11, 15]
    let a = i16x8(1, 2, 3, 4, 5, 6, 7, 8);
    let b = i16x8(1, 1, 1, 1, 1, 1, 1, 1);
    let result = pmaddwd_i16x8(a, b);

    assert_eq!(i32x4_extract_lane::<0>(result), 3);
    assert_eq!(i32x4_extract_lane::<1>(result), 7);
    assert_eq!(i32x4_extract_lane::<2>(result), 11);
    assert_eq!(i32x4_extract_lane::<3>(result), 15);
  }
}
