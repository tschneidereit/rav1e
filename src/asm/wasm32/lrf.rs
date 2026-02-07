// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! WASM SIMD128 accelerated loop restoration filter functions.

use crate::cpu_features::CpuFeatureLevel;
use crate::frame::PlaneSlice;
use crate::lrf::rust;
use crate::lrf::{
  SGRPROJ_MTABLE_BITS, SGRPROJ_RECIP_BITS, SGRPROJ_SGR_BITS,
};
use crate::util::Pixel;
use std::arch::wasm32::*;

/// Compute sum from integral image using SIMD for 4 consecutive x positions
#[inline(always)]
unsafe fn get_integral_square_simd4(
  iimg: &[u32], stride: usize, x: usize, y: usize, size: usize,
) -> v128 {
  // For 4 consecutive x positions, we need:
  // iimg[y*stride + x], iimg[y*stride + x+1], iimg[y*stride + x+2], iimg[y*stride + x+3]
  // etc. for top_left, top_right, bottom_left, bottom_right
  let y_offset = y * stride;
  let y_size_offset = (y + size) * stride;

  // Load 4 top_left values
  let top_left = v128_load(iimg.as_ptr().add(y_offset + x) as *const v128);

  // Load 4 top_right values (offset by size)
  let top_right = v128_load(iimg.as_ptr().add(y_offset + x + size) as *const v128);

  // Load 4 bottom_left values
  let bottom_left =
    v128_load(iimg.as_ptr().add(y_size_offset + x) as *const v128);

  // Load 4 bottom_right values
  let bottom_right =
    v128_load(iimg.as_ptr().add(y_size_offset + x + size) as *const v128);

  // result = top_left + bottom_right - bottom_left - top_right
  // Using wrapping arithmetic (all u32)
  let sum1 = i32x4_add(top_left, bottom_right);
  let sum2 = i32x4_add(bottom_left, top_right);
  i32x4_sub(sum1, sum2)
}

/// SIMD version of sgrproj_sum_finish for 4 values at once
/// Returns (a[4], b[4]) as two v128 vectors
#[inline(always)]
unsafe fn sgrproj_sum_finish_simd<const BD: usize>(
  ssq: v128, sum: v128, n: u32, one_over_n: u32, s: u32,
) -> (v128, v128) {
  let bdm8 = BD - 8;

  // Scale ssq: (ssq + (1 << (2 * bdm8) >> 1)) >> (2 * bdm8)
  let ssq_round = i32x4_splat((1i32 << (2 * bdm8)) >> 1);
  let scaled_ssq = u32x4_shr(i32x4_add(ssq, ssq_round), (2 * bdm8) as u32);

  // Scale sum: (sum + (1 << bdm8 >> 1)) >> bdm8
  let sum_round = i32x4_splat((1i32 << bdm8) >> 1);
  let scaled_sum = u32x4_shr(i32x4_add(sum, sum_round), bdm8 as u32);

  // p = (scaled_ssq * n).saturating_sub(scaled_sum * scaled_sum)
  let n_vec = i32x4_splat(n as i32);
  let ssq_n = i32x4_mul(scaled_ssq, n_vec);
  let sum_sq = i32x4_mul(scaled_sum, scaled_sum);
  let p = i32x4_max(i32x4_sub(ssq_n, sum_sq), i32x4_splat(0)); // saturating_sub

  // z = (p * s + (1 << SGRPROJ_MTABLE_BITS >> 1)) >> SGRPROJ_MTABLE_BITS
  let s_vec = i32x4_splat(s as i32);
  let mtable_round = i32x4_splat((1i32 << SGRPROJ_MTABLE_BITS) >> 1);
  let ps = i32x4_mul(p, s_vec);
  let z = u32x4_shr(i32x4_add(ps, mtable_round), SGRPROJ_MTABLE_BITS as u32);

  // For the a computation, we need to handle the conditional logic:
  // a = if z >= 255 { 256 } else if z == 0 { 1 } else { ((z << SGRPROJ_SGR_BITS) + z / 2) / (z + 1) }
  // This is complex to vectorize, so process scalarly for now
  let z0 = u32x4_extract_lane::<0>(z);
  let z1 = u32x4_extract_lane::<1>(z);
  let z2 = u32x4_extract_lane::<2>(z);
  let z3 = u32x4_extract_lane::<3>(z);

  let compute_a = |z: u32| -> u32 {
    if z >= 255 {
      256
    } else if z == 0 {
      1
    } else {
      ((z << SGRPROJ_SGR_BITS) + z / 2) / (z + 1)
    }
  };

  let a0 = compute_a(z0);
  let a1 = compute_a(z1);
  let a2 = compute_a(z2);
  let a3 = compute_a(z3);

  let a = u32x4(a0, a1, a2, a3);

  // b = ((1 << SGRPROJ_SGR_BITS) - a) * sum * one_over_n

  // Need to compute (1-a) * sum * one_over_n, which can overflow i32
  // Do scalar computation for safety
  let sum0 = u32x4_extract_lane::<0>(sum);
  let sum1 = u32x4_extract_lane::<1>(sum);
  let sum2 = u32x4_extract_lane::<2>(sum);
  let sum3 = u32x4_extract_lane::<3>(sum);

  let compute_b = |a: u32, sum: u32| -> u32 {
    let b = ((1u64 << SGRPROJ_SGR_BITS) - a as u64) * sum as u64 * one_over_n as u64;
    ((b + (1 << (SGRPROJ_RECIP_BITS - 1))) >> SGRPROJ_RECIP_BITS) as u32
  };

  let b0 = compute_b(a0, sum0);
  let b1 = compute_b(a1, sum1);
  let b2 = compute_b(a2, sum2);
  let b3 = compute_b(a3, sum3);

  let b = u32x4(b0, b1, b2, b3);

  (a, b)
}

/// SIMD implementation of sgrproj_box_ab for radius r
#[inline(always)]
unsafe fn sgrproj_box_ab_simd<const BD: usize>(
  r: usize, af: &mut [u32], bf: &mut [u32], iimg: &[u32], iimg_sq: &[u32],
  iimg_stride: usize, start_x: usize, y: usize, stripe_w: usize, s: u32,
) {
  let d: usize = r * 2 + 1;
  let n: usize = d * d;
  let one_over_n = if r == 1 { 455u32 } else { 164u32 };

  let mut x = start_x;

  // Process 4 positions at a time with SIMD
  while x + 4 <= stripe_w + 2 {
    // Check bounds before SIMD load
    if (y + d) * iimg_stride + x + d + 4 <= iimg.len()
      && (y + d) * iimg_stride + x + d + 4 <= iimg_sq.len()
    {
      let sum = get_integral_square_simd4(iimg, iimg_stride, x, y, d);
      let ssq = get_integral_square_simd4(iimg_sq, iimg_stride, x, y, d);
      let (a, b) =
        sgrproj_sum_finish_simd::<BD>(ssq, sum, n as u32, one_over_n, s);

      // Store results
      v128_store(af.as_mut_ptr().add(x) as *mut v128, a);
      v128_store(bf.as_mut_ptr().add(x) as *mut v128, b);
    } else {
      // Fall back to scalar for edge cases
      for xi in x..x + 4 {
        if xi < stripe_w + 2 {
          let sum = get_integral_square_scalar(iimg, iimg_stride, xi, y, d);
          let ssq = get_integral_square_scalar(iimg_sq, iimg_stride, xi, y, d);
          let (a, b) =
            sgrproj_sum_finish_scalar::<BD>(ssq, sum, n as u32, one_over_n, s);
          af[xi] = a;
          bf[xi] = b;
        }
      }
    }
    x += 4;
  }

  // Handle remaining positions
  while x < stripe_w + 2 {
    let sum = get_integral_square_scalar(iimg, iimg_stride, x, y, d);
    let ssq = get_integral_square_scalar(iimg_sq, iimg_stride, x, y, d);
    let (a, b) =
      sgrproj_sum_finish_scalar::<BD>(ssq, sum, n as u32, one_over_n, s);
    af[x] = a;
    bf[x] = b;
    x += 1;
  }
}

#[inline(always)]
fn get_integral_square_scalar(
  iimg: &[u32], stride: usize, x: usize, y: usize, size: usize,
) -> u32 {
  let top_left = iimg[y * stride + x];
  let top_right = iimg[y * stride + x + size];
  let bottom_left = iimg[(y + size) * stride + x];
  let bottom_right = iimg[(y + size) * stride + x + size];
  top_left
    .wrapping_add(bottom_right)
    .wrapping_sub(bottom_left)
    .wrapping_sub(top_right)
}

#[inline(always)]
fn sgrproj_sum_finish_scalar<const BD: usize>(
  ssq: u32, sum: u32, n: u32, one_over_n: u32, s: u32,
) -> (u32, u32) {
  let bdm8 = BD - 8;
  let scaled_ssq = (ssq + (1 << (2 * bdm8) >> 1)) >> (2 * bdm8);
  let scaled_sum = (sum + (1 << bdm8 >> 1)) >> bdm8;
  let p = (scaled_ssq * n).saturating_sub(scaled_sum * scaled_sum);
  let z = (p * s + (1 << SGRPROJ_MTABLE_BITS >> 1)) >> SGRPROJ_MTABLE_BITS;
  let a = if z >= 255 {
    256
  } else if z == 0 {
    1
  } else {
    ((z << SGRPROJ_SGR_BITS) + z / 2) / (z + 1)
  };
  let b = ((1u64 << SGRPROJ_SGR_BITS) - a as u64) * sum as u64 * one_over_n as u64;
  (a, ((b + (1 << (SGRPROJ_RECIP_BITS - 1))) >> SGRPROJ_RECIP_BITS) as u32)
}

// Public API functions

#[inline]
pub fn sgrproj_box_ab_r1<const BD: usize>(
  af: &mut [u32], bf: &mut [u32], iimg: &[u32], iimg_sq: &[u32],
  iimg_stride: usize, y: usize, stripe_w: usize, s: u32, cpu: CpuFeatureLevel,
) {
  // SIMD disabled - overhead exceeds benefit for this workload
  rust::sgrproj_box_ab_r1::<BD>(
    af,
    bf,
    iimg,
    iimg_sq,
    iimg_stride,
    y,
    stripe_w,
    s,
    cpu,
  );
}

#[inline]
pub fn sgrproj_box_ab_r2<const BD: usize>(
  af: &mut [u32], bf: &mut [u32], iimg: &[u32], iimg_sq: &[u32],
  iimg_stride: usize, y: usize, stripe_w: usize, s: u32, cpu: CpuFeatureLevel,
) {
  // SIMD disabled - overhead exceeds benefit for this workload
  rust::sgrproj_box_ab_r2::<BD>(
    af,
    bf,
    iimg,
    iimg_sq,
    iimg_stride,
    y,
    stripe_w,
    s,
    cpu,
  );
}

// The sgrproj_box_f functions don't benefit much from SIMD since they're
// mostly memory copies with shifts. Fall back to Rust implementations.

#[inline]
pub fn sgrproj_box_f_r0<T: Pixel>(
  f: &mut [u32], y: usize, w: usize, cdeffed: &PlaneSlice<T>,
  cpu: CpuFeatureLevel,
) {
  rust::sgrproj_box_f_r0(f, y, w, cdeffed, cpu);
}

#[inline]
pub fn sgrproj_box_f_r1<T: Pixel>(
  af: &[&[u32]; 3], bf: &[&[u32]; 3], f: &mut [u32], y: usize, w: usize,
  cdeffed: &PlaneSlice<T>, cpu: CpuFeatureLevel,
) {
  rust::sgrproj_box_f_r1(af, bf, f, y, w, cdeffed, cpu);
}

#[inline]
pub fn sgrproj_box_f_r2<T: Pixel>(
  af: &[&[u32]; 2], bf: &[&[u32]; 2], f0: &mut [u32], f1: &mut [u32],
  y: usize, w: usize, cdeffed: &PlaneSlice<T>, cpu: CpuFeatureLevel,
) {
  rust::sgrproj_box_f_r2(af, bf, f0, f1, y, w, cdeffed, cpu);
}
