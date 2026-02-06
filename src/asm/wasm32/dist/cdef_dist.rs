// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! CDEF distance kernel functions for wasm32 SIMD.

use crate::activity::apply_ssim_boost;
use crate::cpu_features::CpuFeatureLevel;
use crate::dist::*;
use crate::tiling::PlaneRegion;
use crate::util::Pixel;
use crate::util::PixelType;

type CdefDistKernelFn = unsafe fn(
  src: *const u8,
  src_stride: isize,
  dst: *const u8,
  dst_stride: isize,
  w: usize,
  h: usize,
) -> (u32, u32, u32);

type CdefDistKernelHbdFn = unsafe fn(
  src: *const u16,
  src_stride: isize,
  dst: *const u16,
  dst_stride: isize,
  w: usize,
  h: usize,
) -> (u32, u32, u32);

/// Compute CDEF distortion kernel.
///
/// # Panics
///
/// - If in `check_asm` mode, panics on mismatch between SIMD and Rust results.
#[allow(clippy::let_and_return)]
pub fn cdef_dist_kernel<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, w: usize, h: usize,
  bit_depth: usize, cpu: CpuFeatureLevel,
) -> u32 {
  debug_assert!(src.plane_cfg.xdec == 0);
  debug_assert!(src.plane_cfg.ydec == 0);
  debug_assert!(dst.plane_cfg.xdec == 0);
  debug_assert!(dst.plane_cfg.ydec == 0);

  // Limit kernel to 8x8
  debug_assert!(w <= 8);
  debug_assert!(h <= 8);

  let call_rust =
    || -> u32 { rust::cdef_dist_kernel(dst, src, w, h, bit_depth, cpu) };

  #[cfg(feature = "check_asm")]
  let ref_dist = call_rust();

  let (svar, dvar, sse) = match T::type_enum() {
    PixelType::U8 => {
      if let Some(func) =
        CDEF_DIST_KERNEL_FNS[cpu.as_index()][kernel_fn_index(w, h)]
      {
        // SAFETY: Pointers are valid
        unsafe {
          func(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
            w,
            h,
          )
        }
      } else {
        return call_rust();
      }
    }
    PixelType::U16 => {
      if let Some(func) =
        CDEF_DIST_KERNEL_HBD_FNS[cpu.as_index()][kernel_fn_index(w, h)]
      {
        // SAFETY: Pointers are valid
        unsafe {
          func(
            src.data_ptr() as *const _,
            T::to_asm_stride(src.plane_cfg.stride),
            dst.data_ptr() as *const _,
            T::to_asm_stride(dst.plane_cfg.stride),
            w,
            h,
          )
        }
      } else {
        return call_rust();
      }
    }
  };

  let dist = apply_ssim_boost(sse, svar, dvar, bit_depth);

  #[cfg(feature = "check_asm")]
  assert_eq!(
    dist, ref_dist,
    "CDEF Distortion {}x{}: SIMD doesn't match reference code.",
    w, h
  );

  dist
}

/// Store functions in a 8x8 grid. Most will be empty.
const CDEF_DIST_KERNEL_FNS_LENGTH: usize = 8 * 8;

const fn kernel_fn_index(w: usize, h: usize) -> usize {
  ((w - 1) << 3) | (h - 1)
}

/// Number of bits of precision used in AREA_DIVISORS
const AREA_DIVISOR_BITS: u8 = 14;

/// Lookup table for 2^AREA_DIVISOR_BITS / (1 + x)
#[rustfmt::skip]
const AREA_DIVISORS: [u16; 64] = [
  16384, 8192, 5461, 4096, 3277, 2731, 2341, 2048, 1820, 1638, 1489, 1365,
   1260, 1170, 1092, 1024,  964,  910,  862,  819,  780,  745,  712,  683,
    655,  630,  607,  585,  565,  546,  529,  512,  496,  482,  468,  455,
    443,  431,  420,  410,  400,  390,  381,  372,  364,  356,  349,  341,
    334,  328,  321,  315,  309,  303,  298,  293,  287,  282,  278,  273,
    269,  264,  260,  256,
];

/// CDEF distortion kernel implementation for 8-bit pixels.
///
/// Returns (svar, dvar, sse) tuple.
#[inline(always)]
unsafe fn cdef_dist_kernel_simd128(
  src: *const u8, src_stride: isize, dst: *const u8, dst_stride: isize,
  w: usize, h: usize,
) -> (u32, u32, u32) {
  use core::arch::wasm32::*;

  // Process rows, accumulating in vector registers
  let mut sum_s_vec = i32x4_splat(0);
  let mut sum_d_vec = i32x4_splat(0);
  let mut sum_s2_vec = i32x4_splat(0);
  let mut sum_d2_vec = i32x4_splat(0);
  let mut sum_sd_vec = i32x4_splat(0);

  let mut src_ptr = src;
  let mut dst_ptr = dst;

  // For widths of 8, we can process a full row with one 128-bit load (8 bytes)
  // For widths of 4, we can process a row with partial load
  if w == 8 {
    for _y in 0..h {
      // Load 8 bytes
      let s_raw = v128_load64_zero(src_ptr as *const u64);
      let d_raw = v128_load64_zero(dst_ptr as *const u64);

      // Extend to i16 (low 8 bytes)
      let s_16 = i16x8_extend_low_u8x16(s_raw);
      let d_16 = i16x8_extend_low_u8x16(d_raw);

      // Extend to i32 for accumulation
      let s_lo = i32x4_extend_low_i16x8(s_16);
      let s_hi = i32x4_extend_high_i16x8(s_16);
      let d_lo = i32x4_extend_low_i16x8(d_16);
      let d_hi = i32x4_extend_high_i16x8(d_16);

      // Accumulate sums
      sum_s_vec = i32x4_add(sum_s_vec, i32x4_add(s_lo, s_hi));
      sum_d_vec = i32x4_add(sum_d_vec, i32x4_add(d_lo, d_hi));

      // Accumulate squared sums
      sum_s2_vec = i32x4_add(sum_s2_vec, i32x4_add(i32x4_mul(s_lo, s_lo), i32x4_mul(s_hi, s_hi)));
      sum_d2_vec = i32x4_add(sum_d2_vec, i32x4_add(i32x4_mul(d_lo, d_lo), i32x4_mul(d_hi, d_hi)));
      sum_sd_vec = i32x4_add(sum_sd_vec, i32x4_add(i32x4_mul(s_lo, d_lo), i32x4_mul(s_hi, d_hi)));

      src_ptr = src_ptr.offset(src_stride);
      dst_ptr = dst_ptr.offset(dst_stride);
    }
  } else if w == 4 {
    for _y in 0..h {
      // Load 4 bytes as i32x4
      let s = i32x4(
        *src_ptr as i32, *src_ptr.add(1) as i32,
        *src_ptr.add(2) as i32, *src_ptr.add(3) as i32
      );
      let d = i32x4(
        *dst_ptr as i32, *dst_ptr.add(1) as i32,
        *dst_ptr.add(2) as i32, *dst_ptr.add(3) as i32
      );

      sum_s_vec = i32x4_add(sum_s_vec, s);
      sum_d_vec = i32x4_add(sum_d_vec, d);
      sum_s2_vec = i32x4_add(sum_s2_vec, i32x4_mul(s, s));
      sum_d2_vec = i32x4_add(sum_d2_vec, i32x4_mul(d, d));
      sum_sd_vec = i32x4_add(sum_sd_vec, i32x4_mul(s, d));

      src_ptr = src_ptr.offset(src_stride);
      dst_ptr = dst_ptr.offset(dst_stride);
    }
  } else {
    // Generic fallback for other widths
    for _y in 0..h {
      for x in 0..w {
        let s = *src_ptr.add(x) as i32;
        let d = *dst_ptr.add(x) as i32;

        // Use scalar add to first lane
        sum_s_vec = i32x4_replace_lane::<0>(sum_s_vec, i32x4_extract_lane::<0>(sum_s_vec) + s);
        sum_d_vec = i32x4_replace_lane::<0>(sum_d_vec, i32x4_extract_lane::<0>(sum_d_vec) + d);
        sum_s2_vec = i32x4_replace_lane::<0>(sum_s2_vec, i32x4_extract_lane::<0>(sum_s2_vec) + s * s);
        sum_d2_vec = i32x4_replace_lane::<0>(sum_d2_vec, i32x4_extract_lane::<0>(sum_d2_vec) + d * d);
        sum_sd_vec = i32x4_replace_lane::<0>(sum_sd_vec, i32x4_extract_lane::<0>(sum_sd_vec) + s * d);
      }
      src_ptr = src_ptr.offset(src_stride);
      dst_ptr = dst_ptr.offset(dst_stride);
    }
  }

  // Horizontal sum from vectors
  let sum_s = (i32x4_extract_lane::<0>(sum_s_vec) + i32x4_extract_lane::<1>(sum_s_vec)
    + i32x4_extract_lane::<2>(sum_s_vec) + i32x4_extract_lane::<3>(sum_s_vec)) as u32;
  let sum_d = (i32x4_extract_lane::<0>(sum_d_vec) + i32x4_extract_lane::<1>(sum_d_vec)
    + i32x4_extract_lane::<2>(sum_d_vec) + i32x4_extract_lane::<3>(sum_d_vec)) as u32;
  let sum_s2 = (i32x4_extract_lane::<0>(sum_s2_vec) + i32x4_extract_lane::<1>(sum_s2_vec)
    + i32x4_extract_lane::<2>(sum_s2_vec) + i32x4_extract_lane::<3>(sum_s2_vec)) as u32;
  let sum_d2 = (i32x4_extract_lane::<0>(sum_d2_vec) + i32x4_extract_lane::<1>(sum_d2_vec)
    + i32x4_extract_lane::<2>(sum_d2_vec) + i32x4_extract_lane::<3>(sum_d2_vec)) as u32;
  let sum_sd = (i32x4_extract_lane::<0>(sum_sd_vec) + i32x4_extract_lane::<1>(sum_sd_vec)
    + i32x4_extract_lane::<2>(sum_sd_vec) + i32x4_extract_lane::<3>(sum_sd_vec)) as u32;

  let sse = sum_d2 + sum_s2 - 2 * sum_sd;

  // Calculate variance (scaled by area)
  let sum_s = sum_s as u64;
  let sum_d = sum_d as u64;

  let div = AREA_DIVISORS[w * h - 1] as u64;
  let div_shift = AREA_DIVISOR_BITS;

  let mut svar = sum_s2.saturating_sub(
    ((sum_s * sum_s * div + (1 << div_shift >> 1)) >> div_shift) as u32,
  );
  let mut dvar = sum_d2.saturating_sub(
    ((sum_d * sum_d * div + (1 << div_shift >> 1)) >> div_shift) as u32,
  );

  // Scale variances up to 8x8 size
  let scale_shift = AREA_DIVISOR_BITS - 6;
  svar =
    ((svar as u64 * div + (1 << scale_shift >> 1)) >> scale_shift) as u32;
  dvar =
    ((dvar as u64 * div + (1 << scale_shift >> 1)) >> scale_shift) as u32;

  (svar, dvar, sse)
}

/// CDEF distortion kernel for 16-bit (HBD) pixels.
#[inline(always)]
unsafe fn cdef_dist_kernel_hbd_simd128(
  src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize,
  w: usize, h: usize,
) -> (u32, u32, u32) {
  use core::arch::wasm32::*;

  let stride_elem = (src_stride / 2) as usize;
  let dst_stride_elem = (dst_stride / 2) as usize;

  // Use i64x2 for accumulating squared sums to avoid overflow
  let mut sum_s_vec = i32x4_splat(0);
  let mut sum_d_vec = i32x4_splat(0);
  let mut sum_s2: u64 = 0;
  let mut sum_d2: u64 = 0;
  let mut sum_sd: u64 = 0;

  let mut src_ptr = src;
  let mut dst_ptr = dst;

  if w == 8 {
    for _y in 0..h {
      // Load 8 i16 values (128 bits)
      let s = v128_load(src_ptr as *const v128);
      let d = v128_load(dst_ptr as *const v128);

      // Extend to i32 for sums
      let s_lo = i32x4_extend_low_i16x8(s);
      let s_hi = i32x4_extend_high_i16x8(s);
      let d_lo = i32x4_extend_low_i16x8(d);
      let d_hi = i32x4_extend_high_i16x8(d);

      sum_s_vec = i32x4_add(sum_s_vec, i32x4_add(s_lo, s_hi));
      sum_d_vec = i32x4_add(sum_d_vec, i32x4_add(d_lo, d_hi));

      // Compute squared sums with scalar to avoid overflow in 32-bit lanes
      for x in 0..8 {
        let sv = *src_ptr.add(x) as u64;
        let dv = *dst_ptr.add(x) as u64;
        sum_s2 += sv * sv;
        sum_d2 += dv * dv;
        sum_sd += sv * dv;
      }

      src_ptr = src_ptr.add(stride_elem);
      dst_ptr = dst_ptr.add(dst_stride_elem);
    }
  } else if w == 4 {
    for _y in 0..h {
      let s = i32x4(
        *src_ptr as i32, *src_ptr.add(1) as i32,
        *src_ptr.add(2) as i32, *src_ptr.add(3) as i32
      );
      let d = i32x4(
        *dst_ptr as i32, *dst_ptr.add(1) as i32,
        *dst_ptr.add(2) as i32, *dst_ptr.add(3) as i32
      );

      sum_s_vec = i32x4_add(sum_s_vec, s);
      sum_d_vec = i32x4_add(sum_d_vec, d);

      for x in 0..4 {
        let sv = *src_ptr.add(x) as u64;
        let dv = *dst_ptr.add(x) as u64;
        sum_s2 += sv * sv;
        sum_d2 += dv * dv;
        sum_sd += sv * dv;
      }

      src_ptr = src_ptr.add(stride_elem);
      dst_ptr = dst_ptr.add(dst_stride_elem);
    }
  } else {
    for _y in 0..h {
      for x in 0..w {
        let s = *src_ptr.add(x) as u64;
        let d = *dst_ptr.add(x) as u64;

        sum_s_vec = i32x4_replace_lane::<0>(sum_s_vec, i32x4_extract_lane::<0>(sum_s_vec) + s as i32);
        sum_d_vec = i32x4_replace_lane::<0>(sum_d_vec, i32x4_extract_lane::<0>(sum_d_vec) + d as i32);
        sum_s2 += s * s;
        sum_d2 += d * d;
        sum_sd += s * d;
      }
      src_ptr = src_ptr.add(stride_elem);
      dst_ptr = dst_ptr.add(dst_stride_elem);
    }
  }

  let sum_s = (i32x4_extract_lane::<0>(sum_s_vec) + i32x4_extract_lane::<1>(sum_s_vec)
    + i32x4_extract_lane::<2>(sum_s_vec) + i32x4_extract_lane::<3>(sum_s_vec)) as u32;
  let sum_d = (i32x4_extract_lane::<0>(sum_d_vec) + i32x4_extract_lane::<1>(sum_d_vec)
    + i32x4_extract_lane::<2>(sum_d_vec) + i32x4_extract_lane::<3>(sum_d_vec)) as u32;

  let sse = (sum_d2 + sum_s2 - 2 * sum_sd) as u32;

  let sum_s = sum_s as u64;
  let sum_d = sum_d as u64;

  let div = AREA_DIVISORS[w * h - 1] as u64;
  let div_shift = AREA_DIVISOR_BITS;

  let sum_s2 = sum_s2 as u32;
  let sum_d2 = sum_d2 as u32;

  let mut svar = sum_s2.saturating_sub(
    ((sum_s * sum_s * div + (1 << div_shift >> 1)) >> div_shift) as u32,
  );
  let mut dvar = sum_d2.saturating_sub(
    ((sum_d * sum_d * div + (1 << div_shift >> 1)) >> div_shift) as u32,
  );

  let scale_shift = AREA_DIVISOR_BITS - 6;
  svar =
    ((svar as u64 * div + (1 << scale_shift >> 1)) >> scale_shift) as u32;
  dvar =
    ((dvar as u64 * div + (1 << scale_shift >> 1)) >> scale_shift) as u32;

  (svar, dvar, sse)
}

// Function tables
static CDEF_DIST_KERNEL_FNS_SIMD128: [Option<CdefDistKernelFn>;
  CDEF_DIST_KERNEL_FNS_LENGTH] = {
  let mut out: [Option<CdefDistKernelFn>; CDEF_DIST_KERNEL_FNS_LENGTH] =
    [None; CDEF_DIST_KERNEL_FNS_LENGTH];

  // Support common CDEF block sizes
  out[kernel_fn_index(4, 4)] = Some(cdef_dist_kernel_simd128);
  out[kernel_fn_index(4, 8)] = Some(cdef_dist_kernel_simd128);
  out[kernel_fn_index(8, 4)] = Some(cdef_dist_kernel_simd128);
  out[kernel_fn_index(8, 8)] = Some(cdef_dist_kernel_simd128);

  out
};

static CDEF_DIST_KERNEL_HBD_FNS_SIMD128: [Option<CdefDistKernelHbdFn>;
  CDEF_DIST_KERNEL_FNS_LENGTH] = {
  let mut out: [Option<CdefDistKernelHbdFn>; CDEF_DIST_KERNEL_FNS_LENGTH] =
    [None; CDEF_DIST_KERNEL_FNS_LENGTH];

  out[kernel_fn_index(4, 4)] = Some(cdef_dist_kernel_hbd_simd128);
  out[kernel_fn_index(4, 8)] = Some(cdef_dist_kernel_hbd_simd128);
  out[kernel_fn_index(8, 4)] = Some(cdef_dist_kernel_hbd_simd128);
  out[kernel_fn_index(8, 8)] = Some(cdef_dist_kernel_hbd_simd128);

  out
};

cpu_function_lookup_table!(
  CDEF_DIST_KERNEL_FNS: [[Option<CdefDistKernelFn>; CDEF_DIST_KERNEL_FNS_LENGTH]],
  default: [None; CDEF_DIST_KERNEL_FNS_LENGTH],
  [SIMD128]
);

cpu_function_lookup_table!(
  CDEF_DIST_KERNEL_HBD_FNS: [[Option<CdefDistKernelHbdFn>; CDEF_DIST_KERNEL_FNS_LENGTH]],
  default: [None; CDEF_DIST_KERNEL_FNS_LENGTH],
  [SIMD128]
);
