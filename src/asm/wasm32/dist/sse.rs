// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! Weighted SSE functions for wasm32 SIMD.

use crate::cpu_features::CpuFeatureLevel;
use crate::dist::*;
use crate::encoder::IMPORTANCE_BLOCK_SIZE;
use crate::partition::BlockSize;
use crate::rdo::DistortionScale;
use crate::tiling::PlaneRegion;
use crate::util::*;

use super::to_index;
use super::DIST_FNS_LENGTH;

type WeightedSseFn = unsafe fn(
  src: *const u8,
  src_stride: isize,
  dst: *const u8,
  dst_stride: isize,
  scale: *const u32,
  scale_stride: usize,
  w: usize,
  h: usize,
) -> u64;

type WeightedSseHbdFn = unsafe fn(
  src: *const u16,
  src_stride: isize,
  dst: *const u16,
  dst_stride: isize,
  scale: *const u32,
  scale_stride: usize,
  w: usize,
  h: usize,
) -> u64;

/// Compute weighted sum of squared errors.
#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_weighted_sse<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, scale: &[u32],
  scale_stride: usize, w: usize, h: usize, bit_depth: usize,
  cpu: CpuFeatureLevel,
) -> u64 {
  // Assembly breaks if imp block size changes.
  assert_eq!(IMPORTANCE_BLOCK_SIZE >> 1, 4);

  let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

  let call_rust = || -> u64 {
    rust::get_weighted_sse(dst, src, scale, scale_stride, w, h, bit_depth, cpu)
  };

  #[cfg(feature = "check_asm")]
  let ref_dist = call_rust();

  let den =
    DistortionScale::new(1, 1 << rust::GET_WEIGHTED_SSE_SHIFT).0 as u64;

  let dist = match (bsize_opt, T::type_enum()) {
    (Err(_), _) => call_rust(),
    (Ok(bsize), PixelType::U8) => {
      match SSE_FNS[cpu.as_index()][to_index(bsize)] {
        Some(func) => {
          // SAFETY: Pointers and strides are valid
          let raw = unsafe {
            func(
              src.data_ptr() as *const _,
              T::to_asm_stride(src.plane_cfg.stride),
              dst.data_ptr() as *const _,
              T::to_asm_stride(dst.plane_cfg.stride),
              scale.as_ptr(),
              scale_stride,
              w,
              h,
            )
          };
          (raw + (den >> 1)) / den
        }
        None => call_rust(),
      }
    }
    (Ok(bsize), PixelType::U16) => {
      match SSE_HBD_FNS[cpu.as_index()][to_index(bsize)] {
        Some(func) => {
          // SAFETY: Pointers and strides are valid
          let raw = unsafe {
            func(
              src.data_ptr() as *const _,
              T::to_asm_stride(src.plane_cfg.stride),
              dst.data_ptr() as *const _,
              T::to_asm_stride(dst.plane_cfg.stride),
              scale.as_ptr(),
              scale_stride,
              w,
              h,
            )
          };
          (raw + (den >> 1)) / den
        }
        None => call_rust(),
      }
    }
  };

  #[cfg(feature = "check_asm")]
  assert_eq!(
    dist, ref_dist,
    "Weighted SSE {:?}: SIMD doesn't match reference code.",
    bsize_opt
  );

  dist
}

/// Weighted SSE computation using SIMD for 8-bit pixels.
#[inline(always)]
unsafe fn weighted_sse_simd128(
  src: *const u8, src_stride: isize, dst: *const u8, dst_stride: isize,
  scale: *const u32, scale_stride: usize, w: usize, h: usize,
) -> u64 {
  use core::arch::wasm32::*;

  // Chunk size matches IMPORTANCE_BLOCK_SIZE >> 1 = 4
  const CHUNK_SIZE: usize = 4;

  let mut total_sse = 0u64;
  let mut src_row = src;
  let mut dst_row = dst;
  let mut scale_ptr = scale;

  for chunk_y in (0..h).step_by(CHUNK_SIZE) {
    let chunk_h = CHUNK_SIZE.min(h - chunk_y);

    // Process chunks of width 16 with full SIMD
    let mut chunk_x = 0;
    while chunk_x + 16 <= w {
      // Load 4 scale values (one per 4-wide chunk)
      let scale0 = *scale_ptr.add(chunk_x / CHUNK_SIZE);
      let scale1 = *scale_ptr.add(chunk_x / CHUNK_SIZE + 1);
      let scale2 = *scale_ptr.add(chunk_x / CHUNK_SIZE + 2);
      let scale3 = *scale_ptr.add(chunk_x / CHUNK_SIZE + 3);

      let mut sse0 = i32x4_splat(0);
      let mut sse1 = i32x4_splat(0);
      let mut sse2 = i32x4_splat(0);
      let mut sse3 = i32x4_splat(0);

      for cy in 0..chunk_h {
        let src_ptr = src_row.add(cy * src_stride as usize + chunk_x);
        let dst_ptr = dst_row.add(cy * dst_stride as usize + chunk_x);

        // Load 16 bytes from src and dst
        let s = v128_load(src_ptr as *const v128);
        let d = v128_load(dst_ptr as *const v128);

        // Convert to i16 and compute differences
        let zero = i8x16_splat(0);
        let s_lo = i16x8_extend_low_u8x16(s);
        let s_hi = i16x8_extend_high_u8x16(s);
        let d_lo = i16x8_extend_low_u8x16(d);
        let d_hi = i16x8_extend_high_u8x16(d);

        let diff_lo = i16x8_sub(s_lo, d_lo);
        let diff_hi = i16x8_sub(s_hi, d_hi);

        // Square differences (split into 32-bit to avoid overflow)
        // diff_lo: lanes 0-3A, 4-7B
        let diff0 = i32x4_extend_low_i16x8(diff_lo);   // lanes 0-3 (chunk 0)
        let diff1 = i32x4_extend_high_i16x8(diff_lo);  // lanes 4-7 (chunk 1)
        let diff2 = i32x4_extend_low_i16x8(diff_hi);   // lanes 8-11 (chunk 2)
        let diff3 = i32x4_extend_high_i16x8(diff_hi);  // lanes 12-15 (chunk 3)

        // Accumulate squares
        sse0 = i32x4_add(sse0, i32x4_mul(diff0, diff0));
        sse1 = i32x4_add(sse1, i32x4_mul(diff1, diff1));
        sse2 = i32x4_add(sse2, i32x4_mul(diff2, diff2));
        sse3 = i32x4_add(sse3, i32x4_mul(diff3, diff3));
      }

      // Horizontal sum each SSE accumulator and apply scale
      let sum0 = i32x4_extract_lane::<0>(sse0) + i32x4_extract_lane::<1>(sse0)
        + i32x4_extract_lane::<2>(sse0) + i32x4_extract_lane::<3>(sse0);
      let sum1 = i32x4_extract_lane::<0>(sse1) + i32x4_extract_lane::<1>(sse1)
        + i32x4_extract_lane::<2>(sse1) + i32x4_extract_lane::<3>(sse1);
      let sum2 = i32x4_extract_lane::<0>(sse2) + i32x4_extract_lane::<1>(sse2)
        + i32x4_extract_lane::<2>(sse2) + i32x4_extract_lane::<3>(sse2);
      let sum3 = i32x4_extract_lane::<0>(sse3) + i32x4_extract_lane::<1>(sse3)
        + i32x4_extract_lane::<2>(sse3) + i32x4_extract_lane::<3>(sse3);

      total_sse += ((sum0 as u64 * scale0 as u64)
        + (1 << (rust::GET_WEIGHTED_SSE_SHIFT - 1)))
        >> rust::GET_WEIGHTED_SSE_SHIFT;
      total_sse += ((sum1 as u64 * scale1 as u64)
        + (1 << (rust::GET_WEIGHTED_SSE_SHIFT - 1)))
        >> rust::GET_WEIGHTED_SSE_SHIFT;
      total_sse += ((sum2 as u64 * scale2 as u64)
        + (1 << (rust::GET_WEIGHTED_SSE_SHIFT - 1)))
        >> rust::GET_WEIGHTED_SSE_SHIFT;
      total_sse += ((sum3 as u64 * scale3 as u64)
        + (1 << (rust::GET_WEIGHTED_SSE_SHIFT - 1)))
        >> rust::GET_WEIGHTED_SSE_SHIFT;

      chunk_x += 16;
    }

    // Process remaining chunks of width 4-8 with partial SIMD
    while chunk_x + 4 <= w {
      let scale_val = *scale_ptr.add(chunk_x / CHUNK_SIZE);
      let mut chunk_sse = i32x4_splat(0);

      for cy in 0..chunk_h {
        let src_ptr = src_row.add(cy * src_stride as usize + chunk_x);
        let dst_ptr = dst_row.add(cy * dst_stride as usize + chunk_x);

        // Load 4 bytes, extend to i32
        let s = i32x4(
          *src_ptr as i32, *src_ptr.add(1) as i32,
          *src_ptr.add(2) as i32, *src_ptr.add(3) as i32
        );
        let d = i32x4(
          *dst_ptr as i32, *dst_ptr.add(1) as i32,
          *dst_ptr.add(2) as i32, *dst_ptr.add(3) as i32
        );

        let diff = i32x4_sub(s, d);
        chunk_sse = i32x4_add(chunk_sse, i32x4_mul(diff, diff));
      }

      let sum = i32x4_extract_lane::<0>(chunk_sse) + i32x4_extract_lane::<1>(chunk_sse)
        + i32x4_extract_lane::<2>(chunk_sse) + i32x4_extract_lane::<3>(chunk_sse);

      total_sse += ((sum as u64 * scale_val as u64)
        + (1 << (rust::GET_WEIGHTED_SSE_SHIFT - 1)))
        >> rust::GET_WEIGHTED_SSE_SHIFT;

      chunk_x += 4;
    }

    // Scalar fallback for remaining pixels (should rarely happen with power-of-2 blocks)
    while chunk_x < w {
      let scale_val = *scale_ptr.add(chunk_x / CHUNK_SIZE);
      let mut chunk_sse = 0u32;

      for cy in 0..chunk_h {
        let s = *src_row.add(cy * src_stride as usize + chunk_x) as i32;
        let d = *dst_row.add(cy * dst_stride as usize + chunk_x) as i32;
        let diff = s - d;
        chunk_sse += (diff * diff) as u32;
      }

      total_sse += ((chunk_sse as u64 * scale_val as u64)
        + (1 << (rust::GET_WEIGHTED_SSE_SHIFT - 1)))
        >> rust::GET_WEIGHTED_SSE_SHIFT;

      chunk_x += 1;
    }

    // Move to next row of chunks
    src_row = src_row.offset(chunk_h as isize * src_stride);
    dst_row = dst_row.offset(chunk_h as isize * dst_stride);
    scale_ptr = scale_ptr.add(scale_stride);
  }

  total_sse
}

/// Weighted SSE with SIMD for 16-bit (HBD) pixels.
#[inline(always)]
unsafe fn weighted_sse_hbd_simd128(
  src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize,
  scale: *const u32, scale_stride: usize, w: usize, h: usize,
) -> u64 {
  use core::arch::wasm32::*;

  const CHUNK_SIZE: usize = 4;
  let stride_elem = (src_stride / 2) as usize;
  let dst_stride_elem = (dst_stride / 2) as usize;

  let mut total_sse = 0u64;
  let mut src_row = src;
  let mut dst_row = dst;
  let mut scale_row = scale;

  for chunk_y in (0..h).step_by(CHUNK_SIZE) {
    let chunk_h = CHUNK_SIZE.min(h - chunk_y);

    // Process chunks of width 8 with full SIMD (8 x i16 = 128 bits)
    let mut chunk_x = 0;
    while chunk_x + 8 <= w {
      let scale0 = *scale_row.add(chunk_x / CHUNK_SIZE);
      let scale1 = *scale_row.add(chunk_x / CHUNK_SIZE + 1);

      let mut sse0 = i32x4_splat(0);
      let mut sse1 = i32x4_splat(0);

      for cy in 0..chunk_h {
        let src_ptr = src_row.add(cy * stride_elem + chunk_x);
        let dst_ptr = dst_row.add(cy * dst_stride_elem + chunk_x);

        // Load 8 i16 values (128 bits)
        let s = v128_load(src_ptr as *const v128);
        let d = v128_load(dst_ptr as *const v128);

        // Compute difference as i16, then extend to i32
        let diff = i16x8_sub(s, d);
        let diff_lo = i32x4_extend_low_i16x8(diff);   // lanes 0-3 (chunk 0)
        let diff_hi = i32x4_extend_high_i16x8(diff);  // lanes 4-7 (chunk 1)

        // Accumulate squares
        sse0 = i32x4_add(sse0, i32x4_mul(diff_lo, diff_lo));
        sse1 = i32x4_add(sse1, i32x4_mul(diff_hi, diff_hi));
      }

      // Horizontal sum and apply scale
      let sum0 = i32x4_extract_lane::<0>(sse0) + i32x4_extract_lane::<1>(sse0)
        + i32x4_extract_lane::<2>(sse0) + i32x4_extract_lane::<3>(sse0);
      let sum1 = i32x4_extract_lane::<0>(sse1) + i32x4_extract_lane::<1>(sse1)
        + i32x4_extract_lane::<2>(sse1) + i32x4_extract_lane::<3>(sse1);

      total_sse += ((sum0 as u64 * scale0 as u64)
        + (1 << (rust::GET_WEIGHTED_SSE_SHIFT - 1)))
        >> rust::GET_WEIGHTED_SSE_SHIFT;
      total_sse += ((sum1 as u64 * scale1 as u64)
        + (1 << (rust::GET_WEIGHTED_SSE_SHIFT - 1)))
        >> rust::GET_WEIGHTED_SSE_SHIFT;

      chunk_x += 8;
    }

    // Process remaining 4-wide chunks
    while chunk_x + 4 <= w {
      let scale_val = *scale_row.add(chunk_x / CHUNK_SIZE);
      let mut sse = i32x4_splat(0);

      for cy in 0..chunk_h {
        let src_ptr = src_row.add(cy * stride_elem + chunk_x);
        let dst_ptr = dst_row.add(cy * dst_stride_elem + chunk_x);

        // Load 4 i16 values
        let s = i32x4(
          *src_ptr as i32, *src_ptr.add(1) as i32,
          *src_ptr.add(2) as i32, *src_ptr.add(3) as i32
        );
        let d = i32x4(
          *dst_ptr as i32, *dst_ptr.add(1) as i32,
          *dst_ptr.add(2) as i32, *dst_ptr.add(3) as i32
        );

        let diff = i32x4_sub(s, d);
        sse = i32x4_add(sse, i32x4_mul(diff, diff));
      }

      let sum = i32x4_extract_lane::<0>(sse) + i32x4_extract_lane::<1>(sse)
        + i32x4_extract_lane::<2>(sse) + i32x4_extract_lane::<3>(sse);

      total_sse += ((sum as u64 * scale_val as u64)
        + (1 << (rust::GET_WEIGHTED_SSE_SHIFT - 1)))
        >> rust::GET_WEIGHTED_SSE_SHIFT;

      chunk_x += 4;
    }

    // Scalar fallback
    while chunk_x < w {
      let scale_val = *scale_row.add(chunk_x / CHUNK_SIZE);
      let mut chunk_sse = 0u64;

      for cy in 0..chunk_h {
        let s = *src_row.add(cy * stride_elem + chunk_x) as i32;
        let d = *dst_row.add(cy * dst_stride_elem + chunk_x) as i32;
        let diff = s - d;
        chunk_sse += (diff * diff) as u64;
      }

      total_sse += ((chunk_sse * scale_val as u64)
        + (1 << (rust::GET_WEIGHTED_SSE_SHIFT - 1)))
        >> rust::GET_WEIGHTED_SSE_SHIFT;

      chunk_x += 1;
    }

    // Move to next row of chunks
    src_row = src_row.add(chunk_h * stride_elem);
    dst_row = dst_row.add(chunk_h * dst_stride_elem);
    scale_row = scale_row.add(scale_stride);
  }

  total_sse
}

// Function dispatch tables
static SSE_FNS_SIMD128: [Option<WeightedSseFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<WeightedSseFn>; DIST_FNS_LENGTH] =
    [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_4X8 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_4X16 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_8X4 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_8X8 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_8X16 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_8X32 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_16X4 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_16X8 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_16X16 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_16X32 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_16X64 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_32X8 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_32X16 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_32X32 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_32X64 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_64X16 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_64X32 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_64X64 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_64X128 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_128X64 as usize] = Some(weighted_sse_simd128);
  out[BLOCK_128X128 as usize] = Some(weighted_sse_simd128);

  out
};

static SSE_HBD_FNS_SIMD128: [Option<WeightedSseHbdFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<WeightedSseHbdFn>; DIST_FNS_LENGTH] =
    [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_4X8 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_4X16 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_8X4 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_8X8 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_8X16 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_8X32 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_16X4 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_16X8 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_16X16 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_16X32 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_16X64 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_32X8 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_32X16 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_32X32 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_32X64 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_64X16 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_64X32 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_64X64 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_64X128 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_128X64 as usize] = Some(weighted_sse_hbd_simd128);
  out[BLOCK_128X128 as usize] = Some(weighted_sse_hbd_simd128);

  out
};

cpu_function_lookup_table!(
  SSE_FNS: [[Option<WeightedSseFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [SIMD128]
);

cpu_function_lookup_table!(
  SSE_HBD_FNS: [[Option<WeightedSseHbdFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [SIMD128]
);
