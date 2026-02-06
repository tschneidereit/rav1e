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
  // Chunk size matches IMPORTANCE_BLOCK_SIZE >> 1 = 4
  const CHUNK_SIZE: usize = 4;

  let mut total_sse = 0u64;
  let mut src_ptr = src;
  let mut dst_ptr = dst;
  let mut scale_ptr = scale;

  for _chunk_y in (0..h).step_by(CHUNK_SIZE) {
    let chunk_h = CHUNK_SIZE.min(h - _chunk_y);

    for chunk_x in (0..w).step_by(CHUNK_SIZE) {
      let chunk_w = CHUNK_SIZE.min(w - chunk_x);
      let scale_val = *scale_ptr.add(chunk_x / CHUNK_SIZE);

      // Compute SSE for this 4x4 chunk
      let mut chunk_sse = 0u32;

      for cy in 0..chunk_h {
        for cx in 0..chunk_w {
          let s = *src_ptr.add(cy * src_stride as usize + chunk_x + cx) as i32;
          let d = *dst_ptr.add(cy * dst_stride as usize + chunk_x + cx) as i32;
          let diff = s - d;
          chunk_sse += (diff * diff) as u32;
        }
      }

      // Apply scale and shift
      total_sse += ((chunk_sse as u64 * scale_val as u64)
        + (1 << (rust::GET_WEIGHTED_SSE_SHIFT - 1)))
        >> rust::GET_WEIGHTED_SSE_SHIFT;
    }

    // Move to next row of chunks
    for _ in 0..CHUNK_SIZE.min(h - _chunk_y) {
      src_ptr = src_ptr.offset(src_stride);
      dst_ptr = dst_ptr.offset(dst_stride);
    }
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
  const CHUNK_SIZE: usize = 4;
  let stride_elem = src_stride / 2;
  let dst_stride_elem = dst_stride / 2;

  let mut total_sse = 0u64;
  let mut src_row = src;
  let mut dst_row = dst;
  let mut scale_row = scale;

  for chunk_y in (0..h).step_by(CHUNK_SIZE) {
    let chunk_h = CHUNK_SIZE.min(h - chunk_y);

    for chunk_x in (0..w).step_by(CHUNK_SIZE) {
      let chunk_w = CHUNK_SIZE.min(w - chunk_x);
      let scale_val = *scale_row.add(chunk_x / CHUNK_SIZE);

      let mut chunk_sse = 0u64;

      for cy in 0..chunk_h {
        for cx in 0..chunk_w {
          let s =
            *src_row.offset(cy as isize * stride_elem + (chunk_x + cx) as isize)
              as i32;
          let d = *dst_row
            .offset(cy as isize * dst_stride_elem + (chunk_x + cx) as isize)
            as i32;
          let diff = s - d;
          chunk_sse += (diff * diff) as u64;
        }
      }

      total_sse += ((chunk_sse * scale_val as u64)
        + (1 << (rust::GET_WEIGHTED_SSE_SHIFT - 1)))
        >> rust::GET_WEIGHTED_SSE_SHIFT;
    }

    // Move to next row of chunks
    src_row = src_row.offset(CHUNK_SIZE as isize * stride_elem);
    dst_row = dst_row.offset(CHUNK_SIZE as isize * dst_stride_elem);
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
