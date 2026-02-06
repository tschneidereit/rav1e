// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! SIMD-accelerated distortion functions for wasm32.

pub use self::cdef_dist::*;
pub use self::sse::*;

use crate::cpu_features::CpuFeatureLevel;
use crate::dist::*;
use crate::partition::BlockSize;
use crate::tiling::*;
use crate::util::*;

use core::arch::wasm32::*;

mod cdef_dist;
mod sse;

use super::simd_helpers::*;

// BlockSize::BLOCK_SIZES.next_power_of_two()
const DIST_FNS_LENGTH: usize = 32;

#[inline]
const fn to_index(bsize: BlockSize) -> usize {
  bsize as usize & (DIST_FNS_LENGTH - 1)
}

/// Sum of absolute differences (SAD) for 8-bit pixels
#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_sad<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, w: usize, h: usize,
  bit_depth: usize, cpu: CpuFeatureLevel,
) -> u32 {
  let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

  let call_rust = || -> u32 { rust::get_sad(src, dst, w, h, bit_depth, cpu) };

  #[cfg(feature = "check_asm")]
  let ref_dist = call_rust();

  let dist = match (bsize_opt, T::type_enum()) {
    (Err(_), _) => call_rust(),
    (Ok(bsize), PixelType::U8) => {
      match SAD_FNS[cpu.as_index()][to_index(bsize)] {
        Some(func) => {
          // SAFETY: We're passing valid pointers and strides
          unsafe {
            func(
              src.data_ptr() as *const u8,
              T::to_asm_stride(src.plane_cfg.stride),
              dst.data_ptr() as *const u8,
              T::to_asm_stride(dst.plane_cfg.stride),
              w,
              h,
            )
          }
        }
        None => call_rust(),
      }
    }
    (Ok(bsize), PixelType::U16) => {
      match SAD_HBD_FNS[cpu.as_index()][to_index(bsize)] {
        Some(func) => {
          // SAFETY: We're passing valid pointers and strides
          unsafe {
            func(
              src.data_ptr() as *const u16,
              T::to_asm_stride(src.plane_cfg.stride),
              dst.data_ptr() as *const u16,
              T::to_asm_stride(dst.plane_cfg.stride),
              w,
              h,
            )
          }
        }
        None => call_rust(),
      }
    }
  };

  #[cfg(feature = "check_asm")]
  assert_eq!(dist, ref_dist);

  dist
}

/// Sum of absolute transformed differences (SATD) using Hadamard transform
#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_satd<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, w: usize, h: usize,
  bit_depth: usize, cpu: CpuFeatureLevel,
) -> u32 {
  let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

  let call_rust = || -> u32 { rust::get_satd(src, dst, w, h, bit_depth, cpu) };

  #[cfg(feature = "check_asm")]
  let ref_dist = call_rust();

  let dist = match (bsize_opt, T::type_enum()) {
    (Err(_), _) => call_rust(),
    (Ok(bsize), PixelType::U8) => {
      match SATD_FNS[cpu.as_index()][to_index(bsize)] {
        Some(func) => {
          // SAFETY: We're passing valid pointers and strides
          unsafe {
            func(
              src.data_ptr() as *const u8,
              T::to_asm_stride(src.plane_cfg.stride),
              dst.data_ptr() as *const u8,
              T::to_asm_stride(dst.plane_cfg.stride),
              w,
              h,
            )
          }
        }
        None => call_rust(),
      }
    }
    (Ok(bsize), PixelType::U16) => {
      match SATD_HBD_FNS[cpu.as_index()][to_index(bsize)] {
        Some(func) => {
          // SAFETY: We're passing valid pointers and strides
          unsafe {
            func(
              src.data_ptr() as *const u16,
              T::to_asm_stride(src.plane_cfg.stride),
              dst.data_ptr() as *const u16,
              T::to_asm_stride(dst.plane_cfg.stride),
              w,
              h,
            )
          }
        }
        None => call_rust(),
      }
    }
  };

  #[cfg(feature = "check_asm")]
  assert_eq!(dist, ref_dist);

  dist
}

// Function types for dispatch tables
type SadFn = unsafe fn(
  src: *const u8,
  src_stride: isize,
  dst: *const u8,
  dst_stride: isize,
  w: usize,
  h: usize,
) -> u32;

type SadHbdFn = unsafe fn(
  src: *const u16,
  src_stride: isize,
  dst: *const u16,
  dst_stride: isize,
  w: usize,
  h: usize,
) -> u32;

type SatdFn = SadFn;
type SatdHbdFn = SadHbdFn;

// ============================================================================
// SAD implementations
// ============================================================================

/// Generic SAD implementation using SIMD for 8-bit pixels.
/// Handles any block size by processing in 16-byte chunks.
#[inline(always)]
unsafe fn sad_wxh_simd128(
  src: *const u8, src_stride: isize, dst: *const u8, dst_stride: isize,
  w: usize, h: usize,
) -> u32 {
  let mut acc = u32x4_splat(0);
  let mut src_ptr = src;
  let mut dst_ptr = dst;

  for _y in 0..h {
    let mut x = 0;

    // Process 16 bytes at a time
    while x + 16 <= w {
      let s = v128_load(src_ptr.add(x) as *const v128);
      let d = v128_load(dst_ptr.add(x) as *const v128);
      let diff = abs_diff_u8x16(s, d);
      // Accumulate: extend to u16, then to u32
      let lo16 = u16x8_extend_low_u8x16(diff);
      let hi16 = u16x8_extend_high_u8x16(diff);
      let lo32 = u32x4_add(u32x4_extend_low_u16x8(lo16), u32x4_extend_high_u16x8(lo16));
      let hi32 = u32x4_add(u32x4_extend_low_u16x8(hi16), u32x4_extend_high_u16x8(hi16));
      acc = u32x4_add(acc, u32x4_add(lo32, hi32));
      x += 16;
    }

    // Process 8 bytes at a time
    while x + 8 <= w {
      let s = v128_load64_zero(src_ptr.add(x) as *const u64);
      let d = v128_load64_zero(dst_ptr.add(x) as *const u64);
      let diff = abs_diff_u8x16(s, d);
      let lo16 = u16x8_extend_low_u8x16(diff);
      let lo32 = u32x4_add(u32x4_extend_low_u16x8(lo16), u32x4_extend_high_u16x8(lo16));
      acc = u32x4_add(acc, lo32);
      x += 8;
    }

    // Handle remaining bytes (4 or fewer)
    if x + 4 <= w {
      let s = v128_load32_zero(src_ptr.add(x) as *const u32);
      let d = v128_load32_zero(dst_ptr.add(x) as *const u32);
      let diff = abs_diff_u8x16(s, d);
      let lo16 = u16x8_extend_low_u8x16(diff);
      let lo32 = u32x4_extend_low_u16x8(lo16);
      acc = u32x4_add(acc, lo32);
      x += 4;
    }

    // Scalar remainder
    for i in x..w {
      let s = *src_ptr.add(i) as i32;
      let d = *dst_ptr.add(i) as i32;
      acc = u32x4_add(acc, u32x4_splat((s - d).unsigned_abs()));
    }

    src_ptr = src_ptr.offset(src_stride);
    dst_ptr = dst_ptr.offset(dst_stride);
  }

  horizontal_sum_u32x4(acc)
}

/// SAD for 16-bit (HBD) pixels
#[inline(always)]
unsafe fn sad_hbd_wxh_simd128(
  src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize,
  w: usize, h: usize,
) -> u32 {
  let mut acc = u32x4_splat(0);
  let mut src_ptr = src;
  let mut dst_ptr = dst;

  for _y in 0..h {
    let mut x = 0;

    // Process 8 u16 values at a time
    while x + 8 <= w {
      let s = v128_load(src_ptr.add(x) as *const v128);
      let d = v128_load(dst_ptr.add(x) as *const v128);
      let diff = abs_diff_u16x8(s, d);
      // Extend to u32 and accumulate
      let lo32 = u32x4_extend_low_u16x8(diff);
      let hi32 = u32x4_extend_high_u16x8(diff);
      acc = u32x4_add(acc, u32x4_add(lo32, hi32));
      x += 8;
    }

    // Process 4 u16 values
    if x + 4 <= w {
      let s = v128_load64_zero(src_ptr.add(x) as *const u64);
      let d = v128_load64_zero(dst_ptr.add(x) as *const u64);
      let diff = abs_diff_u16x8(s, d);
      let lo32 = u32x4_extend_low_u16x8(diff);
      acc = u32x4_add(acc, lo32);
      x += 4;
    }

    // Scalar remainder
    for i in x..w {
      let s = *src_ptr.add(i) as i32;
      let d = *dst_ptr.add(i) as i32;
      acc = u32x4_add(acc, u32x4_splat((s - d).unsigned_abs()));
    }

    src_ptr = src_ptr.wrapping_offset(src_stride / 2);
    dst_ptr = dst_ptr.wrapping_offset(dst_stride / 2);
  }

  horizontal_sum_u32x4(acc)
}

// ============================================================================
// SATD implementations (Hadamard transform based)
// ============================================================================

/// 4x4 Hadamard transform butterfly operations
#[inline(always)]
fn hadamard4_1d(a: i32, b: i32, c: i32, d: i32) -> (i32, i32, i32, i32) {
  let t0 = a + b;
  let t1 = a - b;
  let t2 = c + d;
  let t3 = c - d;
  (t0 + t2, t1 + t3, t0 - t2, t1 - t3)
}

/// 4x4 SATD using SIMD
#[inline(always)]
unsafe fn satd_4x4_simd128<T>(
  src: *const T, src_stride: isize, dst: *const T, dst_stride: isize,
) -> u32
where
  T: Copy,
  i32: From<T>,
{
  let stride = src_stride / core::mem::size_of::<T>() as isize;
  let dst_stride = dst_stride / core::mem::size_of::<T>() as isize;

  // Load 4x4 block and compute differences
  let mut diff = [[0i32; 4]; 4];
  for y in 0..4 {
    for x in 0..4 {
      let s = i32::from(*src.offset(y * stride + x as isize));
      let d = i32::from(*dst.offset(y * dst_stride + x as isize));
      diff[y as usize][x] = s - d;
    }
  }

  // Horizontal transform
  let mut tmp = [[0i32; 4]; 4];
  for y in 0..4 {
    let (a, b, c, d) =
      hadamard4_1d(diff[y][0], diff[y][1], diff[y][2], diff[y][3]);
    tmp[y] = [a, b, c, d];
  }

  // Vertical transform and sum absolute values
  let mut sum = 0u32;
  for x in 0..4 {
    let (a, b, c, d) =
      hadamard4_1d(tmp[0][x], tmp[1][x], tmp[2][x], tmp[3][x]);
    sum += a.unsigned_abs() + b.unsigned_abs() + c.unsigned_abs() + d.unsigned_abs();
  }

  // Normalize: for 4x4 transform, ln = msb(4) = 2, so (sum + 2) >> 2
  (sum + 2) >> 2
}

/// 8x8 Hadamard transform
#[inline(always)]
fn hadamard8_1d(
  a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32,
) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
  let (t0, t1, t2, t3) = hadamard4_1d(a, b, c, d);
  let (t4, t5, t6, t7) = hadamard4_1d(e, f, g, h);
  (
    t0 + t4,
    t1 + t5,
    t2 + t6,
    t3 + t7,
    t0 - t4,
    t1 - t5,
    t2 - t6,
    t3 - t7,
  )
}

/// 8x8 SATD
#[inline(always)]
unsafe fn satd_8x8_simd128<T>(
  src: *const T, src_stride: isize, dst: *const T, dst_stride: isize,
) -> u32
where
  T: Copy,
  i32: From<T>,
{
  let stride = src_stride / core::mem::size_of::<T>() as isize;
  let dst_stride_elem = dst_stride / core::mem::size_of::<T>() as isize;

  // Load 8x8 block and compute differences
  let mut diff = [[0i32; 8]; 8];
  for y in 0..8 {
    for x in 0..8 {
      let s = i32::from(*src.offset(y * stride + x as isize));
      let d = i32::from(*dst.offset(y * dst_stride_elem + x as isize));
      diff[y as usize][x] = s - d;
    }
  }

  // Horizontal transform
  let mut tmp = [[0i32; 8]; 8];
  for y in 0..8 {
    let (a, b, c, d, e, f, g, h) = hadamard8_1d(
      diff[y][0],
      diff[y][1],
      diff[y][2],
      diff[y][3],
      diff[y][4],
      diff[y][5],
      diff[y][6],
      diff[y][7],
    );
    tmp[y] = [a, b, c, d, e, f, g, h];
  }

  // Vertical transform and sum absolute values
  let mut sum = 0u32;
  for x in 0..8 {
    let (a, b, c, d, e, f, g, h) = hadamard8_1d(
      tmp[0][x],
      tmp[1][x],
      tmp[2][x],
      tmp[3][x],
      tmp[4][x],
      tmp[5][x],
      tmp[6][x],
      tmp[7][x],
    );
    sum += a.unsigned_abs()
      + b.unsigned_abs()
      + c.unsigned_abs()
      + d.unsigned_abs()
      + e.unsigned_abs()
      + f.unsigned_abs()
      + g.unsigned_abs()
      + h.unsigned_abs();
  }

  // Normalize: for 8x8 transform, ln = msb(8) = 3, so (sum + 4) >> 3
  (sum + 4) >> 3
}

/// Generic SATD that tiles the block with 4x4 or 8x8 transforms
#[inline(always)]
unsafe fn satd_wxh_simd128(
  src: *const u8, src_stride: isize, dst: *const u8, dst_stride: isize,
  w: usize, h: usize,
) -> u32 {
  let size = w.min(h).min(8);
  let mut sum = 0u32;

  if size >= 8 {
    // Use 8x8 transforms
    for chunk_y in (0..h).step_by(8) {
      for chunk_x in (0..w).step_by(8) {
        let chunk_w = (w - chunk_x).min(8);
        let chunk_h = (h - chunk_y).min(8);

        if chunk_w == 8 && chunk_h == 8 {
          sum += satd_8x8_simd128(
            src.offset(chunk_y as isize * src_stride / 1 + chunk_x as isize),
            src_stride,
            dst.offset(chunk_y as isize * dst_stride / 1 + chunk_x as isize),
            dst_stride,
          );
        } else {
          // Edge case: use 4x4 or fall through to scalar
          for sub_y in (0..chunk_h).step_by(4) {
            for sub_x in (0..chunk_w).step_by(4) {
              let sub_w = (chunk_w - sub_x).min(4);
              let sub_h = (chunk_h - sub_y).min(4);
              if sub_w == 4 && sub_h == 4 {
                sum += satd_4x4_simd128(
                  src.offset(
                    (chunk_y + sub_y) as isize * src_stride / 1
                      + (chunk_x + sub_x) as isize,
                  ),
                  src_stride,
                  dst.offset(
                    (chunk_y + sub_y) as isize * dst_stride / 1
                      + (chunk_x + sub_x) as isize,
                  ),
                  dst_stride,
                );
              } else {
                // Very small remainder - use SAD
                sum += sad_block_scalar(
                  src.offset(
                    (chunk_y + sub_y) as isize * src_stride / 1
                      + (chunk_x + sub_x) as isize,
                  ),
                  src_stride,
                  dst.offset(
                    (chunk_y + sub_y) as isize * dst_stride / 1
                      + (chunk_x + sub_x) as isize,
                  ),
                  dst_stride,
                  sub_w,
                  sub_h,
                );
              }
            }
          }
        }
      }
    }
  } else {
    // Use 4x4 transforms for small blocks
    for chunk_y in (0..h).step_by(4) {
      for chunk_x in (0..w).step_by(4) {
        let chunk_w = (w - chunk_x).min(4);
        let chunk_h = (h - chunk_y).min(4);

        if chunk_w == 4 && chunk_h == 4 {
          sum += satd_4x4_simd128(
            src.offset(chunk_y as isize * src_stride / 1 + chunk_x as isize),
            src_stride,
            dst.offset(chunk_y as isize * dst_stride / 1 + chunk_x as isize),
            dst_stride,
          );
        } else {
          // Very small remainder - use SAD
          sum += sad_block_scalar(
            src.offset(chunk_y as isize * src_stride / 1 + chunk_x as isize),
            src_stride,
            dst.offset(chunk_y as isize * dst_stride / 1 + chunk_x as isize),
            dst_stride,
            chunk_w,
            chunk_h,
          );
        }
      }
    }
  }

  sum
}

/// Scalar SAD for small block remainders
#[inline(always)]
unsafe fn sad_block_scalar<T>(
  src: *const T, src_stride: isize, dst: *const T, dst_stride: isize,
  w: usize, h: usize,
) -> u32
where
  T: Copy,
  i32: From<T>,
{
  let src_stride_elem = src_stride / core::mem::size_of::<T>() as isize;
  let dst_stride_elem = dst_stride / core::mem::size_of::<T>() as isize;
  let mut sum = 0u32;

  for y in 0..h {
    for x in 0..w {
      let s = i32::from(*src.offset(y as isize * src_stride_elem + x as isize));
      let d = i32::from(*dst.offset(y as isize * dst_stride_elem + x as isize));
      sum += (s - d).unsigned_abs();
    }
  }

  sum
}

/// HBD SATD
#[inline(always)]
unsafe fn satd_hbd_wxh_simd128(
  src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize,
  w: usize, h: usize,
) -> u32 {
  let size = w.min(h).min(8);
  let mut sum = 0u32;

  if size >= 8 {
    for chunk_y in (0..h).step_by(8) {
      for chunk_x in (0..w).step_by(8) {
        let chunk_w = (w - chunk_x).min(8);
        let chunk_h = (h - chunk_y).min(8);

        if chunk_w == 8 && chunk_h == 8 {
          sum += satd_8x8_simd128(
            src.offset(chunk_y as isize * src_stride / 2 + chunk_x as isize),
            src_stride,
            dst.offset(chunk_y as isize * dst_stride / 2 + chunk_x as isize),
            dst_stride,
          );
        } else {
          // Fall back to 4x4 or scalar
          for sub_y in (0..chunk_h).step_by(4) {
            for sub_x in (0..chunk_w).step_by(4) {
              let sub_w = (chunk_w - sub_x).min(4);
              let sub_h = (chunk_h - sub_y).min(4);
              if sub_w == 4 && sub_h == 4 {
                sum += satd_4x4_simd128(
                  src.offset(
                    (chunk_y + sub_y) as isize * src_stride / 2
                      + (chunk_x + sub_x) as isize,
                  ),
                  src_stride,
                  dst.offset(
                    (chunk_y + sub_y) as isize * dst_stride / 2
                      + (chunk_x + sub_x) as isize,
                  ),
                  dst_stride,
                );
              } else {
                sum += sad_block_scalar(
                  src.offset(
                    (chunk_y + sub_y) as isize * src_stride / 2
                      + (chunk_x + sub_x) as isize,
                  ),
                  src_stride,
                  dst.offset(
                    (chunk_y + sub_y) as isize * dst_stride / 2
                      + (chunk_x + sub_x) as isize,
                  ),
                  dst_stride,
                  sub_w,
                  sub_h,
                );
              }
            }
          }
        }
      }
    }
  } else {
    for chunk_y in (0..h).step_by(4) {
      for chunk_x in (0..w).step_by(4) {
        let chunk_w = (w - chunk_x).min(4);
        let chunk_h = (h - chunk_y).min(4);

        if chunk_w == 4 && chunk_h == 4 {
          sum += satd_4x4_simd128(
            src.offset(chunk_y as isize * src_stride / 2 + chunk_x as isize),
            src_stride,
            dst.offset(chunk_y as isize * dst_stride / 2 + chunk_x as isize),
            dst_stride,
          );
        } else {
          sum += sad_block_scalar(
            src.offset(chunk_y as isize * src_stride / 2 + chunk_x as isize),
            src_stride,
            dst.offset(chunk_y as isize * dst_stride / 2 + chunk_x as isize),
            dst_stride,
            chunk_w,
            chunk_h,
          );
        }
      }
    }
  }

  sum
}

// ============================================================================
// Function dispatch tables
// ============================================================================

/// Unified SAD function that dispatches to the right implementation
unsafe fn sad_simd128(
  src: *const u8, src_stride: isize, dst: *const u8, dst_stride: isize,
  w: usize, h: usize,
) -> u32 {
  sad_wxh_simd128(src, src_stride, dst, dst_stride, w, h)
}

unsafe fn sad_hbd_simd128(
  src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize,
  w: usize, h: usize,
) -> u32 {
  sad_hbd_wxh_simd128(src, src_stride, dst, dst_stride, w, h)
}

unsafe fn satd_simd128(
  src: *const u8, src_stride: isize, dst: *const u8, dst_stride: isize,
  w: usize, h: usize,
) -> u32 {
  satd_wxh_simd128(src, src_stride, dst, dst_stride, w, h)
}

unsafe fn satd_hbd_simd128(
  src: *const u16, src_stride: isize, dst: *const u16, dst_stride: isize,
  w: usize, h: usize,
) -> u32 {
  satd_hbd_wxh_simd128(src, src_stride, dst, dst_stride, w, h)
}

// Build dispatch tables
static SAD_FNS_SIMD128: [Option<SadFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SadFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(sad_simd128);
  out[BLOCK_4X8 as usize] = Some(sad_simd128);
  out[BLOCK_4X16 as usize] = Some(sad_simd128);
  out[BLOCK_8X4 as usize] = Some(sad_simd128);
  out[BLOCK_8X8 as usize] = Some(sad_simd128);
  out[BLOCK_8X16 as usize] = Some(sad_simd128);
  out[BLOCK_8X32 as usize] = Some(sad_simd128);
  out[BLOCK_16X4 as usize] = Some(sad_simd128);
  out[BLOCK_16X8 as usize] = Some(sad_simd128);
  out[BLOCK_16X16 as usize] = Some(sad_simd128);
  out[BLOCK_16X32 as usize] = Some(sad_simd128);
  out[BLOCK_16X64 as usize] = Some(sad_simd128);
  out[BLOCK_32X8 as usize] = Some(sad_simd128);
  out[BLOCK_32X16 as usize] = Some(sad_simd128);
  out[BLOCK_32X32 as usize] = Some(sad_simd128);
  out[BLOCK_32X64 as usize] = Some(sad_simd128);
  out[BLOCK_64X16 as usize] = Some(sad_simd128);
  out[BLOCK_64X32 as usize] = Some(sad_simd128);
  out[BLOCK_64X64 as usize] = Some(sad_simd128);
  out[BLOCK_64X128 as usize] = Some(sad_simd128);
  out[BLOCK_128X64 as usize] = Some(sad_simd128);
  out[BLOCK_128X128 as usize] = Some(sad_simd128);

  out
};

static SAD_HBD_FNS_SIMD128: [Option<SadHbdFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SadHbdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_4X8 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_4X16 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_8X4 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_8X8 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_8X16 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_8X32 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_16X4 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_16X8 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_16X16 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_16X32 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_16X64 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_32X8 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_32X16 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_32X32 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_32X64 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_64X16 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_64X32 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_64X64 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_64X128 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_128X64 as usize] = Some(sad_hbd_simd128);
  out[BLOCK_128X128 as usize] = Some(sad_hbd_simd128);

  out
};

static SATD_FNS_SIMD128: [Option<SatdFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SatdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(satd_simd128);
  out[BLOCK_4X8 as usize] = Some(satd_simd128);
  out[BLOCK_4X16 as usize] = Some(satd_simd128);
  out[BLOCK_8X4 as usize] = Some(satd_simd128);
  out[BLOCK_8X8 as usize] = Some(satd_simd128);
  out[BLOCK_8X16 as usize] = Some(satd_simd128);
  out[BLOCK_8X32 as usize] = Some(satd_simd128);
  out[BLOCK_16X4 as usize] = Some(satd_simd128);
  out[BLOCK_16X8 as usize] = Some(satd_simd128);
  out[BLOCK_16X16 as usize] = Some(satd_simd128);
  out[BLOCK_16X32 as usize] = Some(satd_simd128);
  out[BLOCK_16X64 as usize] = Some(satd_simd128);
  out[BLOCK_32X8 as usize] = Some(satd_simd128);
  out[BLOCK_32X16 as usize] = Some(satd_simd128);
  out[BLOCK_32X32 as usize] = Some(satd_simd128);
  out[BLOCK_32X64 as usize] = Some(satd_simd128);
  out[BLOCK_64X16 as usize] = Some(satd_simd128);
  out[BLOCK_64X32 as usize] = Some(satd_simd128);
  out[BLOCK_64X64 as usize] = Some(satd_simd128);
  out[BLOCK_64X128 as usize] = Some(satd_simd128);
  out[BLOCK_128X64 as usize] = Some(satd_simd128);
  out[BLOCK_128X128 as usize] = Some(satd_simd128);

  out
};

static SATD_HBD_FNS_SIMD128: [Option<SatdHbdFn>; DIST_FNS_LENGTH] = {
  let mut out: [Option<SatdHbdFn>; DIST_FNS_LENGTH] = [None; DIST_FNS_LENGTH];

  use BlockSize::*;

  out[BLOCK_4X4 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_4X8 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_4X16 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_8X4 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_8X8 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_8X16 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_8X32 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_16X4 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_16X8 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_16X16 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_16X32 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_16X64 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_32X8 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_32X16 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_32X32 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_32X64 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_64X16 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_64X32 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_64X64 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_64X128 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_128X64 as usize] = Some(satd_hbd_simd128);
  out[BLOCK_128X128 as usize] = Some(satd_hbd_simd128);

  out
};

cpu_function_lookup_table!(
  SAD_FNS: [[Option<SadFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [SIMD128]
);

cpu_function_lookup_table!(
  SAD_HBD_FNS: [[Option<SadHbdFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [SIMD128]
);

cpu_function_lookup_table!(
  SATD_FNS: [[Option<SatdFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [SIMD128]
);

cpu_function_lookup_table!(
  SATD_HBD_FNS: [[Option<SatdHbdFn>; DIST_FNS_LENGTH]],
  default: [None; DIST_FNS_LENGTH],
  [SIMD128]
);

#[cfg(test)]
mod tests {
  use super::*;
  use crate::frame::{AsRegion, Plane};
  use rand::Rng;

  /// Helper to test SAD for a given block size
  fn test_sad_size(width: usize, height: usize) {
    let mut rng = rand::thread_rng();
    let mut src_data = vec![0u8; width * height];
    let mut dst_data = vec![0u8; width * height];

    for i in 0..(width * height) {
      src_data[i] = rng.gen();
      dst_data[i] = rng.gen();
    }

    let src_plane = Plane::from_slice(&src_data, width);
    let dst_plane = Plane::from_slice(&dst_data, width);

    let rust_result = rust::get_sad(
      &src_plane.as_region(),
      &dst_plane.as_region(),
      width,
      height,
      8,
      CpuFeatureLevel::RUST,
    );

    let simd_result = get_sad(
      &src_plane.as_region(),
      &dst_plane.as_region(),
      width,
      height,
      8,
      CpuFeatureLevel::SIMD128,
    );

    assert_eq!(
      rust_result, simd_result,
      "SAD mismatch for {}x{}: rust={}, simd={}",
      width, height, rust_result, simd_result
    );
  }

  /// Helper to test SATD for a given block size
  fn test_satd_size(width: usize, height: usize) {
    let mut rng = rand::thread_rng();
    let mut src_data = vec![0u8; width * height];
    let mut dst_data = vec![0u8; width * height];

    for i in 0..(width * height) {
      src_data[i] = rng.gen();
      dst_data[i] = rng.gen();
    }

    let src_plane = Plane::from_slice(&src_data, width);
    let dst_plane = Plane::from_slice(&dst_data, width);

    let rust_result = rust::get_satd(
      &src_plane.as_region(),
      &dst_plane.as_region(),
      width,
      height,
      8,
      CpuFeatureLevel::RUST,
    );

    let simd_result = get_satd(
      &src_plane.as_region(),
      &dst_plane.as_region(),
      width,
      height,
      8,
      CpuFeatureLevel::SIMD128,
    );

    assert_eq!(
      rust_result, simd_result,
      "SATD mismatch for {}x{}: rust={}, simd={}",
      width, height, rust_result, simd_result
    );
  }

  /// Helper to test HBD SAD for a given block size
  fn test_sad_hbd_size(width: usize, height: usize) {
    let mut rng = rand::thread_rng();
    let mut src_data = vec![0u16; width * height];
    let mut dst_data = vec![0u16; width * height];

    for i in 0..(width * height) {
      src_data[i] = rng.gen::<u16>() & 0x3FF; // 10-bit
      dst_data[i] = rng.gen::<u16>() & 0x3FF;
    }

    let src_plane = Plane::from_slice(&src_data, width);
    let dst_plane = Plane::from_slice(&dst_data, width);

    let rust_result = rust::get_sad(
      &src_plane.as_region(),
      &dst_plane.as_region(),
      width,
      height,
      10,
      CpuFeatureLevel::RUST,
    );

    let simd_result = get_sad(
      &src_plane.as_region(),
      &dst_plane.as_region(),
      width,
      height,
      10,
      CpuFeatureLevel::SIMD128,
    );

    assert_eq!(
      rust_result, simd_result,
      "SAD HBD mismatch for {}x{}: rust={}, simd={}",
      width, height, rust_result, simd_result
    );
  }

  // Square dimension tests
  #[test]
  fn test_sad_4x4() {
    test_sad_size(4, 4);
  }

  #[test]
  fn test_sad_8x8() {
    test_sad_size(8, 8);
  }

  #[test]
  fn test_sad_16x16() {
    test_sad_size(16, 16);
  }

  #[test]
  fn test_sad_32x32() {
    test_sad_size(32, 32);
  }

  // Asymmetric dimension tests - these catch w/h swap bugs
  #[test]
  fn test_sad_4x8() {
    test_sad_size(4, 8);
  }

  #[test]
  fn test_sad_8x4() {
    test_sad_size(8, 4);
  }

  #[test]
  fn test_sad_8x16() {
    test_sad_size(8, 16);
  }

  #[test]
  fn test_sad_16x8() {
    test_sad_size(16, 8);
  }

  #[test]
  fn test_sad_16x32() {
    test_sad_size(16, 32);
  }

  #[test]
  fn test_sad_32x16() {
    test_sad_size(32, 16);
  }

  #[test]
  fn test_sad_4x16() {
    test_sad_size(4, 16);
  }

  #[test]
  fn test_sad_16x4() {
    test_sad_size(16, 4);
  }

  // SATD tests - square
  #[test]
  fn test_satd_4x4() {
    test_satd_size(4, 4);
  }

  #[test]
  fn test_satd_8x8() {
    test_satd_size(8, 8);
  }

  #[test]
  fn test_satd_16x16() {
    test_satd_size(16, 16);
  }

  // SATD tests - asymmetric
  #[test]
  fn test_satd_4x8() {
    test_satd_size(4, 8);
  }

  #[test]
  fn test_satd_8x4() {
    test_satd_size(8, 4);
  }

  #[test]
  fn test_satd_8x16() {
    test_satd_size(8, 16);
  }

  #[test]
  fn test_satd_16x8() {
    test_satd_size(16, 8);
  }

  // HBD (high bit depth) tests
  #[test]
  fn test_sad_hbd_8x8() {
    test_sad_hbd_size(8, 8);
  }

  #[test]
  fn test_sad_hbd_8x4() {
    test_sad_hbd_size(8, 4);
  }

  #[test]
  fn test_sad_hbd_4x8() {
    test_sad_hbd_size(4, 8);
  }

  #[test]
  fn test_sad_hbd_16x8() {
    test_sad_hbd_size(16, 8);
  }

  #[test]
  fn test_sad_hbd_8x16() {
    test_sad_hbd_size(8, 16);
  }
}
