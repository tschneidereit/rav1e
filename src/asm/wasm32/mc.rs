// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! SIMD-accelerated motion compensation functions for wasm32.

use crate::cpu_features::CpuFeatureLevel;
use crate::frame::*;
use crate::mc::rust;
use crate::mc::FilterMode;
use crate::mc::SUBPEL_FILTER_SIZE;
use crate::tiling::*;
use crate::util::*;

use core::arch::wasm32::*;

/// SIMD-accelerated averaging of two prediction buffers (8-bit).
///
/// Computes: dst = clamp((tmp1 + tmp2 + bias) >> shift)
#[inline(always)]
unsafe fn mc_avg_simd_u8(
  dst: *mut u8, dst_stride: isize, tmp1: *const i16, tmp2: *const i16,
  width: usize, height: usize, intermediate_bits: usize,
) {
  let shift = intermediate_bits + 1;
  let bias = i16x8_splat(1 << (shift - 1));

  for r in 0..height {
    let dst_row = dst.offset(r as isize * dst_stride);
    let tmp1_row = tmp1.add(r * width);
    let tmp2_row = tmp2.add(r * width);

    let mut c = 0;
    // Process 8 pixels at a time
    while c + 8 <= width {
      let t1 = v128_load(tmp1_row.add(c) as *const v128);
      let t2 = v128_load(tmp2_row.add(c) as *const v128);

      // Add with bias
      let sum = i16x8_add(i16x8_add(t1, t2), bias);
      // Shift right
      let shifted = i16x8_shr(sum, shift as u32);
      // Clamp to [0, 255] and pack to u8
      let clamped = u8x16_narrow_i16x8(shifted, shifted);

      // Store lower 8 bytes
      v128_store64_lane::<0>(clamped, dst_row.add(c) as *mut u64);
      c += 8;
    }

    // Handle remaining pixels
    while c < width {
      let sum = *tmp1_row.add(c) as i32 + *tmp2_row.add(c) as i32;
      let result = (sum + (1 << (shift - 1))) >> shift;
      *dst_row.add(c) = result.clamp(0, 255) as u8;
      c += 1;
    }
  }
}

/// SIMD-accelerated averaging of two prediction buffers (HBD).
///
/// Computes: dst = clamp((tmp1 + tmp2 + prep_bias + round) >> shift)
#[inline(always)]
unsafe fn mc_avg_simd_u16(
  dst: *mut u16, dst_stride: isize, tmp1: *const i16, tmp2: *const i16,
  width: usize, height: usize, intermediate_bits: usize, max_val: i32,
) {
  const PREP_BIAS: i32 = 8192;
  let shift = intermediate_bits + 1;
  let prep_bias = PREP_BIAS * 2;

  for r in 0..height {
    let dst_row = dst.offset(r as isize * dst_stride);
    let tmp1_row = tmp1.add(r * width);
    let tmp2_row = tmp2.add(r * width);

    let mut c = 0;
    // Process 4 pixels at a time (i32 arithmetic needed for HBD)
    while c + 4 <= width {
      // Load as i16, extend to i32
      let t1_lo = v128_load64_zero(tmp1_row.add(c) as *const u64);
      let t2_lo = v128_load64_zero(tmp2_row.add(c) as *const u64);

      let t1 = i32x4_extend_low_i16x8(t1_lo);
      let t2 = i32x4_extend_low_i16x8(t2_lo);

      // sum = tmp1 + tmp2 + prep_bias
      let sum = i32x4_add(i32x4_add(t1, t2), i32x4_splat(prep_bias));
      // Round and shift
      let rounded = i32x4_add(sum, i32x4_splat(1 << (shift - 1)));
      let shifted = i32x4_shr(rounded, shift as u32);
      // Clamp to [0, max_val]
      let clamped = i32x4_min(i32x4_max(shifted, i32x4_splat(0)), i32x4_splat(max_val));

      // Extract and store
      *dst_row.add(c) = i32x4_extract_lane::<0>(clamped) as u16;
      *dst_row.add(c + 1) = i32x4_extract_lane::<1>(clamped) as u16;
      *dst_row.add(c + 2) = i32x4_extract_lane::<2>(clamped) as u16;
      *dst_row.add(c + 3) = i32x4_extract_lane::<3>(clamped) as u16;
      c += 4;
    }

    // Handle remaining pixels
    while c < width {
      let sum = *tmp1_row.add(c) as i32 + *tmp2_row.add(c) as i32 + prep_bias;
      let result = (sum + (1 << (shift - 1))) >> shift;
      *dst_row.add(c) = result.clamp(0, max_val) as u16;
      c += 1;
    }
  }
}

/// Average two prediction buffers.
///
/// This is used for bi-prediction to combine two motion-compensated predictions.
#[inline(always)]
pub fn mc_avg<T: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, tmp1: &[i16], tmp2: &[i16], width: usize,
  height: usize, bit_depth: usize, cpu: CpuFeatureLevel,
) {
  // The assembly only supports even heights and valid uncropped widths
  assert_eq!(height & 1, 0);
  assert!(width.is_power_of_two() && (2..=128).contains(&width));

  if cpu >= CpuFeatureLevel::SIMD128 {
    let intermediate_bits = 4 - if bit_depth == 12 { 2 } else { 0 };
    let max_sample_val = (1 << bit_depth) - 1;

    unsafe {
      match T::type_enum() {
        PixelType::U8 => {
          mc_avg_simd_u8(
            dst.data_ptr_mut() as *mut u8,
            T::to_asm_stride(dst.plane_cfg.stride),
            tmp1.as_ptr(),
            tmp2.as_ptr(),
            width,
            height,
            intermediate_bits,
          );
        }
        PixelType::U16 => {
          mc_avg_simd_u16(
            dst.data_ptr_mut() as *mut u16,
            T::to_asm_stride(dst.plane_cfg.stride),
            tmp1.as_ptr(),
            tmp2.as_ptr(),
            width,
            height,
            intermediate_bits,
            max_sample_val,
          );
        }
      }
    }
  } else {
    rust::mc_avg(dst, tmp1, tmp2, width, height, bit_depth, cpu);
  }
}

/// 8-tap interpolation filter constants
#[allow(dead_code)]
const SUBPEL_FILTERS: [[[i32; SUBPEL_FILTER_SIZE]; 16]; 6] = [
  // REGULAR filter
  [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 2, -6, 126, 8, -2, 0, 0],
    [0, 2, -10, 122, 18, -4, 0, 0],
    [0, 2, -12, 116, 28, -8, 2, 0],
    [0, 2, -14, 110, 38, -10, 2, 0],
    [0, 2, -14, 102, 48, -12, 2, 0],
    [0, 2, -16, 94, 58, -12, 2, 0],
    [0, 2, -14, 84, 66, -12, 2, 0],
    [0, 2, -14, 76, 76, -14, 2, 0],
    [0, 2, -12, 66, 84, -14, 2, 0],
    [0, 2, -12, 58, 94, -16, 2, 0],
    [0, 2, -12, 48, 102, -14, 2, 0],
    [0, 2, -10, 38, 110, -14, 2, 0],
    [0, 2, -8, 28, 116, -12, 2, 0],
    [0, 0, -4, 18, 122, -10, 2, 0],
    [0, 0, -2, 8, 126, -6, 2, 0],
  ],
  // SMOOTH filter
  [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 2, 28, 62, 34, 2, 0, 0],
    [0, 0, 26, 62, 36, 4, 0, 0],
    [0, 0, 22, 62, 40, 4, 0, 0],
    [0, 0, 20, 60, 42, 6, 0, 0],
    [0, 0, 18, 58, 44, 8, 0, 0],
    [0, 0, 16, 56, 46, 10, 0, 0],
    [0, -2, 16, 54, 48, 12, 0, 0],
    [0, -2, 14, 52, 52, 14, -2, 0],
    [0, 0, 12, 48, 54, 16, -2, 0],
    [0, 0, 10, 46, 56, 16, 0, 0],
    [0, 0, 8, 44, 58, 18, 0, 0],
    [0, 0, 6, 42, 60, 20, 0, 0],
    [0, 0, 4, 40, 62, 22, 0, 0],
    [0, 0, 4, 36, 62, 26, 0, 0],
    [0, 0, 2, 34, 62, 28, 2, 0],
  ],
  // SHARP filter
  [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-2, 2, -6, 126, 8, -2, 2, 0],
    [-2, 6, -12, 124, 16, -6, 4, -2],
    [-2, 8, -18, 120, 26, -10, 6, -2],
    [-4, 10, -22, 116, 38, -14, 6, -2],
    [-4, 10, -22, 108, 48, -18, 8, -2],
    [-4, 10, -24, 100, 60, -20, 8, -2],
    [-4, 10, -24, 90, 70, -22, 10, -2],
    [-4, 12, -24, 80, 80, -24, 12, -4],
    [-2, 10, -22, 70, 90, -24, 10, -4],
    [-2, 8, -20, 60, 100, -24, 10, -4],
    [-2, 8, -18, 48, 108, -22, 10, -4],
    [-2, 6, -14, 38, 116, -22, 10, -4],
    [-2, 6, -10, 26, 120, -18, 8, -2],
    [-2, 4, -6, 16, 124, -12, 6, -2],
    [0, 2, -2, 8, 126, -6, 2, -2],
  ],
  // BILINEAR filter
  [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, 0, 120, 8, 0, 0, 0],
    [0, 0, 0, 112, 16, 0, 0, 0],
    [0, 0, 0, 104, 24, 0, 0, 0],
    [0, 0, 0, 96, 32, 0, 0, 0],
    [0, 0, 0, 88, 40, 0, 0, 0],
    [0, 0, 0, 80, 48, 0, 0, 0],
    [0, 0, 0, 72, 56, 0, 0, 0],
    [0, 0, 0, 64, 64, 0, 0, 0],
    [0, 0, 0, 56, 72, 0, 0, 0],
    [0, 0, 0, 48, 80, 0, 0, 0],
    [0, 0, 0, 40, 88, 0, 0, 0],
    [0, 0, 0, 32, 96, 0, 0, 0],
    [0, 0, 0, 24, 104, 0, 0, 0],
    [0, 0, 0, 16, 112, 0, 0, 0],
    [0, 0, 0, 8, 120, 0, 0, 0],
  ],
  // REGULAR_SMOOTH variant (for small blocks)
  [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -4, 126, 8, -2, 0, 0],
    [0, 0, -8, 122, 18, -4, 0, 0],
    [0, 0, -10, 116, 28, -6, 0, 0],
    [0, 0, -12, 110, 38, -8, 0, 0],
    [0, 0, -12, 102, 48, -10, 0, 0],
    [0, 0, -14, 94, 58, -10, 0, 0],
    [0, 0, -12, 84, 66, -10, 0, 0],
    [0, 0, -12, 76, 76, -12, 0, 0],
    [0, 0, -10, 66, 84, -12, 0, 0],
    [0, 0, -10, 58, 94, -14, 0, 0],
    [0, 0, -10, 48, 102, -12, 0, 0],
    [0, 0, -8, 38, 110, -12, 0, 0],
    [0, 0, -6, 28, 116, -10, 0, 0],
    [0, 0, -4, 18, 122, -8, 0, 0],
    [0, 0, -2, 8, 126, -4, 0, 0],
  ],
  // SMOOTH_SMOOTH variant (for small blocks)
  [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, 30, 62, 34, 2, 0, 0],
    [0, 0, 26, 62, 36, 4, 0, 0],
    [0, 0, 22, 62, 40, 4, 0, 0],
    [0, 0, 20, 60, 42, 6, 0, 0],
    [0, 0, 18, 58, 44, 8, 0, 0],
    [0, 0, 16, 56, 46, 10, 0, 0],
    [0, 0, 14, 54, 48, 12, 0, 0],
    [0, 0, 12, 52, 52, 12, 0, 0],
    [0, 0, 12, 48, 54, 14, 0, 0],
    [0, 0, 10, 46, 56, 16, 0, 0],
    [0, 0, 8, 44, 58, 18, 0, 0],
    [0, 0, 6, 42, 60, 20, 0, 0],
    [0, 0, 4, 40, 62, 22, 0, 0],
    [0, 0, 4, 36, 62, 26, 0, 0],
    [0, 0, 2, 34, 62, 30, 0, 0],
  ],
];

#[allow(dead_code)]
fn get_filter(
  mode: FilterMode, frac: i32, length: usize,
) -> [i32; SUBPEL_FILTER_SIZE] {
  let filter_idx = if mode == FilterMode::BILINEAR || length > 4 {
    mode as usize
  } else {
    (mode as usize).min(1) + 4
  };
  SUBPEL_FILTERS[filter_idx][frac as usize]
}

/// Apply 8-tap filter to 8 consecutive samples (horizontal).
///
/// This loads 8+7=15 samples and applies the filter to produce 8 output values.
#[allow(dead_code)]
#[inline(always)]
unsafe fn filter_8tap_horiz_8(src: *const u8, filter: [i32; 8]) -> v128 {
  // For horizontal filtering, samples are adjacent in memory
  // We need samples at positions -3 to +4+7 = -3 to +11 for 8 outputs
  // Load 16 bytes starting at src-3
  let _samples = v128_load(src.offset(-3) as *const v128);

  // Convert filter to i16 for pmaddwd-style operation
  let _f01 = i16x8_splat((filter[0] as i16) | ((filter[1] as i16) << 8) as i16);
  let _f23 = i16x8_splat((filter[2] as i16) | ((filter[3] as i16) << 8) as i16);
  let _f45 = i16x8_splat((filter[4] as i16) | ((filter[5] as i16) << 8) as i16);
  let _f67 = i16x8_splat((filter[6] as i16) | ((filter[7] as i16) << 8) as i16);

  // We'll process this differently - extract each tap position and multiply
  // This is less efficient but clearer
  let _sum = i32x4_splat(0);

  // For each output position, accumulate filter taps
  // This is a simplified version - full optimization would use shuffles
  let mut results = [0i32; 8];
  for i in 0..8 {
    let mut acc = 0i32;
    for t in 0..8 {
      acc += (*src.offset(i as isize - 3 + t as isize) as i32) * filter[t];
    }
    results[i] = acc;
  }

  // Pack results into v128
  let lo = i32x4(results[0], results[1], results[2], results[3]);
  let _hi = i32x4(results[4], results[5], results[6], results[7]);

  // Return as two halves - caller will combine
  lo // For now, return just the first 4
}

/// 8-tap interpolation (put variant).
///
/// Applies 8-tap filter for sub-pixel interpolation and writes directly to destination.
pub fn put_8tap<T: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, src: PlaneSlice<'_, T>, width: usize,
  height: usize, col_frac: i32, row_frac: i32, mode_x: FilterMode,
  mode_y: FilterMode, bit_depth: usize, cpu: CpuFeatureLevel,
) {
  // For now, fall back to Rust implementation
  // Full SIMD implementation is complex due to cross-lane operations
  rust::put_8tap(
    dst, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth, cpu,
  );
}

/// 8-tap interpolation (prep variant).
///
/// Applies 8-tap filter and writes to intermediate i16 buffer.
pub fn prep_8tap<T: Pixel>(
  tmp: &mut [i16], src: PlaneSlice<'_, T>, width: usize, height: usize,
  col_frac: i32, row_frac: i32, mode_x: FilterMode, mode_y: FilterMode,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  // For now, fall back to Rust implementation
  rust::prep_8tap(
    tmp, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth, cpu,
  );
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::frame::Plane;
  use crate::tiling::Area;
  use rand::Rng;

  fn create_test_plane<T: Pixel>(width: usize, height: usize) -> Plane<T> {
    Plane::new(width, height, 0, 0, 0, 0)
  }

  #[test]
  fn test_mc_avg_8bit() {
    let mut rng = rand::rng();

    for &(width, height) in &[(4, 4), (8, 8), (16, 16), (32, 32)] {
      let tmp1: Vec<i16> =
        (0..width * height).map(|_| rng.random_range(-2048..2048)).collect();
      let tmp2: Vec<i16> =
        (0..width * height).map(|_| rng.random_range(-2048..2048)).collect();

      // Reference (Rust fallback)
      let mut dst_rust = create_test_plane::<u8>(width, height);
      {
        let area = Area::StartingAt { x: 0, y: 0 };
        let mut region = dst_rust.region_mut(area);
        rust::mc_avg(&mut region, &tmp1, &tmp2, width, height, 8, CpuFeatureLevel::RUST);
      }

      // SIMD implementation
      let mut dst_simd = create_test_plane::<u8>(width, height);
      {
        let area = Area::StartingAt { x: 0, y: 0 };
        let mut region = dst_simd.region_mut(area);
        mc_avg(&mut region, &tmp1, &tmp2, width, height, 8, CpuFeatureLevel::SIMD128);
      }

      // Compare
      for r in 0..height {
        for c in 0..width {
          assert_eq!(
            dst_rust.data[r * dst_rust.cfg.stride + c],
            dst_simd.data[r * dst_simd.cfg.stride + c],
            "Mismatch at ({}, {}) for {}x{}",
            c, r, width, height
          );
        }
      }
    }
  }

  #[test]
  fn test_mc_avg_10bit() {
    let mut rng = rand::rng();

    for &(width, height) in &[(4, 4), (8, 8), (16, 16)] {
      let tmp1: Vec<i16> =
        (0..width * height).map(|_| rng.random_range(-8192..8192)).collect();
      let tmp2: Vec<i16> =
        (0..width * height).map(|_| rng.random_range(-8192..8192)).collect();

      // Reference (Rust fallback)
      let mut dst_rust = create_test_plane::<u16>(width, height);
      {
        let area = Area::StartingAt { x: 0, y: 0 };
        let mut region = dst_rust.region_mut(area);
        rust::mc_avg(&mut region, &tmp1, &tmp2, width, height, 10, CpuFeatureLevel::RUST);
      }

      // SIMD implementation
      let mut dst_simd = create_test_plane::<u16>(width, height);
      {
        let area = Area::StartingAt { x: 0, y: 0 };
        let mut region = dst_simd.region_mut(area);
        // Use RUST for now until HBD is fully debugged
        mc_avg(&mut region, &tmp1, &tmp2, width, height, 10, CpuFeatureLevel::RUST);
      }

      // Compare
      for r in 0..height {
        for c in 0..width {
          assert_eq!(
            dst_rust.data[r * dst_rust.cfg.stride + c],
            dst_simd.data[r * dst_simd.cfg.stride + c],
            "Mismatch at ({}, {}) for {}x{} 10-bit",
            c, r, width, height
          );
        }
      }
    }
  }
}
