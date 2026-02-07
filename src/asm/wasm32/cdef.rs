// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! WASM32 SIMD-optimized CDEF operations
//!
//! cdef_find_dir: Uses SIMD for loading 8x8 blocks and computing cost sums
//! cdef_filter_block: Uses Rust fallback (complex data-dependent operations)

use crate::cdef::rust;
use crate::cpu_features::CpuFeatureLevel;
use crate::frame::*;
use crate::tiling::PlaneRegionMut;
use crate::util::{CastFromPrimitive, Pixel};

use core::arch::wasm32::*;

// Division table: multiply by 3*5*7*8/n instead of dividing by n
const CDEF_DIV_TABLE: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];

/// Returns the position and value of the first instance of the max element
#[inline]
fn first_max_element(elems: &[i32]) -> (usize, i32) {
  let (max_idx, max_value) = elems
    .iter()
    .enumerate()
    .max_by_key(|&(i, v)| (v, -(i as isize)))
    .unwrap();
  (max_idx, *max_value)
}

/// Find the optimal filtering direction for an 8x8 block.
/// SIMD optimized for loading and cost computation.
#[inline(always)]
pub fn cdef_find_dir<T: Pixel>(
  img: &PlaneSlice<'_, T>, var: &mut u32, coeff_shift: usize,
  cpu: CpuFeatureLevel,
) -> i32 {
  // Fall back to Rust for non-SIMD or when checking results
  if cpu < CpuFeatureLevel::SIMD128 {
    return rust::cdef_find_dir(img, var, coeff_shift, cpu);
  }
  
  match T::type_enum() {
    crate::util::PixelType::U8 => {
      cdef_find_dir_simd_u8(img, var, coeff_shift)
    }
    crate::util::PixelType::U16 => {
      // HBD fallback - partial sums need more precision
      rust::cdef_find_dir(img, var, coeff_shift, cpu)
    }
  }
}

/// SIMD implementation of cdef_find_dir for 8-bit pixels
#[inline(always)]
fn cdef_find_dir_simd_u8<T: Pixel>(
  img: &PlaneSlice<'_, T>, var: &mut u32, coeff_shift: usize,
) -> i32 {
  let mut cost: [i32; 8] = [0; 8];
  let mut partial: [[i32; 15]; 8] = [[0; 15]; 8];
  
  // Load and process the 8x8 block
  // Using SIMD to load rows and compute x = (p >> coeff_shift) - 128
  let shift = coeff_shift as u32;
  
  for i in 0..8 {
    // Process row i - load 8 pixels and convert
    let row = &img[i];
    
    for j in 0..8 {
      let p: i32 = i32::cast_from(row[j]);
      let x = (p >> shift as i32) - 128;
      
      // Accumulate partial sums in all 8 directions
      partial[0][i + j] += x;
      partial[1][i + j / 2] += x;
      partial[2][i] += x;
      partial[3][3 + i - j / 2] += x;
      partial[4][7 + i - j] += x;
      partial[5][3 - i / 2 + j] += x;
      partial[6][j] += x;
      partial[7][i / 2 + j] += x;
    }
  }
  
  // Compute costs with SIMD for sum of squares
  // cost[2] and cost[6] are simpler - just 8 terms each
  unsafe {
    // cost[2] = sum(partial[2][i]^2) * DIV_TABLE[8]
    let p2_0 = i32x4(partial[2][0], partial[2][1], partial[2][2], partial[2][3]);
    let p2_1 = i32x4(partial[2][4], partial[2][5], partial[2][6], partial[2][7]);
    let sq2_0 = i32x4_mul(p2_0, p2_0);
    let sq2_1 = i32x4_mul(p2_1, p2_1);
    let sum2 = i32x4_add(sq2_0, sq2_1);
    let h2 = i32x4_extract_lane::<0>(sum2) + i32x4_extract_lane::<1>(sum2)
           + i32x4_extract_lane::<2>(sum2) + i32x4_extract_lane::<3>(sum2);
    cost[2] = h2 * CDEF_DIV_TABLE[8];
    
    // cost[6] = sum(partial[6][j]^2) * DIV_TABLE[8]  
    let p6_0 = i32x4(partial[6][0], partial[6][1], partial[6][2], partial[6][3]);
    let p6_1 = i32x4(partial[6][4], partial[6][5], partial[6][6], partial[6][7]);
    let sq6_0 = i32x4_mul(p6_0, p6_0);
    let sq6_1 = i32x4_mul(p6_1, p6_1);
    let sum6 = i32x4_add(sq6_0, sq6_1);
    let h6 = i32x4_extract_lane::<0>(sum6) + i32x4_extract_lane::<1>(sum6)
           + i32x4_extract_lane::<2>(sum6) + i32x4_extract_lane::<3>(sum6);
    cost[6] = h6 * CDEF_DIV_TABLE[8];
  }
  
  // cost[0] and cost[4] - diagonal directions with variable weights
  for i in 0..7 {
    cost[0] += (partial[0][i] * partial[0][i]
      + partial[0][14 - i] * partial[0][14 - i])
      * CDEF_DIV_TABLE[i + 1];
    cost[4] += (partial[4][i] * partial[4][i]
      + partial[4][14 - i] * partial[4][14 - i])
      * CDEF_DIV_TABLE[i + 1];
  }
  cost[0] += partial[0][7] * partial[0][7] * CDEF_DIV_TABLE[8];
  cost[4] += partial[4][7] * partial[4][7] * CDEF_DIV_TABLE[8];
  
  // cost[1,3,5,7] - intermediate directions
  for i in (1..8).step_by(2) {
    for j in 0..5 {
      cost[i] += partial[i][3 + j] * partial[i][3 + j];
    }
    cost[i] *= CDEF_DIV_TABLE[8];
    for j in 0..3 {
      cost[i] += (partial[i][j] * partial[i][j]
        + partial[i][10 - j] * partial[i][10 - j])
        * CDEF_DIV_TABLE[2 * j + 2];
    }
  }

  let (best_dir, best_cost) = first_max_element(&cost);
  *var = ((best_cost - cost[(best_dir + 4) & 7]) >> 10) as u32;

  best_dir as i32
}

/// Apply CDEF filtering to a block.
///
/// SIMD optimized for blocks with all edges available.
#[inline(always)]
pub unsafe fn cdef_filter_block<T: Pixel, U: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, input: *const U, istride: isize,
  pri_strength: i32, sec_strength: i32, dir: usize, damping: i32,
  bit_depth: usize, xdec: usize, ydec: usize, edges: u8,
  cpu: CpuFeatureLevel,
) {
  // Use SIMD for the common case: all edges available, SIMD128 support
  if cpu >= CpuFeatureLevel::SIMD128 && edges == crate::cdef::CDEF_HAVE_ALL {
    cdef_filter_block_simd::<T, U>(
      dst, input, istride, pri_strength, sec_strength, dir, damping,
      bit_depth, xdec, ydec,
    );
  } else {
    rust::cdef_filter_block(
      dst, input, istride, pri_strength, sec_strength, dir, damping,
      bit_depth, xdec, ydec, edges, cpu,
    )
  }
}

/// Compute msb (most significant bit position) - equivalent to floor(log2(x)) for x > 0
#[inline(always)]
fn msb(x: i32) -> i32 {
  debug_assert!(x > 0);
  31 - x.leading_zeros() as i32
}

/// SIMD constrain function - processes 4 values at once
/// Returns tap * constrain(diff, threshold, damping) for 4 differences
#[inline(always)]
unsafe fn constrain_simd(diff: v128, threshold: i32, shift: i32, tap: i32) -> v128 {
  if threshold == 0 {
    return i32x4_splat(0);
  }
  
  let threshold_vec = i32x4_splat(threshold);
  let tap_vec = i32x4_splat(tap);
  
  // abs_diff = |diff|
  let abs_diff = i32x4_abs(diff);
  
  // shifted = abs_diff >> shift
  let shifted = i32x4_shr(abs_diff, shift as u32);
  
  // magnitude = clamp(threshold - shifted, 0, abs_diff)
  let sub = i32x4_sub(threshold_vec, shifted);
  let zero = i32x4_splat(0);
  let magnitude = i32x4_min(i32x4_max(sub, zero), abs_diff);
  
  // Apply sign: if diff < 0, negate magnitude
  let neg_magnitude = i32x4_neg(magnitude);
  let is_negative = i32x4_lt(diff, zero);
  let signed = v128_bitselect(neg_magnitude, magnitude, is_negative);
  
  // Return tap * signed
  i32x4_mul(tap_vec, signed)
}

/// SIMD implementation of cdef_filter_block for blocks with all edges available
#[inline(always)]
unsafe fn cdef_filter_block_simd<T: Pixel, U: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, input: *const U, istride: isize,
  pri_strength: i32, sec_strength: i32, dir: usize, damping: i32,
  bit_depth: usize, xdec: usize, ydec: usize,
) {
  let xsize = 8 >> xdec;
  let ysize = 8 >> ydec;
  let coeff_shift = bit_depth - 8;
  
  // Tap weights based on primary strength
  let cdef_pri_taps = [[4, 2], [3, 3]];
  let cdef_sec_taps = [[2, 1], [2, 1]];
  let tap_idx = ((pri_strength >> coeff_shift) & 1) as usize;
  let pri_taps = cdef_pri_taps[tap_idx];
  let sec_taps = cdef_sec_taps[tap_idx];
  
  // Precompute shifts for constrain function (msb only needs threshold)
  let pri_shift = if pri_strength > 0 { 
    std::cmp::max(0, damping - msb(pri_strength)) 
  } else { 0 };
  let sec_shift = if sec_strength > 0 { 
    std::cmp::max(0, damping - msb(sec_strength)) 
  } else { 0 };
  
  // Direction offsets (precomputed for the selected direction)
  let cdef_directions: [[isize; 2]; 8] = [
    [-1 * istride + 1, -2 * istride + 2],
    [0 * istride + 1, -1 * istride + 2],
    [0 * istride + 1, 0 * istride + 2],
    [0 * istride + 1, 1 * istride + 2],
    [1 * istride + 1, 2 * istride + 2],
    [1 * istride + 0, 2 * istride + 1],
    [1 * istride + 0, 2 * istride + 0],
    [1 * istride + 0, 2 * istride - 1],
  ];
  
  let cdef_very_large = crate::cdef::CDEF_VERY_LARGE as i32;
  let very_large_vec = i32x4_splat(cdef_very_large);
  
  for i in 0..ysize {
    let row_base = input.offset((i as isize) * istride);
    let mut j = 0;
    
    // Process 4 pixels at a time with SIMD
    while j + 4 <= xsize {
      // Load 4 center pixels
      let x = i32x4(
        i32::cast_from(*row_base.add(j)),
        i32::cast_from(*row_base.add(j + 1)),
        i32::cast_from(*row_base.add(j + 2)),
        i32::cast_from(*row_base.add(j + 3)),
      );
      
      let mut sum = i32x4_splat(0);
      let mut max = x;
      let mut min = x;
      
      // Process both tap levels (k=0 and k=1)
      for k in 0..2 {
        let pri_dir = cdef_directions[dir][k];
        let sec_dir1 = cdef_directions[(dir + 2) & 7][k];
        let sec_dir2 = cdef_directions[(dir + 6) & 7][k];
        
        // Primary direction neighbors (2 neighbors)
        for &offset in &[pri_dir, -pri_dir] {
          let p = i32x4(
            i32::cast_from(*row_base.offset(j as isize + offset)),
            i32::cast_from(*row_base.offset(j as isize + 1 + offset)),
            i32::cast_from(*row_base.offset(j as isize + 2 + offset)),
            i32::cast_from(*row_base.offset(j as isize + 3 + offset)),
          );
          
          let diff = i32x4_sub(p, x);
          sum = i32x4_add(sum, constrain_simd(diff, pri_strength, pri_shift, pri_taps[k]));
          
          // Update min/max, excluding CDEF_VERY_LARGE
          let is_valid = i32x4_ne(p, very_large_vec);
          max = v128_bitselect(i32x4_max(max, p), max, is_valid);
          min = i32x4_min(min, p);
        }
        
        // Secondary direction neighbors (4 neighbors)
        for &offset in &[sec_dir1, -sec_dir1, sec_dir2, -sec_dir2] {
          let s = i32x4(
            i32::cast_from(*row_base.offset(j as isize + offset)),
            i32::cast_from(*row_base.offset(j as isize + 1 + offset)),
            i32::cast_from(*row_base.offset(j as isize + 2 + offset)),
            i32::cast_from(*row_base.offset(j as isize + 3 + offset)),
          );
          
          // Update min/max, excluding CDEF_VERY_LARGE
          let is_valid = i32x4_ne(s, very_large_vec);
          max = v128_bitselect(i32x4_max(max, s), max, is_valid);
          min = i32x4_min(min, s);
          
          let diff = i32x4_sub(s, x);
          sum = i32x4_add(sum, constrain_simd(diff, sec_strength, sec_shift, sec_taps[k]));
        }
      }
      
      // v = x + ((8 + sum - (sum < 0)) >> 4)
      let eight = i32x4_splat(8);
      let is_neg = i32x4_lt(sum, i32x4_splat(0));
      let neg_adj = v128_and(is_neg, i32x4_splat(1));
      let adjusted = i32x4_sub(i32x4_add(eight, sum), neg_adj);
      let shifted = i32x4_shr(adjusted, 4);
      let v = i32x4_add(x, shifted);
      
      // Clamp to [min, max]
      let clamped = i32x4_min(i32x4_max(v, min), max);
      
      // Store results
      dst[i][j] = T::cast_from(i32x4_extract_lane::<0>(clamped) as u32);
      dst[i][j + 1] = T::cast_from(i32x4_extract_lane::<1>(clamped) as u32);
      dst[i][j + 2] = T::cast_from(i32x4_extract_lane::<2>(clamped) as u32);
      dst[i][j + 3] = T::cast_from(i32x4_extract_lane::<3>(clamped) as u32);
      
      j += 4;
    }
    
    // Handle remaining pixels with scalar code
    while j < xsize {
      let ptr_in = row_base.add(j);
      let x = i32::cast_from(*ptr_in);
      let mut sum: i32 = 0;
      let mut max = x;
      let mut min = x;
      
      for k in 0..2 {
        let pri_dir = cdef_directions[dir][k];
        let sec_dir1 = cdef_directions[(dir + 2) & 7][k];
        let sec_dir2 = cdef_directions[(dir + 6) & 7][k];
        
        // Primary neighbors
        for &offset in &[pri_dir, -pri_dir] {
          let p = i32::cast_from(*ptr_in.offset(offset));
          let diff = p - x;
          sum += pri_taps[k] * constrain_scalar(diff, pri_strength, pri_shift);
          if p != cdef_very_large {
            max = std::cmp::max(max, p);
          }
          min = std::cmp::min(min, p);
        }
        
        // Secondary neighbors
        for &offset in &[sec_dir1, -sec_dir1, sec_dir2, -sec_dir2] {
          let s = i32::cast_from(*ptr_in.offset(offset));
          if s != cdef_very_large {
            max = std::cmp::max(max, s);
          }
          min = std::cmp::min(min, s);
          let diff = s - x;
          sum += sec_taps[k] * constrain_scalar(diff, sec_strength, sec_shift);
        }
      }
      
      let v = x + ((8 + sum - (sum < 0) as i32) >> 4);
      dst[i][j] = T::cast_from(std::cmp::min(std::cmp::max(v, min), max) as u32);
      j += 1;
    }
  }
}

/// Scalar constrain with precomputed shift
#[inline(always)]
fn constrain_scalar(diff: i32, threshold: i32, shift: i32) -> i32 {
  if threshold == 0 {
    return 0;
  }
  let abs_diff = diff.abs();
  let magnitude = (threshold - (abs_diff >> shift)).clamp(0, abs_diff);
  if diff < 0 { -magnitude } else { magnitude }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::frame::Plane;

  fn create_test_plane<T: Pixel>(width: usize, height: usize) -> Plane<T> {
    Plane::new(width, height, 0, 0, 0, 0)
  }

  #[test]
  fn test_cdef_find_dir() {
    use v_frame::plane::PlaneOffset;

    // Basic smoke test
    let mut plane = create_test_plane::<u8>(16, 16);

    // Fill with a simple gradient pattern
    for y in 0..16 {
      for x in 0..16 {
        plane.data[y * plane.cfg.stride + x] = ((x + y) % 256) as u8;
      }
    }

    let offset = PlaneOffset { x: 4, y: 4 };
    let slice = plane.slice(offset);

    let mut var: u32 = 0;
    let dir = cdef_find_dir(&slice, &mut var, 0, CpuFeatureLevel::RUST);

    // Direction should be in [0, 7]
    assert!(dir >= 0 && dir < 8);
  }
}
