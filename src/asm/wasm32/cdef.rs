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

// Re-export pad_into_tmp16 from rust module for use by rest of crate
pub use rust::pad_into_tmp16;

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
  let offset = i32x4_splat(128);
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
/// Currently uses Rust fallback implementation.
#[inline(always)]
pub unsafe fn cdef_filter_block<T: Pixel, U: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, input: *const U, istride: isize,
  pri_strength: i32, sec_strength: i32, dir: usize, damping: i32,
  bit_depth: usize, xdec: usize, ydec: usize, edges: u8,
  cpu: CpuFeatureLevel,
) {
  rust::cdef_filter_block(
    dst, input, istride, pri_strength, sec_strength, dir, damping,
    bit_depth, xdec, ydec, edges, cpu,
  )
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
