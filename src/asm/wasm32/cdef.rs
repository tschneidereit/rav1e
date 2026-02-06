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
//! Note: Currently uses Rust fallback. CDEF filtering involves complex
//! data-dependent operations (min/max tracking, direction-based sampling)
//! that are challenging to vectorize efficiently in WASM SIMD.
//!
//! Future optimization opportunities:
//! - cdef_find_dir: partial sum computation could benefit from SIMD
//! - cdef_filter_block: inner loop constrain() could be vectorized

use crate::cdef::rust;
use crate::cpu_features::CpuFeatureLevel;
use crate::frame::*;
use crate::tiling::PlaneRegionMut;
use crate::util::Pixel;

/// Find the optimal filtering direction for an 8x8 block.
///
/// Currently uses Rust fallback implementation.
#[inline(always)]
pub fn cdef_find_dir<T: Pixel>(
  img: &PlaneSlice<'_, T>, var: &mut u32, coeff_shift: usize,
  cpu: CpuFeatureLevel,
) -> i32 {
  rust::cdef_find_dir(img, var, coeff_shift, cpu)
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

pub use rust::pad_into_tmp16;

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
