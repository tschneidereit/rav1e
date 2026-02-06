// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! SIMD-accelerated inverse transform functions for wasm32.
//!
//! Note: The inverse transform is complex due to many transform sizes and types.
//! Currently, this module delegates to the Rust implementation while providing
//! the correct function signature for wasm32 SIMD builds.
//!
//! Future optimization opportunities:
//! - IDCT4/IDCT8 using SIMD for small transform sizes
//! - SIMD-accelerated output clamping and addition

use crate::cpu_features::CpuFeatureLevel;
use crate::tiling::PlaneRegionMut;
use crate::transform::inverse::rust;
use crate::transform::{TxSize, TxType};
use crate::util::Pixel;

/// Inverse transform with SIMD acceleration.
///
/// Performs the inverse 2D transform and adds the result to the output plane.
/// Currently delegates to the Rust implementation.
#[inline(always)]
pub fn inverse_transform_add<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, eob: u16,
  tx_size: TxSize, tx_type: TxType, bd: usize, cpu: CpuFeatureLevel,
) {
  // For now, use Rust fallback.
  // Future: Implement SIMD versions for common transform sizes (4x4, 8x8, 16x16)
  rust::inverse_transform_add(input, output, eob, tx_size, tx_type, bd, cpu);
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::frame::Plane;
  use crate::tiling::Area;
  use crate::transform::TxSize::*;
  use crate::transform::TxType::*;
  use rand::Rng;

  fn create_test_plane<T: Pixel>(width: usize, height: usize) -> Plane<T> {
    Plane::new(width, height, 0, 0, 0, 0)
  }

  #[test]
  fn test_inverse_transform_basic() {
    // Smoke test to ensure the inverse transform wrapper compiles and runs
    let tx_size = TX_4X4;
    let width = tx_size.width();
    let height = tx_size.height();

    let mut rng = rand::rng();

    // Create input coefficients (small values to avoid overflow)
    let mut coeffs = vec![0i16; width * height];
    coeffs[0] = rng.random_range(-100..100); // DC coefficient

    // Create output plane
    let mut plane = create_test_plane::<u8>(width + 16, height + 16);

    // Fill with some values
    for y in 0..height {
      for x in 0..width {
        plane.data[y * plane.cfg.stride + x] = 128;
      }
    }

    {
      let area = Area::StartingAt { x: 0, y: 0 };
      let mut region = plane.region_mut(area);

      inverse_transform_add(
        &coeffs,
        &mut region,
        1, // eob
        tx_size,
        DCT_DCT,
        8,
        CpuFeatureLevel::SIMD128,
      );
    }

    // Just verify it didn't panic
  }
}
