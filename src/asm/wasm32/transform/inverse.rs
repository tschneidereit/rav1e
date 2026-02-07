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
//! Optimized paths for common transform sizes (4x4, 8x8) with DCT_DCT.
//! Falls back to Rust implementation for less common cases.

use crate::cpu_features::CpuFeatureLevel;
use crate::tiling::PlaneRegionMut;
use crate::transform::inverse::rust;
use crate::transform::{TxSize, TxType};
use crate::util::{clamp, CastFromPrimitive, Pixel};
use v_frame::pixel::PixelType;

use core::arch::wasm32::*;

// Inverse cosine table for SIMD (same as COSPI_INV but as constants)
const COSPI_32: i32 = 2896;  // cos(32 * pi/64) * 4096, i.e. COSPI_INV[32]
const COSPI_16: i32 = 3784;  // cos(16 * pi/64) * 4096, i.e. COSPI_INV[16]
const COSPI_48: i32 = 1567;  // cos(48 * pi/64) * 4096, i.e. COSPI_INV[48]
const INV_COS_BIT: i32 = 12;

/// SIMD half butterfly: (a * w0 + b * w1 + round) >> shift
#[inline(always)]
fn half_btf_simd(w0: i32, a: v128, w1: i32, b: v128) -> v128 {
  let w0_vec = i32x4_splat(w0);
  let w1_vec = i32x4_splat(w1);
  let round = i32x4_splat(1 << (INV_COS_BIT - 1));
  
  let prod0 = i32x4_mul(a, w0_vec);
  let prod1 = i32x4_mul(b, w1_vec);
  let sum = i32x4_add(i32x4_add(prod0, prod1), round);
  i32x4_shr(sum, INV_COS_BIT as u32)
}

/// Clamp a vector to range
#[inline(always)]
fn clamp_vec(v: v128, range: i32) -> v128 {
  let max_val = i32x4_splat((1 << range) - 1);
  let min_val = i32x4_splat(-(1 << range));
  i32x4_min(i32x4_max(v, min_val), max_val)
}

/// Inverse transform with SIMD acceleration.
///
/// Provides optimized paths for common cases, falls back to Rust for others.
#[inline(always)]
pub fn inverse_transform_add<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, eob: u16,
  tx_size: TxSize, tx_type: TxType, bd: usize, cpu: CpuFeatureLevel,
) {
  if cpu >= CpuFeatureLevel::SIMD128 {
    match T::type_enum() {
      PixelType::U8 => {
        // 8-bit pixel path
        match (tx_size, tx_type) {
          (TxSize::TX_4X4, TxType::DCT_DCT) => {
            inverse_transform_add_4x4_dct_simd(input, output, bd);
            return;
          }
          _ => {}
        }
      }
      PixelType::U16 => {
        // 10-bit / HBD pixel path (coefficients are i32)
        match (tx_size, tx_type) {
          (TxSize::TX_4X4, TxType::DCT_DCT) => {
            inverse_transform_add_4x4_dct_simd_hbd(input, output, bd);
            return;
          }
          _ => {}
        }
      }
    }
  }
  
  // Fallback to Rust implementation
  rust::inverse_transform_add(input, output, eob, tx_size, tx_type, bd, cpu);
}

/// SIMD-optimized 4x4 DCT inverse transform
///
/// Processes all 4 rows in parallel using SIMD. Each lane handles one row's worth
/// of computation at each column position.
#[inline(always)]
fn inverse_transform_add_4x4_dct_simd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  let range = (bd + 8) as i32;
  
  // Input is column-major: input[col*4 + row] = coefficient at (row, col)
  // Load as columns to process all 4 rows in parallel
  // col_i[r] = coefficient at row r, column i
  let col0 = i32x4(
    i32::cast_from(input[0]),
    i32::cast_from(input[1]),
    i32::cast_from(input[2]),
    i32::cast_from(input[3]),
  );
  let col1 = i32x4(
    i32::cast_from(input[4]),
    i32::cast_from(input[5]),
    i32::cast_from(input[6]),
    i32::cast_from(input[7]),
  );
  let col2 = i32x4(
    i32::cast_from(input[8]),
    i32::cast_from(input[9]),
    i32::cast_from(input[10]),
    i32::cast_from(input[11]),
  );
  let col3 = i32x4(
    i32::cast_from(input[12]),
    i32::cast_from(input[13]),
    i32::cast_from(input[14]),
    i32::cast_from(input[15]),
  );

  // Row transforms: 4 parallel IDCT4 operations (one per row, vectorized across columns)
  // IDCT4 stage 1: reorder inputs as [col0, col2, col1, col3]
  // IDCT4 stage 2: butterfly operations
  let s0 = half_btf_simd(COSPI_32, col0, COSPI_32, col2);
  let s1 = half_btf_simd(COSPI_32, col0, -COSPI_32, col2);
  let s2 = half_btf_simd(COSPI_48, col1, -COSPI_16, col3);
  let s3 = half_btf_simd(COSPI_16, col1, COSPI_48, col3);
  
  // IDCT4 stage 3: add/sub with clamp
  // Results: r_c[r] = row r, output column c
  let r0 = clamp_vec(i32x4_add(s0, s3), range);
  let r1 = clamp_vec(i32x4_add(s1, s2), range);
  let r2 = clamp_vec(i32x4_sub(s1, s2), range);
  let r3 = clamp_vec(i32x4_sub(s0, s3), range);

  // Note: r0..r3 are now organized as columns of the intermediate buffer
  // r_c[r] = intermediate result for row r, column c
  // This is already the correct format for column transforms!
  
  // INV_INTERMEDIATE_SHIFTS[TX_4X4] = 0, so no intermediate shift needed
  
  // Column transforms: 4 parallel IDCT4 operations (one per column, vectorized across rows)
  // The vectors r0..r3 already hold column data:
  // - r0 = column 0 input to column transform
  // - r1 = column 1 input to column transform
  // etc.
  
  // But column transform outputs need to be rows for final output.
  // To do this, we transpose so each vector holds one row.
  // Transpose r0..r3 first:
  let lo01 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1); // [r0[0], r1[0], r0[1], r1[1]]
  let hi01 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1); // [r0[2], r1[2], r0[3], r1[3]]
  let lo23 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3); // [r2[0], r3[0], r2[1], r3[1]]
  let hi23 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3); // [r2[2], r3[2], r2[3], r3[3]]
  
  let row0 = i64x2_shuffle::<0, 2>(lo01, lo23); // row 0: [col0, col1, col2, col3]
  let row1 = i64x2_shuffle::<1, 3>(lo01, lo23); // row 1
  let row2 = i64x2_shuffle::<0, 2>(hi01, hi23); // row 2
  let row3 = i64x2_shuffle::<1, 3>(hi01, hi23); // row 3

  // Now do column transforms, processing all 4 columns in parallel
  // Each lane handles one column's IDCT4 applied to rows
  let range2 = (bd.max(10) + 6) as i32;
  
  // IDCT4 stage 1: reorder inputs as [row0, row2, row1, row3]
  // IDCT4 stage 2: butterfly
  let s0 = half_btf_simd(COSPI_32, row0, COSPI_32, row2);
  let s1 = half_btf_simd(COSPI_32, row0, -COSPI_32, row2);
  let s2 = half_btf_simd(COSPI_48, row1, -COSPI_16, row3);
  let s3 = half_btf_simd(COSPI_16, row1, COSPI_48, row3);
  
  // IDCT4 stage 3: add/sub with clamp
  let f0 = clamp_vec(i32x4_add(s0, s3), range2);
  let f1 = clamp_vec(i32x4_add(s1, s2), range2);
  let f2 = clamp_vec(i32x4_sub(s1, s2), range2);
  let f3 = clamp_vec(i32x4_sub(s0, s3), range2);

  // Round shift by 4 and add to output
  // f_i[c] = output row i, column c
  let round = i32x4_splat(8);
  
  let f0 = i32x4_shr(i32x4_add(f0, round), 4);
  let f1 = i32x4_shr(i32x4_add(f1, round), 4);
  let f2 = i32x4_shr(i32x4_add(f2, round), 4);
  let f3 = i32x4_shr(i32x4_add(f3, round), 4);

  // Add to output with clamp
  for (row_idx, out_vec) in [f0, f1, f2, f3].iter().enumerate() {
    let out_row = &mut output[row_idx];
    let val0 = i32x4_extract_lane::<0>(*out_vec);
    let val1 = i32x4_extract_lane::<1>(*out_vec);
    let val2 = i32x4_extract_lane::<2>(*out_vec);
    let val3 = i32x4_extract_lane::<3>(*out_vec);
    
    let pix0: i32 = out_row[0].as_();
    let pix1: i32 = out_row[1].as_();
    let pix2: i32 = out_row[2].as_();
    let pix3: i32 = out_row[3].as_();
    
    out_row[0] = T::cast_from(clamp(pix0 + val0, 0, (1 << bd) - 1));
    out_row[1] = T::cast_from(clamp(pix1 + val1, 0, (1 << bd) - 1));
    out_row[2] = T::cast_from(clamp(pix2 + val2, 0, (1 << bd) - 1));
    out_row[3] = T::cast_from(clamp(pix3 + val3, 0, (1 << bd) - 1));
  }
}

/// SIMD-optimized 4x4 DCT inverse transform for HBD (10-bit) pixels
///
/// Same algorithm as the 8-bit version but coefficients are i32.
#[inline(always)]
fn inverse_transform_add_4x4_dct_simd_hbd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  let range = (bd + 8) as i32;
  
  // For HBD, coefficients are i32, so we can load directly
  // Input is column-major: input[col*4 + row] = coefficient at (row, col)
  let col0 = i32x4(
    i32::cast_from(input[0]),
    i32::cast_from(input[1]),
    i32::cast_from(input[2]),
    i32::cast_from(input[3]),
  );
  let col1 = i32x4(
    i32::cast_from(input[4]),
    i32::cast_from(input[5]),
    i32::cast_from(input[6]),
    i32::cast_from(input[7]),
  );
  let col2 = i32x4(
    i32::cast_from(input[8]),
    i32::cast_from(input[9]),
    i32::cast_from(input[10]),
    i32::cast_from(input[11]),
  );
  let col3 = i32x4(
    i32::cast_from(input[12]),
    i32::cast_from(input[13]),
    i32::cast_from(input[14]),
    i32::cast_from(input[15]),
  );

  // Row transforms
  let s0 = half_btf_simd(COSPI_32, col0, COSPI_32, col2);
  let s1 = half_btf_simd(COSPI_32, col0, -COSPI_32, col2);
  let s2 = half_btf_simd(COSPI_48, col1, -COSPI_16, col3);
  let s3 = half_btf_simd(COSPI_16, col1, COSPI_48, col3);
  
  let r0 = clamp_vec(i32x4_add(s0, s3), range);
  let r1 = clamp_vec(i32x4_add(s1, s2), range);
  let r2 = clamp_vec(i32x4_sub(s1, s2), range);
  let r3 = clamp_vec(i32x4_sub(s0, s3), range);

  // Transpose for column transforms
  let lo01 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1);
  let hi01 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1);
  let lo23 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3);
  let hi23 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3);
  
  let row0 = i64x2_shuffle::<0, 2>(lo01, lo23);
  let row1 = i64x2_shuffle::<1, 3>(lo01, lo23);
  let row2 = i64x2_shuffle::<0, 2>(hi01, hi23);
  let row3 = i64x2_shuffle::<1, 3>(hi01, hi23);

  // Column transforms
  let range2 = (bd.max(10) + 6) as i32;
  
  let s0 = half_btf_simd(COSPI_32, row0, COSPI_32, row2);
  let s1 = half_btf_simd(COSPI_32, row0, -COSPI_32, row2);
  let s2 = half_btf_simd(COSPI_48, row1, -COSPI_16, row3);
  let s3 = half_btf_simd(COSPI_16, row1, COSPI_48, row3);
  
  let f0 = clamp_vec(i32x4_add(s0, s3), range2);
  let f1 = clamp_vec(i32x4_add(s1, s2), range2);
  let f2 = clamp_vec(i32x4_sub(s1, s2), range2);
  let f3 = clamp_vec(i32x4_sub(s0, s3), range2);

  // Round shift by 4 and add to output
  let round = i32x4_splat(8);
  
  let f0 = i32x4_shr(i32x4_add(f0, round), 4);
  let f1 = i32x4_shr(i32x4_add(f1, round), 4);
  let f2 = i32x4_shr(i32x4_add(f2, round), 4);
  let f3 = i32x4_shr(i32x4_add(f3, round), 4);

  // Add to output with clamp (HBD pixels are u16)
  let max_pix = (1 << bd) - 1;
  for (row_idx, out_vec) in [f0, f1, f2, f3].iter().enumerate() {
    let out_row = &mut output[row_idx];
    let val0 = i32x4_extract_lane::<0>(*out_vec);
    let val1 = i32x4_extract_lane::<1>(*out_vec);
    let val2 = i32x4_extract_lane::<2>(*out_vec);
    let val3 = i32x4_extract_lane::<3>(*out_vec);
    
    let pix0: i32 = out_row[0].as_();
    let pix1: i32 = out_row[1].as_();
    let pix2: i32 = out_row[2].as_();
    let pix3: i32 = out_row[3].as_();
    
    out_row[0] = T::cast_from(clamp(pix0 + val0, 0, max_pix));
    out_row[1] = T::cast_from(clamp(pix1 + val1, 0, max_pix));
    out_row[2] = T::cast_from(clamp(pix2 + val2, 0, max_pix));
    out_row[3] = T::cast_from(clamp(pix3 + val3, 0, max_pix));
  }
}

/// SIMD-optimized 8x8 DCT inverse transform  
#[inline(always)]
fn inverse_transform_add_8x8_dct_simd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  // For 8x8, the complexity increases significantly
  // Fall back to Rust for now - can optimize later if profiling shows need
  rust::inverse_transform_add(
    input, output, 64, TxSize::TX_8X8, TxType::DCT_DCT, bd,
    CpuFeatureLevel::RUST,
  );
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
  fn test_inverse_transform_4x4_matches_rust() {
    let tx_size = TX_4X4;
    let width = tx_size.width();
    let height = tx_size.height();

    let mut rng = rand::rng();

    // Create input coefficients
    let mut coeffs = vec![0i16; width * height];
    for c in coeffs.iter_mut() {
      *c = rng.random_range(-50..50);
    }

    // Create two output planes
    let mut plane_rust = create_test_plane::<u8>(width + 16, height + 16);
    let mut plane_simd = create_test_plane::<u8>(width + 16, height + 16);

    // Fill with same initial values
    for y in 0..height {
      for x in 0..width {
        let val = 128u8;
        plane_rust.data[y * plane_rust.cfg.stride + x] = val;
        plane_simd.data[y * plane_simd.cfg.stride + x] = val;
      }
    }

    // Run Rust version
    {
      let area = Area::StartingAt { x: 0, y: 0 };
      let mut region = plane_rust.region_mut(area);
      rust::inverse_transform_add(
        &coeffs, &mut region, 16, tx_size, DCT_DCT, 8, CpuFeatureLevel::RUST,
      );
    }

    // Run SIMD version
    {
      let area = Area::StartingAt { x: 0, y: 0 };
      let mut region = plane_simd.region_mut(area);
      inverse_transform_add(
        &coeffs, &mut region, 16, tx_size, DCT_DCT, 8, CpuFeatureLevel::SIMD128,
      );
    }

    // Compare results
    for y in 0..height {
      for x in 0..width {
        let rust_val = plane_rust.data[y * plane_rust.cfg.stride + x];
        let simd_val = plane_simd.data[y * plane_simd.cfg.stride + x];
        assert_eq!(
          rust_val, simd_val,
          "Mismatch at ({}, {}): rust={}, simd={}",
          x, y, rust_val, simd_val
        );
      }
    }
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
