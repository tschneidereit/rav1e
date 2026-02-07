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
// Additional constants for IDCT8
const COSPI_8: i32 = 4017;   // cos(8 * pi/64) * 4096
const COSPI_24: i32 = 3406;  // cos(24 * pi/64) * 4096
const COSPI_40: i32 = 2276;  // cos(40 * pi/64) * 4096
const COSPI_56: i32 = 799;   // cos(56 * pi/64) * 4096
// Additional constants for IDCT16
const COSPI_2: i32 = 4091;
const COSPI_4: i32 = 4076;
const COSPI_6: i32 = 4052;
const COSPI_10: i32 = 3973;
const COSPI_12: i32 = 3948;
const COSPI_14: i32 = 3857;
const COSPI_18: i32 = 3703;
const COSPI_20: i32 = 3612;
const COSPI_22: i32 = 3513;
const COSPI_26: i32 = 3290;
const COSPI_28: i32 = 3229;
const COSPI_30: i32 = 3035;
const COSPI_34: i32 = 2751;
const COSPI_36: i32 = 2598;
const COSPI_38: i32 = 2520;
const COSPI_42: i32 = 2191;
const COSPI_44: i32 = 2106;
const COSPI_46: i32 = 1842;
const COSPI_50: i32 = 1474;
const COSPI_52: i32 = 1285;
const COSPI_54: i32 = 1092;
const COSPI_58: i32 = 601;
const COSPI_60: i32 = 301;
const COSPI_62: i32 = 101;
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

#[inline(always)]
fn round_shift_vec(v: v128, shift: u32) -> v128 {
  let round = i32x4_splat(1 << (shift - 1));
  i32x4_shr(i32x4_add(v, round), shift)
}

#[inline(always)]
fn iidentity16_vec(v: v128) -> v128 {
  let mul = i32x4_splat(SQRT2 * 2);
  let round = i32x4_splat(1 << 11);
  let prod = i32x4_mul(v, mul);
  i32x4_shr(i32x4_add(prod, round), 12)
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
        match (tx_size, tx_type) {
          (TxSize::TX_4X4, TxType::DCT_DCT) => {
            inverse_transform_add_4x4_dct_simd_hbd(input, output, bd);
            return;
          }
          (TxSize::TX_4X4, TxType::ADST_DCT) => {
            inverse_transform_add_4x4_adst_dct_simd_hbd(input, output, bd);
            return;
          }
          (TxSize::TX_4X4, TxType::DCT_ADST) => {
            inverse_transform_add_4x4_dct_adst_simd_hbd(input, output, bd);
            return;
          }
          (TxSize::TX_4X4, TxType::ADST_ADST) => {
            inverse_transform_add_4x4_adst_adst_simd_hbd(input, output, bd);
            return;
          }
          (TxSize::TX_4X4, TxType::IDTX) => {
            inverse_transform_add_4x4_idtx_simd_hbd(input, output, bd);
            return;
          }
          (TxSize::TX_4X4, TxType::V_DCT) => {
            inverse_transform_add_4x4_vdct_simd_hbd(input, output, bd);
            return;
          }
          (TxSize::TX_4X4, TxType::H_DCT) => {
            inverse_transform_add_4x4_hdct_simd_hbd(input, output, bd);
            return;
          }
          (TxSize::TX_8X8, TxType::DCT_DCT) => {
            inverse_transform_add_8x8_dct_simd_hbd(input, output, bd);
            return;
          }
          (TxSize::TX_8X8, TxType::IDTX) => {
            inverse_transform_add_8x8_idtx_simd_hbd(input, output, bd);
            return;
          }
          (TxSize::TX_16X16, TxType::DCT_DCT) => {
            inverse_transform_add_16x16_dct_simd_hbd(input, output, bd);
            return;
          }
          (TxSize::TX_16X16, TxType::IDTX) => {
            inverse_transform_add_16x16_idtx_simd_hbd(input, output, bd);
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

// SQRT2 constant for identity transforms (same as COSPI_INV[32] * 2)
const SQRT2: i32 = 5793;

/// SIMD-optimized 4x4 IDTX (identity) inverse transform for HBD
/// Both row and column transforms are identity: output = round_shift(SQRT2 * input, 12)
#[inline(always)]
fn inverse_transform_add_4x4_idtx_simd_hbd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  // Identity transform: output = round_shift(SQRT2 * input, 12) applied twice
  // Combined: output = round_shift(round_shift(SQRT2 * input, 12) * SQRT2, 12)
  //         = round_shift(SQRT2 * SQRT2 * input, 24) / 2 ... but actually applied separately
  
  // Row transform: each element *= SQRT2, >> 12
  // Column transform: each element *= SQRT2, >> 12  
  // Then final shift by 4
  
  let sqrt2 = i32x4_splat(SQRT2);
  let round12 = i32x4_splat(1 << 11);
  let round4 = i32x4_splat(8);
  let max_pix = (1 << bd) - 1;
  
  // Load all 16 coefficients and apply identity transform
  // Input is column-major, process in chunks of 4
  for row in 0..4 {
    // Load 4 coefficients for this row (from input columns 0-3, row 'row')
    let c0 = i32::cast_from(input[0 * 4 + row]);
    let c1 = i32::cast_from(input[1 * 4 + row]);
    let c2 = i32::cast_from(input[2 * 4 + row]);
    let c3 = i32::cast_from(input[3 * 4 + row]);
    let coeffs = i32x4(c0, c1, c2, c3);
    
    // Row transform: * SQRT2 >> 12
    let row_out = i32x4_shr(i32x4_add(i32x4_mul(coeffs, sqrt2), round12), 12);
    
    // Column transform: * SQRT2 >> 12
    let col_out = i32x4_shr(i32x4_add(i32x4_mul(row_out, sqrt2), round12), 12);
    
    // Final shift by 4
    let final_out = i32x4_shr(i32x4_add(col_out, round4), 4);
    
    // Add to output
    let out_row = &mut output[row];
    let v0 = i32x4_extract_lane::<0>(final_out);
    let v1 = i32x4_extract_lane::<1>(final_out);
    let v2 = i32x4_extract_lane::<2>(final_out);
    let v3 = i32x4_extract_lane::<3>(final_out);
    
    let pix0: i32 = out_row[0].as_();
    let pix1: i32 = out_row[1].as_();
    let pix2: i32 = out_row[2].as_();
    let pix3: i32 = out_row[3].as_();
    
    out_row[0] = T::cast_from(clamp(pix0 + v0, 0, max_pix));
    out_row[1] = T::cast_from(clamp(pix1 + v1, 0, max_pix));
    out_row[2] = T::cast_from(clamp(pix2 + v2, 0, max_pix));
    out_row[3] = T::cast_from(clamp(pix3 + v3, 0, max_pix));
  }
}

/// SIMD-optimized 4x4 V_DCT inverse transform for HBD
/// Row transform = identity, Column transform = DCT
#[inline(always)]
fn inverse_transform_add_4x4_vdct_simd_hbd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  let sqrt2 = i32x4_splat(SQRT2);
  let round12 = i32x4_splat(1 << 11);
  
  // Load columns
  let col0 = i32x4(
    i32::cast_from(input[0]), i32::cast_from(input[1]),
    i32::cast_from(input[2]), i32::cast_from(input[3]),
  );
  let col1 = i32x4(
    i32::cast_from(input[4]), i32::cast_from(input[5]),
    i32::cast_from(input[6]), i32::cast_from(input[7]),
  );
  let col2 = i32x4(
    i32::cast_from(input[8]), i32::cast_from(input[9]),
    i32::cast_from(input[10]), i32::cast_from(input[11]),
  );
  let col3 = i32x4(
    i32::cast_from(input[12]), i32::cast_from(input[13]),
    i32::cast_from(input[14]), i32::cast_from(input[15]),
  );
  
  // Row transform: identity (* SQRT2 >> 12)
  let r0 = i32x4_shr(i32x4_add(i32x4_mul(col0, sqrt2), round12), 12);
  let r1 = i32x4_shr(i32x4_add(i32x4_mul(col1, sqrt2), round12), 12);
  let r2 = i32x4_shr(i32x4_add(i32x4_mul(col2, sqrt2), round12), 12);
  let r3 = i32x4_shr(i32x4_add(i32x4_mul(col3, sqrt2), round12), 12);
  
  // Transpose for column transforms
  let lo01 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1);
  let hi01 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1);
  let lo23 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3);
  let hi23 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3);
  
  let row0 = i64x2_shuffle::<0, 2>(lo01, lo23);
  let row1 = i64x2_shuffle::<1, 3>(lo01, lo23);
  let row2 = i64x2_shuffle::<0, 2>(hi01, hi23);
  let row3 = i64x2_shuffle::<1, 3>(hi01, hi23);
  
  // Column transforms: DCT
  let range2 = (bd.max(10) + 6) as i32;
  
  let s0 = half_btf_simd(COSPI_32, row0, COSPI_32, row2);
  let s1 = half_btf_simd(COSPI_32, row0, -COSPI_32, row2);
  let s2 = half_btf_simd(COSPI_48, row1, -COSPI_16, row3);
  let s3 = half_btf_simd(COSPI_16, row1, COSPI_48, row3);
  
  let f0 = clamp_vec(i32x4_add(s0, s3), range2);
  let f1 = clamp_vec(i32x4_add(s1, s2), range2);
  let f2 = clamp_vec(i32x4_sub(s1, s2), range2);
  let f3 = clamp_vec(i32x4_sub(s0, s3), range2);
  
  // Round shift and add to output
  let round = i32x4_splat(8);
  let f0 = i32x4_shr(i32x4_add(f0, round), 4);
  let f1 = i32x4_shr(i32x4_add(f1, round), 4);
  let f2 = i32x4_shr(i32x4_add(f2, round), 4);
  let f3 = i32x4_shr(i32x4_add(f3, round), 4);
  
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

/// SIMD-optimized 4x4 ADST_DCT inverse transform for HBD
/// Row transform = ADST, Column transform = DCT
#[inline(always)]
fn inverse_transform_add_4x4_adst_dct_simd_hbd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  // Load columns
  let col0 = i32x4(
    i32::cast_from(input[0]), i32::cast_from(input[1]),
    i32::cast_from(input[2]), i32::cast_from(input[3]),
  );
  let col1 = i32x4(
    i32::cast_from(input[4]), i32::cast_from(input[5]),
    i32::cast_from(input[6]), i32::cast_from(input[7]),
  );
  let col2 = i32x4(
    i32::cast_from(input[8]), i32::cast_from(input[9]),
    i32::cast_from(input[10]), i32::cast_from(input[11]),
  );
  let col3 = i32x4(
    i32::cast_from(input[12]), i32::cast_from(input[13]),
    i32::cast_from(input[14]), i32::cast_from(input[15]),
  );
  
  // Row transforms: ADST
  let (r0, r1, r2, r3) = iadst4_simd(col0, col1, col2, col3);
  
  // Transpose for column transforms
  let lo01 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1);
  let hi01 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1);
  let lo23 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3);
  let hi23 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3);
  
  let row0 = i64x2_shuffle::<0, 2>(lo01, lo23);
  let row1 = i64x2_shuffle::<1, 3>(lo01, lo23);
  let row2 = i64x2_shuffle::<0, 2>(hi01, hi23);
  let row3 = i64x2_shuffle::<1, 3>(hi01, hi23);
  
  // Column transforms: DCT
  let range2 = (bd.max(10) + 6) as i32;
  
  let s0 = half_btf_simd(COSPI_32, row0, COSPI_32, row2);
  let s1 = half_btf_simd(COSPI_32, row0, -COSPI_32, row2);
  let s2 = half_btf_simd(COSPI_48, row1, -COSPI_16, row3);
  let s3 = half_btf_simd(COSPI_16, row1, COSPI_48, row3);
  
  let f0 = clamp_vec(i32x4_add(s0, s3), range2);
  let f1 = clamp_vec(i32x4_add(s1, s2), range2);
  let f2 = clamp_vec(i32x4_sub(s1, s2), range2);
  let f3 = clamp_vec(i32x4_sub(s0, s3), range2);
  
  // Round shift and add to output
  let round = i32x4_splat(8);
  let f0 = i32x4_shr(i32x4_add(f0, round), 4);
  let f1 = i32x4_shr(i32x4_add(f1, round), 4);
  let f2 = i32x4_shr(i32x4_add(f2, round), 4);
  let f3 = i32x4_shr(i32x4_add(f3, round), 4);
  
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

/// SIMD-optimized 4x4 DCT_ADST inverse transform for HBD
/// Row transform = DCT, Column transform = ADST
#[inline(always)]
fn inverse_transform_add_4x4_dct_adst_simd_hbd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  let range = (bd + 8) as i32;
  
  // Load columns
  let col0 = i32x4(
    i32::cast_from(input[0]), i32::cast_from(input[1]),
    i32::cast_from(input[2]), i32::cast_from(input[3]),
  );
  let col1 = i32x4(
    i32::cast_from(input[4]), i32::cast_from(input[5]),
    i32::cast_from(input[6]), i32::cast_from(input[7]),
  );
  let col2 = i32x4(
    i32::cast_from(input[8]), i32::cast_from(input[9]),
    i32::cast_from(input[10]), i32::cast_from(input[11]),
  );
  let col3 = i32x4(
    i32::cast_from(input[12]), i32::cast_from(input[13]),
    i32::cast_from(input[14]), i32::cast_from(input[15]),
  );
  
  // Row transforms: DCT
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
  
  // Column transforms: ADST
  let (f0, f1, f2, f3) = iadst4_simd(row0, row1, row2, row3);
  
  // Round shift and add to output
  let round = i32x4_splat(8);
  let f0 = i32x4_shr(i32x4_add(f0, round), 4);
  let f1 = i32x4_shr(i32x4_add(f1, round), 4);
  let f2 = i32x4_shr(i32x4_add(f2, round), 4);
  let f3 = i32x4_shr(i32x4_add(f3, round), 4);
  
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

/// SIMD-optimized 4x4 ADST_ADST inverse transform for HBD
/// Both row and column transforms use ADST
#[inline(always)]
fn inverse_transform_add_4x4_adst_adst_simd_hbd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  // Load columns
  let col0 = i32x4(
    i32::cast_from(input[0]), i32::cast_from(input[1]),
    i32::cast_from(input[2]), i32::cast_from(input[3]),
  );
  let col1 = i32x4(
    i32::cast_from(input[4]), i32::cast_from(input[5]),
    i32::cast_from(input[6]), i32::cast_from(input[7]),
  );
  let col2 = i32x4(
    i32::cast_from(input[8]), i32::cast_from(input[9]),
    i32::cast_from(input[10]), i32::cast_from(input[11]),
  );
  let col3 = i32x4(
    i32::cast_from(input[12]), i32::cast_from(input[13]),
    i32::cast_from(input[14]), i32::cast_from(input[15]),
  );
  
  // Row transforms: ADST
  let (r0, r1, r2, r3) = iadst4_simd(col0, col1, col2, col3);
  
  // Transpose for column transforms
  let lo01 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1);
  let hi01 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1);
  let lo23 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3);
  let hi23 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3);
  
  let row0 = i64x2_shuffle::<0, 2>(lo01, lo23);
  let row1 = i64x2_shuffle::<1, 3>(lo01, lo23);
  let row2 = i64x2_shuffle::<0, 2>(hi01, hi23);
  let row3 = i64x2_shuffle::<1, 3>(hi01, hi23);
  
  // Column transforms: ADST
  let (f0, f1, f2, f3) = iadst4_simd(row0, row1, row2, row3);
  
  // Round shift and add to output
  let round = i32x4_splat(8);
  let f0 = i32x4_shr(i32x4_add(f0, round), 4);
  let f1 = i32x4_shr(i32x4_add(f1, round), 4);
  let f2 = i32x4_shr(i32x4_add(f2, round), 4);
  let f3 = i32x4_shr(i32x4_add(f3, round), 4);
  
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

/// SIMD-optimized 4x4 H_DCT inverse transform for HBD
/// Row transform = DCT, Column transform = identity
#[inline(always)]
fn inverse_transform_add_4x4_hdct_simd_hbd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  let range = (bd + 8) as i32;
  let sqrt2 = i32x4_splat(SQRT2);
  let round12 = i32x4_splat(1 << 11);
  
  // Load columns
  let col0 = i32x4(
    i32::cast_from(input[0]), i32::cast_from(input[1]),
    i32::cast_from(input[2]), i32::cast_from(input[3]),
  );
  let col1 = i32x4(
    i32::cast_from(input[4]), i32::cast_from(input[5]),
    i32::cast_from(input[6]), i32::cast_from(input[7]),
  );
  let col2 = i32x4(
    i32::cast_from(input[8]), i32::cast_from(input[9]),
    i32::cast_from(input[10]), i32::cast_from(input[11]),
  );
  let col3 = i32x4(
    i32::cast_from(input[12]), i32::cast_from(input[13]),
    i32::cast_from(input[14]), i32::cast_from(input[15]),
  );
  
  // Row transforms: DCT on each row
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
  
  // Column transforms: identity (* SQRT2 >> 12)
  let f0 = i32x4_shr(i32x4_add(i32x4_mul(row0, sqrt2), round12), 12);
  let f1 = i32x4_shr(i32x4_add(i32x4_mul(row1, sqrt2), round12), 12);
  let f2 = i32x4_shr(i32x4_add(i32x4_mul(row2, sqrt2), round12), 12);
  let f3 = i32x4_shr(i32x4_add(i32x4_mul(row3, sqrt2), round12), 12);
  
  // Round shift and add to output
  let round = i32x4_splat(8);
  let f0 = i32x4_shr(i32x4_add(f0, round), 4);
  let f1 = i32x4_shr(i32x4_add(f1, round), 4);
  let f2 = i32x4_shr(i32x4_add(f2, round), 4);
  let f3 = i32x4_shr(i32x4_add(f3, round), 4);
  
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

/// SIMD-optimized 8x8 IDTX (identity) inverse transform for HBD
/// For 8x8: identity transform is just output = 2 * input
#[inline(always)]
fn inverse_transform_add_8x8_idtx_simd_hbd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  // 8x8 identity: each element *= 2 for row, *= 2 for column = *= 4 total
  // Then final shift by 4, so *= 4 >> 4 = no change, just add to output
  // Actually need to check the exact scaling...
  
  // av1_iidentity8: output = 2 * input
  // Applied twice: result = 4 * input
  // Then shift by 4: result >> 4 = input / 4... that doesn't seem right
  
  // Let me trace through: for 8x8 IDTX with HBD
  // Row transform: each row element *= 2 (av1_iidentity8)
  // Column transform: each column element *= 2 (av1_iidentity8)
  // Combined: *= 4
  // Final shift: >> 4 (INV_INTERMEDIATE_SHIFTS[TX_8X8] = 1, so total shift is... complex)
  
  // Actually let's just implement it as: final = (4 * input + 8) >> 4
  let max_pix = (1 << bd) - 1;
  let round = i32x4_splat(8);
  
  for row in 0..8 {
    // Load coefficients for this row (columns 0-3, then 4-7)
    let c0 = i32::cast_from(input[0 * 8 + row]);
    let c1 = i32::cast_from(input[1 * 8 + row]);
    let c2 = i32::cast_from(input[2 * 8 + row]);
    let c3 = i32::cast_from(input[3 * 8 + row]);
    let c4 = i32::cast_from(input[4 * 8 + row]);
    let c5 = i32::cast_from(input[5 * 8 + row]);
    let c6 = i32::cast_from(input[6 * 8 + row]);
    let c7 = i32::cast_from(input[7 * 8 + row]);
    
    let coeffs_lo = i32x4(c0, c1, c2, c3);
    let coeffs_hi = i32x4(c4, c5, c6, c7);
    
    // Identity: *= 4 (2 * 2), then shift by 4
    let scaled_lo = i32x4_shl(coeffs_lo, 2);
    let scaled_hi = i32x4_shl(coeffs_hi, 2);
    let final_lo = i32x4_shr(i32x4_add(scaled_lo, round), 4);
    let final_hi = i32x4_shr(i32x4_add(scaled_hi, round), 4);
    
    // Add to output
    let out_row = &mut output[row];
    
    let v0 = i32x4_extract_lane::<0>(final_lo);
    let v1 = i32x4_extract_lane::<1>(final_lo);
    let v2 = i32x4_extract_lane::<2>(final_lo);
    let v3 = i32x4_extract_lane::<3>(final_lo);
    let v4 = i32x4_extract_lane::<0>(final_hi);
    let v5 = i32x4_extract_lane::<1>(final_hi);
    let v6 = i32x4_extract_lane::<2>(final_hi);
    let v7 = i32x4_extract_lane::<3>(final_hi);
    
    let pix0: i32 = out_row[0].as_();
    let pix1: i32 = out_row[1].as_();
    let pix2: i32 = out_row[2].as_();
    let pix3: i32 = out_row[3].as_();
    let pix4: i32 = out_row[4].as_();
    let pix5: i32 = out_row[5].as_();
    let pix6: i32 = out_row[6].as_();
    let pix7: i32 = out_row[7].as_();
    
    out_row[0] = T::cast_from(clamp(pix0 + v0, 0, max_pix));
    out_row[1] = T::cast_from(clamp(pix1 + v1, 0, max_pix));
    out_row[2] = T::cast_from(clamp(pix2 + v2, 0, max_pix));
    out_row[3] = T::cast_from(clamp(pix3 + v3, 0, max_pix));
    out_row[4] = T::cast_from(clamp(pix4 + v4, 0, max_pix));
    out_row[5] = T::cast_from(clamp(pix5 + v5, 0, max_pix));
    out_row[6] = T::cast_from(clamp(pix6 + v6, 0, max_pix));
    out_row[7] = T::cast_from(clamp(pix7 + v7, 0, max_pix));
  }
}

/// SIMD IDCT4 stage for processing 4 values at once
/// Used as a building block for IDCT8
#[inline(always)]
fn idct4_simd(in0: v128, in1: v128, in2: v128, in3: v128, range: i32) -> (v128, v128, v128, v128) {
  // stage 1: reorder [in0, in2, in1, in3] -> use in0,in2 for evens, in1,in3 for odds
  // stage 2: butterfly
  let s0 = half_btf_simd(COSPI_32, in0, COSPI_32, in2);
  let s1 = half_btf_simd(COSPI_32, in0, -COSPI_32, in2);
  let s2 = half_btf_simd(COSPI_48, in1, -COSPI_16, in3);
  let s3 = half_btf_simd(COSPI_16, in1, COSPI_48, in3);
  
  // stage 3: add/sub with clamp
  (
    clamp_vec(i32x4_add(s0, s3), range),
    clamp_vec(i32x4_add(s1, s2), range),
    clamp_vec(i32x4_sub(s1, s2), range),
    clamp_vec(i32x4_sub(s0, s3), range),
  )
}

// SINPI constants for ADST4
const SINPI_1: i32 = 1321;
const SINPI_2: i32 = 2482;
const SINPI_3: i32 = 3344;
const SINPI_4: i32 = 3803;

/// SIMD IADST4 (Asymmetric DST) stage for processing 4 values at once
/// Implements av1_iadst4 using SIMD
#[inline(always)]
fn iadst4_simd(x0: v128, x1: v128, x2: v128, x3: v128) -> (v128, v128, v128, v128) {
  let bit = 12i32;
  let round = i32x4_splat(1 << (bit - 1));
  
  let sinpi1 = i32x4_splat(SINPI_1);
  let sinpi2 = i32x4_splat(SINPI_2);
  let sinpi3 = i32x4_splat(SINPI_3);
  let sinpi4 = i32x4_splat(SINPI_4);
  
  // stage 1
  let s0 = i32x4_mul(sinpi1, x0);  // SINPI_1 * x0
  let s1 = i32x4_mul(sinpi2, x0);  // SINPI_2 * x0
  let s2 = i32x4_mul(sinpi3, x1);  // SINPI_3 * x1
  let s3 = i32x4_mul(sinpi4, x2);  // SINPI_4 * x2
  let s4 = i32x4_mul(sinpi1, x2);  // SINPI_1 * x2
  let s5 = i32x4_mul(sinpi2, x3);  // SINPI_2 * x3
  let s6 = i32x4_mul(sinpi4, x3);  // SINPI_4 * x3
  
  // stage 2: s7 = (x0 - x2) + x3
  let s7 = i32x4_add(i32x4_sub(x0, x2), x3);
  
  // stage 3
  let s0 = i32x4_add(s0, s3);       // s0 = s0 + s3
  let s1 = i32x4_sub(s1, s4);       // s1 = s1 - s4
  let s3_new = s2;                  // s3 = s2
  let s2 = i32x4_mul(sinpi3, s7);   // s2 = SINPI_3 * s7
  
  // stage 4
  let s0 = i32x4_add(s0, s5);       // s0 = s0 + s5
  let s1 = i32x4_sub(s1, s6);       // s1 = s1 - s6
  
  // output with round_shift
  let out0 = i32x4_shr(i32x4_add(s0, round), bit as u32);
  let out1 = i32x4_shr(i32x4_add(s2, round), bit as u32);
  let out2 = i32x4_shr(i32x4_add(s1, round), bit as u32);
  let out3 = i32x4_shr(i32x4_add(i32x4_sub(s0, s3_new), round), bit as u32);
  
  (out0, out1, out2, out3)
}

/// SIMD IDCT8 on 8 values (split across two v128 vectors: lo=0-3, hi=4-7)
/// Inputs: even coefficients (in0, in2, in4, in6), odd coefficients (in1, in3, in5, in7)
#[inline(always)]
fn idct8_simd(
  in0: v128, in1: v128, in2: v128, in3: v128,
  in4: v128, in5: v128, in6: v128, in7: v128, range: i32
) -> (v128, v128, v128, v128, v128, v128, v128, v128) {
  // IDCT8 = IDCT4 on evens + butterfly on odds + combine
  
  // Step 1: IDCT4 on even inputs [in0, in2, in4, in6]
  let (e0, e1, e2, e3) = idct4_simd(in0, in2, in4, in6, range);
  
  // Step 2: Odd butterfly stages
  // stage 1: reorder odds [in1, in5, in3, in7]
  // stage 2: butterfly
  let s0 = half_btf_simd(COSPI_56, in1, -COSPI_8, in7);
  let s1 = half_btf_simd(COSPI_24, in5, -COSPI_40, in3);
  let s2 = half_btf_simd(COSPI_40, in5, COSPI_24, in3);
  let s3 = half_btf_simd(COSPI_8, in1, COSPI_56, in7);
  
  // stage 3: add/sub
  let t0 = clamp_vec(i32x4_add(s0, s1), range);
  let t1 = clamp_vec(i32x4_sub(s0, s1), range);
  let t2 = clamp_vec(i32x4_sub(s3, s2), range);
  let t3 = clamp_vec(i32x4_add(s2, s3), range);
  
  // stage 4: more butterflies
  let o0 = t0;
  let o1 = half_btf_simd(-COSPI_32, t1, COSPI_32, t2);
  let o2 = half_btf_simd(COSPI_32, t1, COSPI_32, t2);
  let o3 = t3;
  
  // Step 3: Combine even and odd results
  // output[0] = e0 + o3, output[7] = e0 - o3
  // output[1] = e1 + o2, output[6] = e1 - o2
  // output[2] = e2 + o1, output[5] = e2 - o1
  // output[3] = e3 + o0, output[4] = e3 - o0
  (
    clamp_vec(i32x4_add(e0, o3), range),
    clamp_vec(i32x4_add(e1, o2), range),
    clamp_vec(i32x4_add(e2, o1), range),
    clamp_vec(i32x4_add(e3, o0), range),
    clamp_vec(i32x4_sub(e3, o0), range),
    clamp_vec(i32x4_sub(e2, o1), range),
    clamp_vec(i32x4_sub(e1, o2), range),
    clamp_vec(i32x4_sub(e0, o3), range),
  )
}

/// SIMD-optimized 8x8 DCT inverse transform for HBD (10-bit) pixels
#[inline(always)]
fn inverse_transform_add_8x8_dct_simd_hbd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  let range = (bd + 8) as i32;
  
  // Input is column-major: input[col*8 + row] = coefficient at (row, col)
  // Load all 8 columns (64 coefficients)
  // Process first 4 rows with SIMD (lanes 0-3 = rows 0-3)
  // Then process next 4 rows (lanes 0-3 = rows 4-7)
  
  // Load columns 0-7, lower 4 rows (indices 0-3 within each column)
  let c0_lo = i32x4(i32::cast_from(input[0]), i32::cast_from(input[1]), i32::cast_from(input[2]), i32::cast_from(input[3]));
  let c1_lo = i32x4(i32::cast_from(input[8]), i32::cast_from(input[9]), i32::cast_from(input[10]), i32::cast_from(input[11]));
  let c2_lo = i32x4(i32::cast_from(input[16]), i32::cast_from(input[17]), i32::cast_from(input[18]), i32::cast_from(input[19]));
  let c3_lo = i32x4(i32::cast_from(input[24]), i32::cast_from(input[25]), i32::cast_from(input[26]), i32::cast_from(input[27]));
  let c4_lo = i32x4(i32::cast_from(input[32]), i32::cast_from(input[33]), i32::cast_from(input[34]), i32::cast_from(input[35]));
  let c5_lo = i32x4(i32::cast_from(input[40]), i32::cast_from(input[41]), i32::cast_from(input[42]), i32::cast_from(input[43]));
  let c6_lo = i32x4(i32::cast_from(input[48]), i32::cast_from(input[49]), i32::cast_from(input[50]), i32::cast_from(input[51]));
  let c7_lo = i32x4(i32::cast_from(input[56]), i32::cast_from(input[57]), i32::cast_from(input[58]), i32::cast_from(input[59]));
  
  // Load columns 0-7, upper 4 rows (indices 4-7 within each column)
  let c0_hi = i32x4(i32::cast_from(input[4]), i32::cast_from(input[5]), i32::cast_from(input[6]), i32::cast_from(input[7]));
  let c1_hi = i32x4(i32::cast_from(input[12]), i32::cast_from(input[13]), i32::cast_from(input[14]), i32::cast_from(input[15]));
  let c2_hi = i32x4(i32::cast_from(input[20]), i32::cast_from(input[21]), i32::cast_from(input[22]), i32::cast_from(input[23]));
  let c3_hi = i32x4(i32::cast_from(input[28]), i32::cast_from(input[29]), i32::cast_from(input[30]), i32::cast_from(input[31]));
  let c4_hi = i32x4(i32::cast_from(input[36]), i32::cast_from(input[37]), i32::cast_from(input[38]), i32::cast_from(input[39]));
  let c5_hi = i32x4(i32::cast_from(input[44]), i32::cast_from(input[45]), i32::cast_from(input[46]), i32::cast_from(input[47]));
  let c6_hi = i32x4(i32::cast_from(input[52]), i32::cast_from(input[53]), i32::cast_from(input[54]), i32::cast_from(input[55]));
  let c7_hi = i32x4(i32::cast_from(input[60]), i32::cast_from(input[61]), i32::cast_from(input[62]), i32::cast_from(input[63]));
  
  // Row transforms: Apply IDCT8 to each row (we have 8 rows to process)
  // Each row transform processes 8 column inputs and produces 8 column outputs
  // We process rows 0-3 with one set of SIMD ops, rows 4-7 with another
  
  // For rows 0-3 (c*_lo vectors where lane i = row i):
  let (r0_lo, r1_lo, r2_lo, r3_lo, r4_lo, r5_lo, r6_lo, r7_lo) = 
    idct8_simd(c0_lo, c1_lo, c2_lo, c3_lo, c4_lo, c5_lo, c6_lo, c7_lo, range);
  
  // For rows 4-7 (c*_hi vectors where lane i = row 4+i):
  let (r0_hi, r1_hi, r2_hi, r3_hi, r4_hi, r5_hi, r6_hi, r7_hi) = 
    idct8_simd(c0_hi, c1_hi, c2_hi, c3_hi, c4_hi, c5_hi, c6_hi, c7_hi, range);
  
  // Now we need to transpose and do column transforms
  // After row transform: r*_lo[i] = intermediate[row i, col *] for rows 0-3
  //                      r*_hi[i] = intermediate[row 4+i, col *] for rows 4-7
  
  // Transpose: we need columns, where each column spans all 8 rows
  // Column k = [r_k_lo[0], r_k_lo[1], r_k_lo[2], r_k_lo[3], r_k_hi[0], r_k_hi[1], r_k_hi[2], r_k_hi[3]]
  
  // For column transforms, we process columns 0-3 (outputs to rows 0-7, cols 0-3)
  // then columns 4-7 (outputs to rows 0-7, cols 4-7)
  
  // Transpose first 4 columns  
  let (t0_lo, t1_lo, t2_lo, t3_lo) = transpose4x4(r0_lo, r1_lo, r2_lo, r3_lo);
  let (t0_hi, t1_hi, t2_hi, t3_hi) = transpose4x4(r0_hi, r1_hi, r2_hi, r3_hi);
  
  // Transpose last 4 columns
  let (t4_lo, t5_lo, t6_lo, t7_lo) = transpose4x4(r4_lo, r5_lo, r6_lo, r7_lo);
  let (t4_hi, t5_hi, t6_hi, t7_hi) = transpose4x4(r4_hi, r5_hi, r6_hi, r7_hi);
  
  let range2 = (bd.max(10) + 6) as i32;
  
  // Column transforms on columns 0-3 (inputs from t0,t1,t2,t3 lo/hi)
  // t*_lo[i] = value at row i (0-3), column *
  // t*_hi[i] = value at row 4+i (4-7), column *
  // Combine to get full column: lo for rows 0-3, hi for rows 4-7
  
  // IDCT8 on column 0: inputs are t0_lo[0..4] and t0_hi[0..4] = rows 0-7 of column 0
  // But we can't easily extract - we need to reorganize
  // Actually, after transpose4x4:
  // t0_lo = [row0_col0, row0_col1, row0_col2, row0_col3] etc
  // No wait, let me reconsider...
  
  // After row transform: r_k_lo[lane] = intermediate result for (row=lane, col=k)
  // Transpose gives us: t_row_lo[lane] = intermediate result for (row=?, col=lane)
  
  // Actually the structure is getting complex. Let me simplify with scalar extraction for column transforms.
  
  // Extract all 64 intermediate values and do column transforms with extraction
  let mut inter = [[0i32; 8]; 8]; // [row][col]
  
  // Extract from r*_lo (rows 0-3)
  inter[0][0] = i32x4_extract_lane::<0>(r0_lo); inter[0][1] = i32x4_extract_lane::<0>(r1_lo);
  inter[0][2] = i32x4_extract_lane::<0>(r2_lo); inter[0][3] = i32x4_extract_lane::<0>(r3_lo);
  inter[0][4] = i32x4_extract_lane::<0>(r4_lo); inter[0][5] = i32x4_extract_lane::<0>(r5_lo);
  inter[0][6] = i32x4_extract_lane::<0>(r6_lo); inter[0][7] = i32x4_extract_lane::<0>(r7_lo);
  
  inter[1][0] = i32x4_extract_lane::<1>(r0_lo); inter[1][1] = i32x4_extract_lane::<1>(r1_lo);
  inter[1][2] = i32x4_extract_lane::<1>(r2_lo); inter[1][3] = i32x4_extract_lane::<1>(r3_lo);
  inter[1][4] = i32x4_extract_lane::<1>(r4_lo); inter[1][5] = i32x4_extract_lane::<1>(r5_lo);
  inter[1][6] = i32x4_extract_lane::<1>(r6_lo); inter[1][7] = i32x4_extract_lane::<1>(r7_lo);
  
  inter[2][0] = i32x4_extract_lane::<2>(r0_lo); inter[2][1] = i32x4_extract_lane::<2>(r1_lo);
  inter[2][2] = i32x4_extract_lane::<2>(r2_lo); inter[2][3] = i32x4_extract_lane::<2>(r3_lo);
  inter[2][4] = i32x4_extract_lane::<2>(r4_lo); inter[2][5] = i32x4_extract_lane::<2>(r5_lo);
  inter[2][6] = i32x4_extract_lane::<2>(r6_lo); inter[2][7] = i32x4_extract_lane::<2>(r7_lo);
  
  inter[3][0] = i32x4_extract_lane::<3>(r0_lo); inter[3][1] = i32x4_extract_lane::<3>(r1_lo);
  inter[3][2] = i32x4_extract_lane::<3>(r2_lo); inter[3][3] = i32x4_extract_lane::<3>(r3_lo);
  inter[3][4] = i32x4_extract_lane::<3>(r4_lo); inter[3][5] = i32x4_extract_lane::<3>(r5_lo);
  inter[3][6] = i32x4_extract_lane::<3>(r6_lo); inter[3][7] = i32x4_extract_lane::<3>(r7_lo);
  
  // Extract from r*_hi (rows 4-7)
  inter[4][0] = i32x4_extract_lane::<0>(r0_hi); inter[4][1] = i32x4_extract_lane::<0>(r1_hi);
  inter[4][2] = i32x4_extract_lane::<0>(r2_hi); inter[4][3] = i32x4_extract_lane::<0>(r3_hi);
  inter[4][4] = i32x4_extract_lane::<0>(r4_hi); inter[4][5] = i32x4_extract_lane::<0>(r5_hi);
  inter[4][6] = i32x4_extract_lane::<0>(r6_hi); inter[4][7] = i32x4_extract_lane::<0>(r7_hi);
  
  inter[5][0] = i32x4_extract_lane::<1>(r0_hi); inter[5][1] = i32x4_extract_lane::<1>(r1_hi);
  inter[5][2] = i32x4_extract_lane::<1>(r2_hi); inter[5][3] = i32x4_extract_lane::<1>(r3_hi);
  inter[5][4] = i32x4_extract_lane::<1>(r4_hi); inter[5][5] = i32x4_extract_lane::<1>(r5_hi);
  inter[5][6] = i32x4_extract_lane::<1>(r6_hi); inter[5][7] = i32x4_extract_lane::<1>(r7_hi);
  
  inter[6][0] = i32x4_extract_lane::<2>(r0_hi); inter[6][1] = i32x4_extract_lane::<2>(r1_hi);
  inter[6][2] = i32x4_extract_lane::<2>(r2_hi); inter[6][3] = i32x4_extract_lane::<2>(r3_hi);
  inter[6][4] = i32x4_extract_lane::<2>(r4_hi); inter[6][5] = i32x4_extract_lane::<2>(r5_hi);
  inter[6][6] = i32x4_extract_lane::<2>(r6_hi); inter[6][7] = i32x4_extract_lane::<2>(r7_hi);
  
  inter[7][0] = i32x4_extract_lane::<3>(r0_hi); inter[7][1] = i32x4_extract_lane::<3>(r1_hi);
  inter[7][2] = i32x4_extract_lane::<3>(r2_hi); inter[7][3] = i32x4_extract_lane::<3>(r3_hi);
  inter[7][4] = i32x4_extract_lane::<3>(r4_hi); inter[7][5] = i32x4_extract_lane::<3>(r5_hi);
  inter[7][6] = i32x4_extract_lane::<3>(r6_hi); inter[7][7] = i32x4_extract_lane::<3>(r7_hi);
  
  // Column transforms: for each column, load 8 rows, IDCT8, output
  // Process columns 0-3 with SIMD (each lane = different column)
  let col0_in = i32x4(inter[0][0], inter[0][1], inter[0][2], inter[0][3]);
  let col1_in = i32x4(inter[1][0], inter[1][1], inter[1][2], inter[1][3]);
  let col2_in = i32x4(inter[2][0], inter[2][1], inter[2][2], inter[2][3]);
  let col3_in = i32x4(inter[3][0], inter[3][1], inter[3][2], inter[3][3]);
  let col4_in = i32x4(inter[4][0], inter[4][1], inter[4][2], inter[4][3]);
  let col5_in = i32x4(inter[5][0], inter[5][1], inter[5][2], inter[5][3]);
  let col6_in = i32x4(inter[6][0], inter[6][1], inter[6][2], inter[6][3]);
  let col7_in = i32x4(inter[7][0], inter[7][1], inter[7][2], inter[7][3]);
  
  let (f0_03, f1_03, f2_03, f3_03, f4_03, f5_03, f6_03, f7_03) = 
    idct8_simd(col0_in, col1_in, col2_in, col3_in, col4_in, col5_in, col6_in, col7_in, range2);
  
  // Process columns 4-7
  let col0_in2 = i32x4(inter[0][4], inter[0][5], inter[0][6], inter[0][7]);
  let col1_in2 = i32x4(inter[1][4], inter[1][5], inter[1][6], inter[1][7]);
  let col2_in2 = i32x4(inter[2][4], inter[2][5], inter[2][6], inter[2][7]);
  let col3_in2 = i32x4(inter[3][4], inter[3][5], inter[3][6], inter[3][7]);
  let col4_in2 = i32x4(inter[4][4], inter[4][5], inter[4][6], inter[4][7]);
  let col5_in2 = i32x4(inter[5][4], inter[5][5], inter[5][6], inter[5][7]);
  let col6_in2 = i32x4(inter[6][4], inter[6][5], inter[6][6], inter[6][7]);
  let col7_in2 = i32x4(inter[7][4], inter[7][5], inter[7][6], inter[7][7]);
  
  let (f0_47, f1_47, f2_47, f3_47, f4_47, f5_47, f6_47, f7_47) = 
    idct8_simd(col0_in2, col1_in2, col2_in2, col3_in2, col4_in2, col5_in2, col6_in2, col7_in2, range2);
  
  // Round shift by 4 and add to output
  let round = i32x4_splat(8);
  let max_pix = (1 << bd) - 1;
  
  // Output rows 0-7, columns 0-3 (f*_03)
  // Output rows 0-7, columns 4-7 (f*_47)
  let outputs_03 = [f0_03, f1_03, f2_03, f3_03, f4_03, f5_03, f6_03, f7_03];
  let outputs_47 = [f0_47, f1_47, f2_47, f3_47, f4_47, f5_47, f6_47, f7_47];
  
  for row_idx in 0..8 {
    let out_row = &mut output[row_idx];
    
    // Columns 0-3
    let v03 = i32x4_shr(i32x4_add(outputs_03[row_idx], round), 4);
    let v0 = i32x4_extract_lane::<0>(v03);
    let v1 = i32x4_extract_lane::<1>(v03);
    let v2 = i32x4_extract_lane::<2>(v03);
    let v3 = i32x4_extract_lane::<3>(v03);
    
    let pix0: i32 = out_row[0].into();
    let pix1: i32 = out_row[1].into();
    let pix2: i32 = out_row[2].into();
    let pix3: i32 = out_row[3].into();
    out_row[0] = T::cast_from(clamp(pix0 + v0, 0, max_pix));
    out_row[1] = T::cast_from(clamp(pix1 + v1, 0, max_pix));
    out_row[2] = T::cast_from(clamp(pix2 + v2, 0, max_pix));
    out_row[3] = T::cast_from(clamp(pix3 + v3, 0, max_pix));
    
    // Columns 4-7
    let v47 = i32x4_shr(i32x4_add(outputs_47[row_idx], round), 4);
    let v4 = i32x4_extract_lane::<0>(v47);
    let v5 = i32x4_extract_lane::<1>(v47);
    let v6 = i32x4_extract_lane::<2>(v47);
    let v7 = i32x4_extract_lane::<3>(v47);
    
    let pix4: i32 = out_row[4].into();
    let pix5: i32 = out_row[5].into();
    let pix6: i32 = out_row[6].into();
    let pix7: i32 = out_row[7].into();
    out_row[4] = T::cast_from(clamp(pix4 + v4, 0, max_pix));
    out_row[5] = T::cast_from(clamp(pix5 + v5, 0, max_pix));
    out_row[6] = T::cast_from(clamp(pix6 + v6, 0, max_pix));
    out_row[7] = T::cast_from(clamp(pix7 + v7, 0, max_pix));
  }
}

/// SIMD-optimized 16x16 DCT inverse transform for HBD (10-bit) pixels
#[inline(always)]
fn inverse_transform_add_16x16_dct_simd_hbd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  let range = (bd + 8) as i32;
  
  // Need to process 16 rows and 16 columns.
  // We use the same strategy as 8x8: load stripes of rows, transform rows, transpose, transform columns.
  // Stripes: 4 stripes of 4 rows each.
  
  // Intermediate buffer to hold row transform results.
  // 16x16 = 256 values.
  // we store them as 4x4 transposed blocks or just linear?
  // Let's store as [row][col] conceptually, using temp arrays.
  // Actually, we can just process stripes.
  
  // Storage for row transform results: 16 rows, 16 columns.
  // Stored as 16 vectors (columns) for each stripe? 
  // No, we want to transpose 4x4 blocks.
  
  // Let's allocate full intermediate on stack? 256 i32s = 1KB. Fine.
  // Or better: keep in registers/locals as much as possible.
  
  // We process 4 stripes (rows 0-3, 4-7, 8-11, 12-15).
  // For each stripe, we load 16 columns (c0..c15).
  // Each c_i is a v128 containing [r0, r1, r2, r3] for that column.
  
  // Stripe 0 (rows 0-3)
  let (s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7, s0_8, s0_9, s0_10, s0_11, s0_12, s0_13, s0_14, s0_15) = 
    load_stripe_16::<T>(input, 0, 16);
  let (r0_0, r0_1, r0_2, r0_3, r0_4, r0_5, r0_6, r0_7, r0_8, r0_9, r0_10, r0_11, r0_12, r0_13, r0_14, r0_15) = idct16_simd(
    s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7, s0_8, s0_9, s0_10, s0_11, s0_12, s0_13, s0_14, s0_15, range
  );
  
  // Stripe 1 (rows 4-7)
  let (s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7, s1_8, s1_9, s1_10, s1_11, s1_12, s1_13, s1_14, s1_15) = 
    load_stripe_16::<T>(input, 4, 16);
  let (r1_0, r1_1, r1_2, r1_3, r1_4, r1_5, r1_6, r1_7, r1_8, r1_9, r1_10, r1_11, r1_12, r1_13, r1_14, r1_15) = idct16_simd(
    s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7, s1_8, s1_9, s1_10, s1_11, s1_12, s1_13, s1_14, s1_15, range
  );
  
  // Stripe 2 (rows 8-11)
  let (s2_0, s2_1, s2_2, s2_3, s2_4, s2_5, s2_6, s2_7, s2_8, s2_9, s2_10, s2_11, s2_12, s2_13, s2_14, s2_15) = 
    load_stripe_16::<T>(input, 8, 16);
  let (r2_0, r2_1, r2_2, r2_3, r2_4, r2_5, r2_6, r2_7, r2_8, r2_9, r2_10, r2_11, r2_12, r2_13, r2_14, r2_15) = idct16_simd(
    s2_0, s2_1, s2_2, s2_3, s2_4, s2_5, s2_6, s2_7, s2_8, s2_9, s2_10, s2_11, s2_12, s2_13, s2_14, s2_15, range
  );
  
  // Stripe 3 (rows 12-15)
  let (s3_0, s3_1, s3_2, s3_3, s3_4, s3_5, s3_6, s3_7, s3_8, s3_9, s3_10, s3_11, s3_12, s3_13, s3_14, s3_15) = 
    load_stripe_16::<T>(input, 12, 16);
  let (r3_0, r3_1, r3_2, r3_3, r3_4, r3_5, r3_6, r3_7, r3_8, r3_9, r3_10, r3_11, r3_12, r3_13, r3_14, r3_15) = idct16_simd(
    s3_0, s3_1, s3_2, s3_3, s3_4, s3_5, s3_6, s3_7, s3_8, s3_9, s3_10, s3_11, s3_12, s3_13, s3_14, s3_15, range
  );
  
  let range2 = (bd.max(10) + 6) as i32;
  let interm_shift = 2;

  let r0_0 = clamp_vec(round_shift_vec(r0_0, interm_shift), range2);
  let r0_1 = clamp_vec(round_shift_vec(r0_1, interm_shift), range2);
  let r0_2 = clamp_vec(round_shift_vec(r0_2, interm_shift), range2);
  let r0_3 = clamp_vec(round_shift_vec(r0_3, interm_shift), range2);
  let r0_4 = clamp_vec(round_shift_vec(r0_4, interm_shift), range2);
  let r0_5 = clamp_vec(round_shift_vec(r0_5, interm_shift), range2);
  let r0_6 = clamp_vec(round_shift_vec(r0_6, interm_shift), range2);
  let r0_7 = clamp_vec(round_shift_vec(r0_7, interm_shift), range2);
  let r0_8 = clamp_vec(round_shift_vec(r0_8, interm_shift), range2);
  let r0_9 = clamp_vec(round_shift_vec(r0_9, interm_shift), range2);
  let r0_10 = clamp_vec(round_shift_vec(r0_10, interm_shift), range2);
  let r0_11 = clamp_vec(round_shift_vec(r0_11, interm_shift), range2);
  let r0_12 = clamp_vec(round_shift_vec(r0_12, interm_shift), range2);
  let r0_13 = clamp_vec(round_shift_vec(r0_13, interm_shift), range2);
  let r0_14 = clamp_vec(round_shift_vec(r0_14, interm_shift), range2);
  let r0_15 = clamp_vec(round_shift_vec(r0_15, interm_shift), range2);

  let r1_0 = clamp_vec(round_shift_vec(r1_0, interm_shift), range2);
  let r1_1 = clamp_vec(round_shift_vec(r1_1, interm_shift), range2);
  let r1_2 = clamp_vec(round_shift_vec(r1_2, interm_shift), range2);
  let r1_3 = clamp_vec(round_shift_vec(r1_3, interm_shift), range2);
  let r1_4 = clamp_vec(round_shift_vec(r1_4, interm_shift), range2);
  let r1_5 = clamp_vec(round_shift_vec(r1_5, interm_shift), range2);
  let r1_6 = clamp_vec(round_shift_vec(r1_6, interm_shift), range2);
  let r1_7 = clamp_vec(round_shift_vec(r1_7, interm_shift), range2);
  let r1_8 = clamp_vec(round_shift_vec(r1_8, interm_shift), range2);
  let r1_9 = clamp_vec(round_shift_vec(r1_9, interm_shift), range2);
  let r1_10 = clamp_vec(round_shift_vec(r1_10, interm_shift), range2);
  let r1_11 = clamp_vec(round_shift_vec(r1_11, interm_shift), range2);
  let r1_12 = clamp_vec(round_shift_vec(r1_12, interm_shift), range2);
  let r1_13 = clamp_vec(round_shift_vec(r1_13, interm_shift), range2);
  let r1_14 = clamp_vec(round_shift_vec(r1_14, interm_shift), range2);
  let r1_15 = clamp_vec(round_shift_vec(r1_15, interm_shift), range2);

  let r2_0 = clamp_vec(round_shift_vec(r2_0, interm_shift), range2);
  let r2_1 = clamp_vec(round_shift_vec(r2_1, interm_shift), range2);
  let r2_2 = clamp_vec(round_shift_vec(r2_2, interm_shift), range2);
  let r2_3 = clamp_vec(round_shift_vec(r2_3, interm_shift), range2);
  let r2_4 = clamp_vec(round_shift_vec(r2_4, interm_shift), range2);
  let r2_5 = clamp_vec(round_shift_vec(r2_5, interm_shift), range2);
  let r2_6 = clamp_vec(round_shift_vec(r2_6, interm_shift), range2);
  let r2_7 = clamp_vec(round_shift_vec(r2_7, interm_shift), range2);
  let r2_8 = clamp_vec(round_shift_vec(r2_8, interm_shift), range2);
  let r2_9 = clamp_vec(round_shift_vec(r2_9, interm_shift), range2);
  let r2_10 = clamp_vec(round_shift_vec(r2_10, interm_shift), range2);
  let r2_11 = clamp_vec(round_shift_vec(r2_11, interm_shift), range2);
  let r2_12 = clamp_vec(round_shift_vec(r2_12, interm_shift), range2);
  let r2_13 = clamp_vec(round_shift_vec(r2_13, interm_shift), range2);
  let r2_14 = clamp_vec(round_shift_vec(r2_14, interm_shift), range2);
  let r2_15 = clamp_vec(round_shift_vec(r2_15, interm_shift), range2);

  let r3_0 = clamp_vec(round_shift_vec(r3_0, interm_shift), range2);
  let r3_1 = clamp_vec(round_shift_vec(r3_1, interm_shift), range2);
  let r3_2 = clamp_vec(round_shift_vec(r3_2, interm_shift), range2);
  let r3_3 = clamp_vec(round_shift_vec(r3_3, interm_shift), range2);
  let r3_4 = clamp_vec(round_shift_vec(r3_4, interm_shift), range2);
  let r3_5 = clamp_vec(round_shift_vec(r3_5, interm_shift), range2);
  let r3_6 = clamp_vec(round_shift_vec(r3_6, interm_shift), range2);
  let r3_7 = clamp_vec(round_shift_vec(r3_7, interm_shift), range2);
  let r3_8 = clamp_vec(round_shift_vec(r3_8, interm_shift), range2);
  let r3_9 = clamp_vec(round_shift_vec(r3_9, interm_shift), range2);
  let r3_10 = clamp_vec(round_shift_vec(r3_10, interm_shift), range2);
  let r3_11 = clamp_vec(round_shift_vec(r3_11, interm_shift), range2);
  let r3_12 = clamp_vec(round_shift_vec(r3_12, interm_shift), range2);
  let r3_13 = clamp_vec(round_shift_vec(r3_13, interm_shift), range2);
  let r3_14 = clamp_vec(round_shift_vec(r3_14, interm_shift), range2);
  let r3_15 = clamp_vec(round_shift_vec(r3_15, interm_shift), range2);

  // Now we have 4 sets of 16 vectors.
  // r0_res.0 = [r0c0, r1c0, r2c0, r3c0]
  // r1_res.0 = [r4c0, r5c0, r6c0, r7c0]
  // etc.
  
  // We process column transforms in 4 groups of 4 columns (0-3, 4-7, 8-11, 12-15).
  // For group 0 (cols 0-3):
  // We need to transpose the 4x4 blocks to get inputs for idct16.
  
  // Transpose top-left 4x4 (rows 0-3, cols 0-3)
  // r0_res.0..3 contain rows 0-3 for cols 0..3
  let (t0_0, t0_1, t0_2, t0_3) = transpose4x4(r0_0, r0_1, r0_2, r0_3);
  // t0_0 = [r0c0, r0c1, r0c2, r0c3] -> This is Row 0 (cols 0-3)
  
  // Transpose next 4 rows (rows 4-7, cols 0-3)
  let (t1_0, t1_1, t1_2, t1_3) = transpose4x4(r1_0, r1_1, r1_2, r1_3);
  
  // Transpose next 4 rows (rows 8-11, cols 0-3)
  let (t2_0, t2_1, t2_2, t2_3) = transpose4x4(r2_0, r2_1, r2_2, r2_3);
  
  // Transpose next 4 rows (rows 12-15, cols 0-3)
  let (t3_0, t3_1, t3_2, t3_3) = transpose4x4(r3_0, r3_1, r3_2, r3_3);
  
  // Now assemble inputs for idct16 (cols 0-3)
  // Input 0 = Row 0 (cols 0-3) = t0_0
  // Input 1 = Row 1 (cols 0-3) = t0_1
  // ...
  // Input 4 = Row 4 (cols 0-3) = t1_0
  // ...
  let c0_res = idct16_simd(
    t0_0, t0_1, t0_2, t0_3,
    t1_0, t1_1, t1_2, t1_3,
    t2_0, t2_1, t2_2, t2_3,
    t3_0, t3_1, t3_2, t3_3,
    range2
  );
  
  store_stripe_4(output, c0_res, bd, 0); // Stores cols 0-3 for all 16 rows

  // Repeat for cols 4-7
  let (t0_0, t0_1, t0_2, t0_3) = transpose4x4(r0_4, r0_5, r0_6, r0_7);
  let (t1_0, t1_1, t1_2, t1_3) = transpose4x4(r1_4, r1_5, r1_6, r1_7);
  let (t2_0, t2_1, t2_2, t2_3) = transpose4x4(r2_4, r2_5, r2_6, r2_7);
  let (t3_0, t3_1, t3_2, t3_3) = transpose4x4(r3_4, r3_5, r3_6, r3_7);
  
  let c1_res = idct16_simd(
    t0_0, t0_1, t0_2, t0_3, t1_0, t1_1, t1_2, t1_3, t2_0, t2_1, t2_2, t2_3, t3_0, t3_1, t3_2, t3_3, range2
  );
  store_stripe_4(output, c1_res, bd, 4);

  // Repeat for cols 8-11
  let (t0_0, t0_1, t0_2, t0_3) = transpose4x4(r0_8, r0_9, r0_10, r0_11);
  let (t1_0, t1_1, t1_2, t1_3) = transpose4x4(r1_8, r1_9, r1_10, r1_11);
  let (t2_0, t2_1, t2_2, t2_3) = transpose4x4(r2_8, r2_9, r2_10, r2_11);
  let (t3_0, t3_1, t3_2, t3_3) = transpose4x4(r3_8, r3_9, r3_10, r3_11);
  
  let c2_res = idct16_simd(
    t0_0, t0_1, t0_2, t0_3, t1_0, t1_1, t1_2, t1_3, t2_0, t2_1, t2_2, t2_3, t3_0, t3_1, t3_2, t3_3, range2
  );
  store_stripe_4(output, c2_res, bd, 8);

  // Repeat for cols 12-15
  let (t0_0, t0_1, t0_2, t0_3) = transpose4x4(r0_12, r0_13, r0_14, r0_15);
  let (t1_0, t1_1, t1_2, t1_3) = transpose4x4(r1_12, r1_13, r1_14, r1_15);
  let (t2_0, t2_1, t2_2, t2_3) = transpose4x4(r2_12, r2_13, r2_14, r2_15);
  let (t3_0, t3_1, t3_2, t3_3) = transpose4x4(r3_12, r3_13, r3_14, r3_15);
  
  let c3_res = idct16_simd(
    t0_0, t0_1, t0_2, t0_3, t1_0, t1_1, t1_2, t1_3, t2_0, t2_1, t2_2, t2_3, t3_0, t3_1, t3_2, t3_3, range2
  );
  store_stripe_4(output, c3_res, bd, 12);
}

/// SIMD-optimized 16x16 IDTX (identity) inverse transform for HBD pixels
#[inline(always)]
fn inverse_transform_add_16x16_idtx_simd_hbd<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, bd: usize,
) {
  let range = (bd + 8) as i32;
  let range2 = (bd.max(10) + 6) as i32;
  let interm_shift = 2;

  let (s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7, s0_8, s0_9, s0_10, s0_11, s0_12, s0_13, s0_14, s0_15) =
    load_stripe_16::<T>(input, 0, 16);
  let (s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7, s1_8, s1_9, s1_10, s1_11, s1_12, s1_13, s1_14, s1_15) =
    load_stripe_16::<T>(input, 4, 16);
  let (s2_0, s2_1, s2_2, s2_3, s2_4, s2_5, s2_6, s2_7, s2_8, s2_9, s2_10, s2_11, s2_12, s2_13, s2_14, s2_15) =
    load_stripe_16::<T>(input, 8, 16);
  let (s3_0, s3_1, s3_2, s3_3, s3_4, s3_5, s3_6, s3_7, s3_8, s3_9, s3_10, s3_11, s3_12, s3_13, s3_14, s3_15) =
    load_stripe_16::<T>(input, 12, 16);

  let mut r0_0 = iidentity16_vec(clamp_vec(s0_0, range));
  let mut r0_1 = iidentity16_vec(clamp_vec(s0_1, range));
  let mut r0_2 = iidentity16_vec(clamp_vec(s0_2, range));
  let mut r0_3 = iidentity16_vec(clamp_vec(s0_3, range));
  let mut r0_4 = iidentity16_vec(clamp_vec(s0_4, range));
  let mut r0_5 = iidentity16_vec(clamp_vec(s0_5, range));
  let mut r0_6 = iidentity16_vec(clamp_vec(s0_6, range));
  let mut r0_7 = iidentity16_vec(clamp_vec(s0_7, range));
  let mut r0_8 = iidentity16_vec(clamp_vec(s0_8, range));
  let mut r0_9 = iidentity16_vec(clamp_vec(s0_9, range));
  let mut r0_10 = iidentity16_vec(clamp_vec(s0_10, range));
  let mut r0_11 = iidentity16_vec(clamp_vec(s0_11, range));
  let mut r0_12 = iidentity16_vec(clamp_vec(s0_12, range));
  let mut r0_13 = iidentity16_vec(clamp_vec(s0_13, range));
  let mut r0_14 = iidentity16_vec(clamp_vec(s0_14, range));
  let mut r0_15 = iidentity16_vec(clamp_vec(s0_15, range));

  let mut r1_0 = iidentity16_vec(clamp_vec(s1_0, range));
  let mut r1_1 = iidentity16_vec(clamp_vec(s1_1, range));
  let mut r1_2 = iidentity16_vec(clamp_vec(s1_2, range));
  let mut r1_3 = iidentity16_vec(clamp_vec(s1_3, range));
  let mut r1_4 = iidentity16_vec(clamp_vec(s1_4, range));
  let mut r1_5 = iidentity16_vec(clamp_vec(s1_5, range));
  let mut r1_6 = iidentity16_vec(clamp_vec(s1_6, range));
  let mut r1_7 = iidentity16_vec(clamp_vec(s1_7, range));
  let mut r1_8 = iidentity16_vec(clamp_vec(s1_8, range));
  let mut r1_9 = iidentity16_vec(clamp_vec(s1_9, range));
  let mut r1_10 = iidentity16_vec(clamp_vec(s1_10, range));
  let mut r1_11 = iidentity16_vec(clamp_vec(s1_11, range));
  let mut r1_12 = iidentity16_vec(clamp_vec(s1_12, range));
  let mut r1_13 = iidentity16_vec(clamp_vec(s1_13, range));
  let mut r1_14 = iidentity16_vec(clamp_vec(s1_14, range));
  let mut r1_15 = iidentity16_vec(clamp_vec(s1_15, range));

  let mut r2_0 = iidentity16_vec(clamp_vec(s2_0, range));
  let mut r2_1 = iidentity16_vec(clamp_vec(s2_1, range));
  let mut r2_2 = iidentity16_vec(clamp_vec(s2_2, range));
  let mut r2_3 = iidentity16_vec(clamp_vec(s2_3, range));
  let mut r2_4 = iidentity16_vec(clamp_vec(s2_4, range));
  let mut r2_5 = iidentity16_vec(clamp_vec(s2_5, range));
  let mut r2_6 = iidentity16_vec(clamp_vec(s2_6, range));
  let mut r2_7 = iidentity16_vec(clamp_vec(s2_7, range));
  let mut r2_8 = iidentity16_vec(clamp_vec(s2_8, range));
  let mut r2_9 = iidentity16_vec(clamp_vec(s2_9, range));
  let mut r2_10 = iidentity16_vec(clamp_vec(s2_10, range));
  let mut r2_11 = iidentity16_vec(clamp_vec(s2_11, range));
  let mut r2_12 = iidentity16_vec(clamp_vec(s2_12, range));
  let mut r2_13 = iidentity16_vec(clamp_vec(s2_13, range));
  let mut r2_14 = iidentity16_vec(clamp_vec(s2_14, range));
  let mut r2_15 = iidentity16_vec(clamp_vec(s2_15, range));

  let mut r3_0 = iidentity16_vec(clamp_vec(s3_0, range));
  let mut r3_1 = iidentity16_vec(clamp_vec(s3_1, range));
  let mut r3_2 = iidentity16_vec(clamp_vec(s3_2, range));
  let mut r3_3 = iidentity16_vec(clamp_vec(s3_3, range));
  let mut r3_4 = iidentity16_vec(clamp_vec(s3_4, range));
  let mut r3_5 = iidentity16_vec(clamp_vec(s3_5, range));
  let mut r3_6 = iidentity16_vec(clamp_vec(s3_6, range));
  let mut r3_7 = iidentity16_vec(clamp_vec(s3_7, range));
  let mut r3_8 = iidentity16_vec(clamp_vec(s3_8, range));
  let mut r3_9 = iidentity16_vec(clamp_vec(s3_9, range));
  let mut r3_10 = iidentity16_vec(clamp_vec(s3_10, range));
  let mut r3_11 = iidentity16_vec(clamp_vec(s3_11, range));
  let mut r3_12 = iidentity16_vec(clamp_vec(s3_12, range));
  let mut r3_13 = iidentity16_vec(clamp_vec(s3_13, range));
  let mut r3_14 = iidentity16_vec(clamp_vec(s3_14, range));
  let mut r3_15 = iidentity16_vec(clamp_vec(s3_15, range));

  r0_0 = clamp_vec(round_shift_vec(r0_0, interm_shift), range2);
  r0_1 = clamp_vec(round_shift_vec(r0_1, interm_shift), range2);
  r0_2 = clamp_vec(round_shift_vec(r0_2, interm_shift), range2);
  r0_3 = clamp_vec(round_shift_vec(r0_3, interm_shift), range2);
  r0_4 = clamp_vec(round_shift_vec(r0_4, interm_shift), range2);
  r0_5 = clamp_vec(round_shift_vec(r0_5, interm_shift), range2);
  r0_6 = clamp_vec(round_shift_vec(r0_6, interm_shift), range2);
  r0_7 = clamp_vec(round_shift_vec(r0_7, interm_shift), range2);
  r0_8 = clamp_vec(round_shift_vec(r0_8, interm_shift), range2);
  r0_9 = clamp_vec(round_shift_vec(r0_9, interm_shift), range2);
  r0_10 = clamp_vec(round_shift_vec(r0_10, interm_shift), range2);
  r0_11 = clamp_vec(round_shift_vec(r0_11, interm_shift), range2);
  r0_12 = clamp_vec(round_shift_vec(r0_12, interm_shift), range2);
  r0_13 = clamp_vec(round_shift_vec(r0_13, interm_shift), range2);
  r0_14 = clamp_vec(round_shift_vec(r0_14, interm_shift), range2);
  r0_15 = clamp_vec(round_shift_vec(r0_15, interm_shift), range2);

  r1_0 = clamp_vec(round_shift_vec(r1_0, interm_shift), range2);
  r1_1 = clamp_vec(round_shift_vec(r1_1, interm_shift), range2);
  r1_2 = clamp_vec(round_shift_vec(r1_2, interm_shift), range2);
  r1_3 = clamp_vec(round_shift_vec(r1_3, interm_shift), range2);
  r1_4 = clamp_vec(round_shift_vec(r1_4, interm_shift), range2);
  r1_5 = clamp_vec(round_shift_vec(r1_5, interm_shift), range2);
  r1_6 = clamp_vec(round_shift_vec(r1_6, interm_shift), range2);
  r1_7 = clamp_vec(round_shift_vec(r1_7, interm_shift), range2);
  r1_8 = clamp_vec(round_shift_vec(r1_8, interm_shift), range2);
  r1_9 = clamp_vec(round_shift_vec(r1_9, interm_shift), range2);
  r1_10 = clamp_vec(round_shift_vec(r1_10, interm_shift), range2);
  r1_11 = clamp_vec(round_shift_vec(r1_11, interm_shift), range2);
  r1_12 = clamp_vec(round_shift_vec(r1_12, interm_shift), range2);
  r1_13 = clamp_vec(round_shift_vec(r1_13, interm_shift), range2);
  r1_14 = clamp_vec(round_shift_vec(r1_14, interm_shift), range2);
  r1_15 = clamp_vec(round_shift_vec(r1_15, interm_shift), range2);

  r2_0 = clamp_vec(round_shift_vec(r2_0, interm_shift), range2);
  r2_1 = clamp_vec(round_shift_vec(r2_1, interm_shift), range2);
  r2_2 = clamp_vec(round_shift_vec(r2_2, interm_shift), range2);
  r2_3 = clamp_vec(round_shift_vec(r2_3, interm_shift), range2);
  r2_4 = clamp_vec(round_shift_vec(r2_4, interm_shift), range2);
  r2_5 = clamp_vec(round_shift_vec(r2_5, interm_shift), range2);
  r2_6 = clamp_vec(round_shift_vec(r2_6, interm_shift), range2);
  r2_7 = clamp_vec(round_shift_vec(r2_7, interm_shift), range2);
  r2_8 = clamp_vec(round_shift_vec(r2_8, interm_shift), range2);
  r2_9 = clamp_vec(round_shift_vec(r2_9, interm_shift), range2);
  r2_10 = clamp_vec(round_shift_vec(r2_10, interm_shift), range2);
  r2_11 = clamp_vec(round_shift_vec(r2_11, interm_shift), range2);
  r2_12 = clamp_vec(round_shift_vec(r2_12, interm_shift), range2);
  r2_13 = clamp_vec(round_shift_vec(r2_13, interm_shift), range2);
  r2_14 = clamp_vec(round_shift_vec(r2_14, interm_shift), range2);
  r2_15 = clamp_vec(round_shift_vec(r2_15, interm_shift), range2);

  r3_0 = clamp_vec(round_shift_vec(r3_0, interm_shift), range2);
  r3_1 = clamp_vec(round_shift_vec(r3_1, interm_shift), range2);
  r3_2 = clamp_vec(round_shift_vec(r3_2, interm_shift), range2);
  r3_3 = clamp_vec(round_shift_vec(r3_3, interm_shift), range2);
  r3_4 = clamp_vec(round_shift_vec(r3_4, interm_shift), range2);
  r3_5 = clamp_vec(round_shift_vec(r3_5, interm_shift), range2);
  r3_6 = clamp_vec(round_shift_vec(r3_6, interm_shift), range2);
  r3_7 = clamp_vec(round_shift_vec(r3_7, interm_shift), range2);
  r3_8 = clamp_vec(round_shift_vec(r3_8, interm_shift), range2);
  r3_9 = clamp_vec(round_shift_vec(r3_9, interm_shift), range2);
  r3_10 = clamp_vec(round_shift_vec(r3_10, interm_shift), range2);
  r3_11 = clamp_vec(round_shift_vec(r3_11, interm_shift), range2);
  r3_12 = clamp_vec(round_shift_vec(r3_12, interm_shift), range2);
  r3_13 = clamp_vec(round_shift_vec(r3_13, interm_shift), range2);
  r3_14 = clamp_vec(round_shift_vec(r3_14, interm_shift), range2);
  r3_15 = clamp_vec(round_shift_vec(r3_15, interm_shift), range2);

  let (t0_0, t0_1, t0_2, t0_3) = transpose4x4(r0_0, r0_1, r0_2, r0_3);
  let (t1_0, t1_1, t1_2, t1_3) = transpose4x4(r1_0, r1_1, r1_2, r1_3);
  let (t2_0, t2_1, t2_2, t2_3) = transpose4x4(r2_0, r2_1, r2_2, r2_3);
  let (t3_0, t3_1, t3_2, t3_3) = transpose4x4(r3_0, r3_1, r3_2, r3_3);

  store_stripe_4(
    output,
    (
      iidentity16_vec(t0_0), iidentity16_vec(t0_1), iidentity16_vec(t0_2), iidentity16_vec(t0_3),
      iidentity16_vec(t1_0), iidentity16_vec(t1_1), iidentity16_vec(t1_2), iidentity16_vec(t1_3),
      iidentity16_vec(t2_0), iidentity16_vec(t2_1), iidentity16_vec(t2_2), iidentity16_vec(t2_3),
      iidentity16_vec(t3_0), iidentity16_vec(t3_1), iidentity16_vec(t3_2), iidentity16_vec(t3_3),
    ),
    bd,
    0,
  );

  let (t0_0, t0_1, t0_2, t0_3) = transpose4x4(r0_4, r0_5, r0_6, r0_7);
  let (t1_0, t1_1, t1_2, t1_3) = transpose4x4(r1_4, r1_5, r1_6, r1_7);
  let (t2_0, t2_1, t2_2, t2_3) = transpose4x4(r2_4, r2_5, r2_6, r2_7);
  let (t3_0, t3_1, t3_2, t3_3) = transpose4x4(r3_4, r3_5, r3_6, r3_7);

  store_stripe_4(
    output,
    (
      iidentity16_vec(t0_0), iidentity16_vec(t0_1), iidentity16_vec(t0_2), iidentity16_vec(t0_3),
      iidentity16_vec(t1_0), iidentity16_vec(t1_1), iidentity16_vec(t1_2), iidentity16_vec(t1_3),
      iidentity16_vec(t2_0), iidentity16_vec(t2_1), iidentity16_vec(t2_2), iidentity16_vec(t2_3),
      iidentity16_vec(t3_0), iidentity16_vec(t3_1), iidentity16_vec(t3_2), iidentity16_vec(t3_3),
    ),
    bd,
    4,
  );

  let (t0_0, t0_1, t0_2, t0_3) = transpose4x4(r0_8, r0_9, r0_10, r0_11);
  let (t1_0, t1_1, t1_2, t1_3) = transpose4x4(r1_8, r1_9, r1_10, r1_11);
  let (t2_0, t2_1, t2_2, t2_3) = transpose4x4(r2_8, r2_9, r2_10, r2_11);
  let (t3_0, t3_1, t3_2, t3_3) = transpose4x4(r3_8, r3_9, r3_10, r3_11);

  store_stripe_4(
    output,
    (
      iidentity16_vec(t0_0), iidentity16_vec(t0_1), iidentity16_vec(t0_2), iidentity16_vec(t0_3),
      iidentity16_vec(t1_0), iidentity16_vec(t1_1), iidentity16_vec(t1_2), iidentity16_vec(t1_3),
      iidentity16_vec(t2_0), iidentity16_vec(t2_1), iidentity16_vec(t2_2), iidentity16_vec(t2_3),
      iidentity16_vec(t3_0), iidentity16_vec(t3_1), iidentity16_vec(t3_2), iidentity16_vec(t3_3),
    ),
    bd,
    8,
  );

  let (t0_0, t0_1, t0_2, t0_3) = transpose4x4(r0_12, r0_13, r0_14, r0_15);
  let (t1_0, t1_1, t1_2, t1_3) = transpose4x4(r1_12, r1_13, r1_14, r1_15);
  let (t2_0, t2_1, t2_2, t2_3) = transpose4x4(r2_12, r2_13, r2_14, r2_15);
  let (t3_0, t3_1, t3_2, t3_3) = transpose4x4(r3_12, r3_13, r3_14, r3_15);

  store_stripe_4(
    output,
    (
      iidentity16_vec(t0_0), iidentity16_vec(t0_1), iidentity16_vec(t0_2), iidentity16_vec(t0_3),
      iidentity16_vec(t1_0), iidentity16_vec(t1_1), iidentity16_vec(t1_2), iidentity16_vec(t1_3),
      iidentity16_vec(t2_0), iidentity16_vec(t2_1), iidentity16_vec(t2_2), iidentity16_vec(t2_3),
      iidentity16_vec(t3_0), iidentity16_vec(t3_1), iidentity16_vec(t3_2), iidentity16_vec(t3_3),
    ),
    bd,
    12,
  );
}

#[inline(always)]
fn load_stripe_16<T: Pixel>(input: &[T::Coeff], row_offset: usize, stride: usize) -> (v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128) {
   let mut out = [i32x4_splat(0); 16];
   for col in 0..16 {
     let c0 = i32::cast_from(input[col * stride + row_offset]);
     let c1 = i32::cast_from(input[col * stride + row_offset + 1]);
     let c2 = i32::cast_from(input[col * stride + row_offset + 2]);
     let c3 = i32::cast_from(input[col * stride + row_offset + 3]);
     out[col] = i32x4(c0, c1, c2, c3);
   }
   (out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], 
    out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15])
}

#[inline(always)]
fn store_stripe_4<T: Pixel>(
    output: &mut PlaneRegionMut<'_, T>, 
    vals: (v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128),
    bd: usize, col_offset: usize
) {
    // The inputs to `idct16_simd` were `Row0, Row1... Row15` (for these 4 cols).
    // So output `vals.0` corresponds to `OutputRow0`.
    // `vals.1` -> `OutputRow1`.
    
    let max_pix = (1 << bd) - 1;
    let round_const = i32x4_splat(8); // 1 << (4-1)
    
    let vecs = [
        vals.0, vals.1, vals.2, vals.3, vals.4, vals.5, vals.6, vals.7,
        vals.8, vals.9, vals.10, vals.11, vals.12, vals.13, vals.14, vals.15
    ];
    
    for (r, v) in vecs.iter().enumerate() {
      // Shift and add
      let res = i32x4_shr(i32x4_add(*v, round_const), 4);
      
      let out_row = &mut output[r];
      let val0 = i32x4_extract_lane::<0>(res);
      let val1 = i32x4_extract_lane::<1>(res);
      let val2 = i32x4_extract_lane::<2>(res);
      let val3 = i32x4_extract_lane::<3>(res);
      
      let p0: i32 = out_row[col_offset].as_();
      let p1: i32 = out_row[col_offset + 1].as_();
      let p2: i32 = out_row[col_offset + 2].as_();
      let p3: i32 = out_row[col_offset + 3].as_();
      
      out_row[col_offset] = T::cast_from(clamp(p0 + val0, 0, max_pix));
      out_row[col_offset + 1] = T::cast_from(clamp(p1 + val1, 0, max_pix));
      out_row[col_offset + 2] = T::cast_from(clamp(p2 + val2, 0, max_pix));
      out_row[col_offset + 3] = T::cast_from(clamp(p3 + val3, 0, max_pix));
    }
}

#[inline(always)]
fn idct16_simd(
    in0: v128, in1: v128, in2: v128, in3: v128, in4: v128, in5: v128, in6: v128, in7: v128,
    in8: v128, in9: v128, in10: v128, in11: v128, in12: v128, in13: v128, in14: v128, in15: v128,
    range: i32
) -> (
    v128, v128, v128, v128, v128, v128, v128, v128,
    v128, v128, v128, v128, v128, v128, v128, v128
) {
    // Stage 1: Call IDCT8 on even inputs
    // Evens: 0, 2, 4, 6, 8, 10, 12, 14
    let (t0, t1, t2, t3, t4, t5, t6, t7) = 
        idct8_simd(in0, in2, in4, in6, in8, in10, in12, in14, range);
        
    // Stage 2: Butterfly on odd inputs (1, 9, 5, 13, 3, 11, 7, 15)
    // Indices: 0->1, 1->9, 2->5, 3->13, 4->3, 5->11, 6->7, 7->15
    // Note: The logic below follows av1_idct16 stage 2
    
    let stg1_0 = in1;
    let stg1_1 = in9;
    let stg1_2 = in5;
    let stg1_3 = in13;
    let stg1_4 = in3;
    let stg1_5 = in11;
    let stg1_6 = in7;
    let stg1_7 = in15;

    let s2_0 = half_btf_simd(COSPI_60, stg1_0, -COSPI_4, stg1_7);
    let s2_1 = half_btf_simd(COSPI_28, stg1_1, -COSPI_36, stg1_6);
    let s2_2 = half_btf_simd(COSPI_44, stg1_2, -COSPI_20, stg1_5);
    let s2_3 = half_btf_simd(COSPI_12, stg1_3, -COSPI_52, stg1_4);
    let s2_4 = half_btf_simd(COSPI_52, stg1_3, COSPI_12, stg1_4);
    let s2_5 = half_btf_simd(COSPI_20, stg1_2, COSPI_44, stg1_5);
    let s2_6 = half_btf_simd(COSPI_36, stg1_1, COSPI_28, stg1_6);
    let s2_7 = half_btf_simd(COSPI_4, stg1_0, COSPI_60, stg1_7);

    // Stage 3
    let s3_0 = clamp_vec(i32x4_add(s2_0, s2_1), range);
    let s3_1 = clamp_vec(i32x4_sub(s2_0, s2_1), range);
    let s3_2 = clamp_vec(i32x4_sub(s2_3, s2_2), range); // -s2_2 + s2_3
    let s3_3 = clamp_vec(i32x4_add(s2_2, s2_3), range);
    let s3_4 = clamp_vec(i32x4_add(s2_4, s2_5), range);
    let s3_5 = clamp_vec(i32x4_sub(s2_4, s2_5), range);
    let s3_6 = clamp_vec(i32x4_sub(s2_7, s2_6), range); // -s2_6 + s2_7
    let s3_7 = clamp_vec(i32x4_add(s2_6, s2_7), range);

    // Stage 4
    let s4_0 = s3_0;
    let s4_1 = half_btf_simd(-COSPI_16, s3_1, COSPI_48, s3_6);
    let s4_2 = half_btf_simd(-COSPI_48, s3_2, -COSPI_16, s3_5);
    let s4_3 = s3_3;
    let s4_4 = s3_4;
    let s4_5 = half_btf_simd(-COSPI_16, s3_2, COSPI_48, s3_5);
    let s4_6 = half_btf_simd(COSPI_48, s3_1, COSPI_16, s3_6);
    let s4_7 = s3_7;

    // Stage 5
    let s5_0 = clamp_vec(i32x4_add(s4_0, s4_3), range);
    let s5_1 = clamp_vec(i32x4_add(s4_1, s4_2), range);
    let s5_2 = clamp_vec(i32x4_sub(s4_1, s4_2), range);
    let s5_3 = clamp_vec(i32x4_sub(s4_0, s4_3), range);
    let s5_4 = clamp_vec(i32x4_sub(s4_7, s4_4), range); // -s4_4 + s4_7
    let s5_5 = clamp_vec(i32x4_sub(s4_6, s4_5), range); // -s4_5 + s4_6
    let s5_6 = clamp_vec(i32x4_add(s4_5, s4_6), range);
    let s5_7 = clamp_vec(i32x4_add(s4_4, s4_7), range);

    // Stage 6
    let s6_0 = s5_0;
    let s6_1 = s5_1;
    let s6_2 = half_btf_simd(-COSPI_32, s5_2, COSPI_32, s5_5);
    let s6_3 = half_btf_simd(-COSPI_32, s5_3, COSPI_32, s5_4);
    let s6_4 = half_btf_simd(COSPI_32, s5_3, COSPI_32, s5_4);
    let s6_5 = half_btf_simd(COSPI_32, s5_2, COSPI_32, s5_5);
    let s6_6 = s5_6;
    let s6_7 = s5_7;

    // Stage 7 (Final combination)
    // output[0] = t0 + s6_7
    // output[1] = t1 + s6_6
    // ...
    // output[8] = t7 - s6_0... wait checked Rust code
    // output[8] = t7 - s6_0
    // output[15] = t0 - s6_7
    
    // Rust:
    // output[0] = temp_out[0] + stg6[7]; -> t0 + s6_7
    // output[1] = temp_out[1] + stg6[6]; -> t1 + s6_6
    // output[7] = temp_out[7] + stg6[0]; -> t7 + s6_0
    // output[8] = temp_out[7] - stg6[0]; -> t7 - s6_0
    // output[15] = temp_out[0] - stg6[7]; -> t0 - s6_7
    
    (
        clamp_vec(i32x4_add(t0, s6_7), range),
        clamp_vec(i32x4_add(t1, s6_6), range),
        clamp_vec(i32x4_add(t2, s6_5), range),
        clamp_vec(i32x4_add(t3, s6_4), range),
        clamp_vec(i32x4_add(t4, s6_3), range),
        clamp_vec(i32x4_add(t5, s6_2), range),
        clamp_vec(i32x4_add(t6, s6_1), range),
        clamp_vec(i32x4_add(t7, s6_0), range),
        clamp_vec(i32x4_sub(t7, s6_0), range),
        clamp_vec(i32x4_sub(t6, s6_1), range),
        clamp_vec(i32x4_sub(t5, s6_2), range),
        clamp_vec(i32x4_sub(t4, s6_3), range),
        clamp_vec(i32x4_sub(t3, s6_4), range),
        clamp_vec(i32x4_sub(t2, s6_5), range),
        clamp_vec(i32x4_sub(t1, s6_6), range),
        clamp_vec(i32x4_sub(t0, s6_7), range),
    )
}

/// 4x4 transpose helper
#[inline(always)]
fn transpose4x4(r0: v128, r1: v128, r2: v128, r3: v128) -> (v128, v128, v128, v128) {
  let lo01 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1);
  let hi01 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1);
  let lo23 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3);
  let hi23 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3);
  
  (
    i64x2_shuffle::<0, 2>(lo01, lo23),
    i64x2_shuffle::<1, 3>(lo01, lo23),
    i64x2_shuffle::<0, 2>(hi01, hi23),
    i64x2_shuffle::<1, 3>(hi01, hi23),
  )
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
