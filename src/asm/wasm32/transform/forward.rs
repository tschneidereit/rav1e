// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! SIMD-accelerated forward transform functions for wasm32.

use crate::asm::shared::transform::forward::cast;
use crate::cpu_features::CpuFeatureLevel;
use crate::transform::forward::rust;
use crate::transform::forward_shared::*;
use crate::transform::*;
use crate::util::*;
use std::mem::MaybeUninit;

use core::arch::wasm32::*;

/// A wrapper around v128 representing 4 i32 values.
#[derive(Copy, Clone)]
struct I32X4 {
  data: v128,
}

impl I32X4 {
  #[inline(always)]
  const fn vec(self) -> v128 {
    self.data
  }

  #[inline(always)]
  const fn new(a: v128) -> I32X4 {
    I32X4 { data: a }
  }
}

type TxfmFunc = fn(&mut [I32X4]);

impl_1d_tx!(allow(unused_attributes), );

impl TxOperations for I32X4 {
  #[inline(always)]
  fn zero() -> Self {
    I32X4::new(i32x4_splat(0))
  }

  #[inline(always)]
  fn tx_mul<const SHIFT: i32>(self, mul: i32) -> Self {
    // (self * mul + (1 << (SHIFT - 1))) >> SHIFT
    // Use 64-bit intermediates to avoid overflow
    let mulx4 = i32x4_splat(mul);

    // Multiply: get all 4 products as i64
    // extmul_low does lanes 0,1 -> i64x2
    // extmul_high does lanes 2,3 -> i64x2
    let prod_lo = i64x2_extmul_low_i32x4(self.vec(), mulx4);
    let prod_hi = i64x2_extmul_high_i32x4(self.vec(), mulx4);

    // Rounding constant as i64
    let rounding = i64x2_splat((1i64 << SHIFT) >> 1);

    // Add rounding and shift
    let round_lo = i64x2_add(prod_lo, rounding);
    let round_hi = i64x2_add(prod_hi, rounding);

    let shifted_lo = i64x2_shr(round_lo, SHIFT as u32);
    let shifted_hi = i64x2_shr(round_hi, SHIFT as u32);

    // Pack back to i32 by taking low 32 bits of each i64
    // shifted_lo as i32x4: [r0_lo, r0_hi, r1_lo, r1_hi]
    // shifted_hi as i32x4: [r2_lo, r2_hi, r3_lo, r3_hi]
    // We want: [r0_lo, r1_lo, r2_lo, r3_lo]
    let result = i32x4_shuffle::<0, 2, 4, 6>(shifted_lo, shifted_hi);

    I32X4::new(result)
  }

  #[inline(always)]
  fn rshift1(self) -> Self {
    // (self + (self < 0 ? 1 : 0)) >> 1
    // This handles rounding towards zero for negative numbers
    let is_negative = i32x4_lt(self.vec(), i32x4_splat(0));
    // is_negative is all 1s (-1) when negative, all 0s otherwise
    // We need to add 1 when negative, so we subtract the mask (which is -1)
    let adjusted = i32x4_sub(self.vec(), is_negative);
    I32X4::new(i32x4_shr(adjusted, 1))
  }

  #[inline(always)]
  fn add(self, b: Self) -> Self {
    I32X4::new(i32x4_add(self.vec(), b.vec()))
  }

  #[inline(always)]
  fn sub(self, b: Self) -> Self {
    I32X4::new(i32x4_sub(self.vec(), b.vec()))
  }

  #[inline(always)]
  fn add_avg(self, b: Self) -> Self {
    I32X4::new(i32x4_shr(i32x4_add(self.vec(), b.vec()), 1))
  }

  #[inline(always)]
  fn sub_avg(self, b: Self) -> Self {
    I32X4::new(i32x4_shr(i32x4_sub(self.vec(), b.vec()), 1))
  }
}

/// Transpose a 4x4 matrix stored in 4 v128 registers.
///
/// Input: rows [r0, r1, r2, r3] where each row is [a, b, c, d]
/// Output: transposed rows where each column becomes a row
#[inline(always)]
fn transpose_4x4(input: &[I32X4; 4], into: &mut [I32X4; 4]) {
  // Interleave low pairs
  // r0 = [a0, b0, c0, d0], r1 = [a1, b1, c1, d1]
  // lo01 = [a0, a1, b0, b1]
  // hi01 = [c0, c1, d0, d1]
  let lo01 = i32x4_shuffle::<0, 4, 1, 5>(input[0].vec(), input[1].vec());
  let hi01 = i32x4_shuffle::<2, 6, 3, 7>(input[0].vec(), input[1].vec());
  let lo23 = i32x4_shuffle::<0, 4, 1, 5>(input[2].vec(), input[3].vec());
  let hi23 = i32x4_shuffle::<2, 6, 3, 7>(input[2].vec(), input[3].vec());

  // Final interleave to get transposed result
  // out0 = [a0, a1, a2, a3]
  // out1 = [b0, b1, b2, b3]
  // out2 = [c0, c1, c2, c3]
  // out3 = [d0, d1, d2, d3]
  into[0] = I32X4::new(i32x4_shuffle::<0, 1, 4, 5>(lo01, lo23));
  into[1] = I32X4::new(i32x4_shuffle::<2, 3, 6, 7>(lo01, lo23));
  into[2] = I32X4::new(i32x4_shuffle::<0, 1, 4, 5>(hi01, hi23));
  into[3] = I32X4::new(i32x4_shuffle::<2, 3, 6, 7>(hi01, hi23));
}

/// Transpose an 8x4 matrix (8 rows, 4 columns) stored in 8 v128 registers
/// into a 4x8 matrix (4 rows, 8 columns) stored in 4 pairs of v128 registers.
///
/// Since we only have 128-bit vectors (4 i32s), we output as interleaved chunks.
#[allow(dead_code)]
#[inline(always)]
fn transpose_8x4_to_4x8(input: &[I32X4; 8], into: &mut [I32X4; 8]) {
  // Transpose in two 4x4 blocks
  let mut top: [I32X4; 4] = [I32X4::zero(); 4];
  let mut bot: [I32X4; 4] = [I32X4::zero(); 4];

  transpose_4x4(cast::<4, _>(&input[0..4]), &mut top);
  transpose_4x4(cast::<4, _>(&input[4..8]), &mut bot);

  // Interleave the results: each output row spans the original 8 input rows
  // Row 0 of output = [col0 of inputs 0-3, col0 of inputs 4-7]
  into[0] = top[0];
  into[1] = bot[0];
  into[2] = top[1];
  into[3] = bot[1];
  into[4] = top[2];
  into[5] = bot[2];
  into[6] = top[3];
  into[7] = bot[3];
}

/// Transpose a 4x8 matrix (4 rows × 8 columns, stored as 8 v128s with 4 values each)
/// into an 8x4 matrix (8 rows × 4 columns).
#[allow(dead_code)]
#[inline(always)]
fn transpose_4x8_to_8x4(input: &[I32X4; 8], into: &mut [I32X4; 8]) {
  // Input: 4 rows, each row stored in 2 consecutive v128s
  // input[0], input[1] = row 0 (8 values)
  // input[2], input[3] = row 1 (8 values)
  // etc.

  // Output: 8 rows, each row has 4 values

  // First, transpose the left 4x4 block (cols 0-3)
  let left_in = [input[0], input[2], input[4], input[6]];
  let mut left_out: [I32X4; 4] = [I32X4::zero(); 4];
  transpose_4x4(&left_in, &mut left_out);

  // Then transpose the right 4x4 block (cols 4-7)
  let right_in = [input[1], input[3], input[5], input[7]];
  let mut right_out: [I32X4; 4] = [I32X4::zero(); 4];
  transpose_4x4(&right_in, &mut right_out);

  // Combine: output row i gets left_out[i] and right_out[i]
  into[0] = left_out[0];
  into[1] = right_out[0];
  into[2] = left_out[1];
  into[3] = right_out[1];
  into[4] = left_out[2];
  into[5] = right_out[2];
  into[6] = left_out[3];
  into[7] = right_out[3];
}

/// Transpose an 8x8 matrix stored as 16 v128 registers (each row is 2 v128s).
#[allow(dead_code)]
#[inline(always)]
fn transpose_8x8(input: &[I32X4; 16], into: &mut [I32X4; 16]) {
  // Process as four 4x4 blocks:
  // Input layout: row i = [input[2*i], input[2*i+1]]
  //
  //  [A B]
  //  [C D]
  //
  // Output layout after transpose:
  //  [A' C']
  //  [B' D']

  // Extract the four 4x4 quadrants
  let a_in = [input[0], input[2], input[4], input[6]];     // Top-left
  let b_in = [input[1], input[3], input[5], input[7]];     // Top-right
  let c_in = [input[8], input[10], input[12], input[14]];  // Bottom-left
  let d_in = [input[9], input[11], input[13], input[15]];  // Bottom-right

  let mut a_out: [I32X4; 4] = [I32X4::zero(); 4];
  let mut b_out: [I32X4; 4] = [I32X4::zero(); 4];
  let mut c_out: [I32X4; 4] = [I32X4::zero(); 4];
  let mut d_out: [I32X4; 4] = [I32X4::zero(); 4];

  transpose_4x4(&a_in, &mut a_out);
  transpose_4x4(&b_in, &mut b_out);
  transpose_4x4(&c_in, &mut c_out);
  transpose_4x4(&d_in, &mut d_out);

  // Reassemble: output row i = [a_out or c_out, b_out or d_out]
  // First 4 output rows come from A' (left) and B' (right) -> [A', B']
  // But wait - we need [A' C'] and [B' D'] for the transposed layout
  // Actually for transpose: output row i gets column i from input
  // Row 0-3 of output = columns 0-3 = [A', C']
  // Row 4-7 of output = columns 4-7 = [B', D']

  into[0] = a_out[0];
  into[1] = c_out[0];
  into[2] = a_out[1];
  into[3] = c_out[1];
  into[4] = a_out[2];
  into[5] = c_out[2];
  into[6] = a_out[3];
  into[7] = c_out[3];

  into[8] = b_out[0];
  into[9] = d_out[0];
  into[10] = b_out[1];
  into[11] = d_out[1];
  into[12] = b_out[2];
  into[13] = d_out[2];
  into[14] = b_out[3];
  into[15] = d_out[3];
}

#[inline(always)]
fn shift_left(a: I32X4, shift: u32) -> I32X4 {
  I32X4::new(i32x4_shl(a.vec(), shift))
}

#[inline(always)]
fn shift_right(a: I32X4, shift: u32) -> I32X4 {
  let rounding = i32x4_splat(1i32.wrapping_shl(shift.saturating_sub(1)));
  let rounded = if shift > 0 {
    i32x4_add(a.vec(), rounding)
  } else {
    a.vec()
  };
  I32X4::new(i32x4_shr(rounded, shift))
}

#[inline(always)]
fn round_shift_array(arr: &mut [I32X4], bit: i8) {
  if bit == 0 {
    return;
  }
  if bit > 0 {
    let shift = bit as u32;
    for chunk in arr.iter_mut() {
      *chunk = shift_right(*chunk, shift);
    }
  } else {
    let shift = (-bit) as u32;
    for chunk in arr.iter_mut() {
      *chunk = shift_left(*chunk, shift);
    }
  }
}

/// Main forward transform function for wasm32 SIMD128.
///
/// This processes the input in groups of 4 values (128-bit SIMD).
#[allow(clippy::identity_op)]
fn forward_transform_simd128<T: Coefficient>(
  input: &[i16], output: &mut [MaybeUninit<T>], stride: usize,
  tx_size: TxSize, tx_type: TxType, bd: usize,
) {
  let txfm_size_col = tx_size.width();
  let txfm_size_row = tx_size.height();

  // Buffer layout: buf[row_group * txfm_size_col + col] = I32X4 with 4 row values
  // Number of row groups = ceil(txfm_size_row / 4)
  let num_row_groups = (txfm_size_row + 3) / 4;

  // SAFETY: I32X4 is Copy and can be safely left uninitialized
  let mut tmp: Aligned<[I32X4; 64 * 64 / 4]> = unsafe { Aligned::uninitialized() };
  let buf = &mut tmp.data[..txfm_size_col * num_row_groups];

  let cfg = Txfm2DFlipCfg::fwd(tx_type, tx_size, bd);

  let txfm_func_col = get_func(cfg.txfm_type_col);
  let txfm_func_row = get_func(cfg.txfm_type_row);

  // Process columns in groups of 4
  for cg in (0..txfm_size_col).step_by(4) {
    let shift = cfg.shift[0] as u32;

    // Allocate column buffer
    let mut tx_in_storage = [MaybeUninit::<I32X4>::uninit(); 64];
    let tx_in = &mut tx_in_storage[..txfm_size_row];

    // Calculate how many columns we're processing (may be less than 4 at edge)
    let cols_in_group = (txfm_size_col - cg).min(4);

    if cfg.ud_flip {
      // flip upside down
      for (r, out_reg) in tx_in.iter_mut().enumerate().take(txfm_size_row) {
        let src_row = txfm_size_row - r - 1;
        let mut vals = [0i32; 4];
        for c in 0..cols_in_group {
          vals[c] = i32::from(input[src_row * stride + cg + c]);
        }
        let v = i32x4(vals[0], vals[1], vals[2], vals[3]);
        *out_reg = MaybeUninit::new(shift_left(I32X4::new(v), shift));
      }
    } else {
      for (r, out_reg) in tx_in.iter_mut().enumerate().take(txfm_size_row) {
        let mut vals = [0i32; 4];
        for c in 0..cols_in_group {
          vals[c] = i32::from(input[r * stride + cg + c]);
        }
        let v = i32x4(vals[0], vals[1], vals[2], vals[3]);
        *out_reg = MaybeUninit::new(shift_left(I32X4::new(v), shift));
      }
    }

    // SAFETY: We just initialized all elements
    let col_coeffs = unsafe { slice_assume_init_mut(tx_in) };

    // Apply column transform
    txfm_func_col(col_coeffs);
    round_shift_array(col_coeffs, -cfg.shift[1]);

    // Transpose and store to buffer
    // For wasm32, we process 4 columns at a time, so we need to handle
    // the storage layout carefully.

    // The buffer layout is: buf[row / 4 * txfm_size_col + col]
    // After column transform, col_coeffs[r] contains 4 transformed values
    // for columns cg..cg+4 at row r.

    // Transpose and store to buffer.
    // Buffer layout: buf[row_group * txfm_size_col + col] = I32X4 with 4 row values for that column
    // After transpose, transposed[i] contains values for column (cg + i), with 4 rows packed.
    // We process row groups of 4 rows at a time.

    for rg in (0..txfm_size_row).step_by(4) {
      let row_group = rg / 4;
      let input_block = &col_coeffs[rg..rg + 4.min(txfm_size_row - rg)];

      let mut transposed: [I32X4; 4] = [I32X4::zero(); 4];

      // Pad input block to 4 if needed
      let mut padded: [I32X4; 4] = [I32X4::zero(); 4];
      for (i, v) in input_block.iter().enumerate() {
        padded[i] = *v;
      }

      transpose_4x4(&padded, &mut transposed);

      // Store: transposed[i] goes to column (cg + i), or flipped position if lr_flip
      for i in 0..4.min(txfm_size_col - cg) {
        let src_col = cg + i;
        let dst_col = if cfg.lr_flip {
          txfm_size_col - src_col - 1
        } else {
          src_col
        };
        buf[row_group * txfm_size_col + dst_col] = transposed[i];
      }
    }
  }

  // Process rows in groups of 4
  // After column transform + transpose, buf[row_group * txfm_size_col + col] contains
  // the 4 row values (for rows row_group*4 .. row_group*4+4) at that column.
  // The row transform operates on all columns at once, transforming 4 rows in parallel.

  let num_row_groups = (txfm_size_row + 3) / 4;

  for row_group in 0..num_row_groups {
    let row_coeffs =
      &mut buf[row_group * txfm_size_col..(row_group + 1) * txfm_size_col];

    // Apply row transform - this transforms 4 rows (packed in I32X4) in parallel
    txfm_func_row(row_coeffs);
    round_shift_array(row_coeffs, -cfg.shift[2]);

    // Store output
    // Each row_coeffs[col] contains transformed values for 4 rows at that column
    // We need to extract and store individually

    let rows_in_group = (txfm_size_row - row_group * 4).min(4);
    let output_stride = txfm_size_row.min(32);

    for r_offset in 0..rows_in_group {
      let r = row_group * 4 + r_offset;
      let output_base =
        (r >= 32) as usize * output_stride * txfm_size_col.min(32);

      for cg in (0..txfm_size_col).step_by(32) {
        let output_offset = output_base + txfm_size_row * cg;

        for c in 0..txfm_size_col.min(32) {
          let col_idx = cg + c;

          // Extract the value for row r_offset from row_coeffs[col_idx]
          let val = match r_offset {
            0 => i32x4_extract_lane::<0>(row_coeffs[col_idx].vec()),
            1 => i32x4_extract_lane::<1>(row_coeffs[col_idx].vec()),
            2 => i32x4_extract_lane::<2>(row_coeffs[col_idx].vec()),
            3 => i32x4_extract_lane::<3>(row_coeffs[col_idx].vec()),
            _ => unreachable!(),
          };

          output[output_offset + c * output_stride + (r & 31)]
            .write(T::cast_from(val));
        }
      }
    }
  }
}

/// Forward transform entry point for wasm32 SIMD128.
///
/// # Panics
///
/// - If called with an invalid combination of `tx_size` and `tx_type`
#[inline]
pub fn forward_transform<T: Coefficient>(
  input: &[i16], output: &mut [MaybeUninit<T>], stride: usize,
  tx_size: TxSize, tx_type: TxType, bd: usize, cpu: CpuFeatureLevel,
) {
  assert!(valid_av1_transform(tx_size, tx_type));

  if cpu >= CpuFeatureLevel::SIMD128 {
    forward_transform_simd128(input, output, stride, tx_size, tx_type, bd);
  } else {
    rust::forward_transform(input, output, stride, tx_size, tx_type, bd, cpu);
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::transform::get_valid_txfm_types;
  use rand::Rng;

  /// Test that SIMD and Rust implementations produce the same results
  fn test_forward_transform_matches(tx_size: TxSize, tx_type: TxType) {
    let mut rng = rand::rng();

    let area = tx_size.area();
    let width = tx_size.width();

    let input: Vec<i16> =
      (0..area).map(|_| rng.random_range(-255..256)).collect();

    let mut output_rust = vec![MaybeUninit::new(0i16); area];
    let mut output_simd = vec![MaybeUninit::new(0i16); area];

    rust::forward_transform(
      &input,
      &mut output_rust,
      width,
      tx_size,
      tx_type,
      8,
      CpuFeatureLevel::RUST,
    );

    forward_transform_simd128(
      &input,
      &mut output_simd,
      width,
      tx_size,
      tx_type,
      8,
    );

    let output_rust = unsafe { slice_assume_init_mut(&mut output_rust) };
    let output_simd = unsafe { slice_assume_init_mut(&mut output_simd) };

    assert_eq!(
      output_rust, output_simd,
      "Mismatch for {:?} {:?}",
      tx_size, tx_type
    );
  }

  #[test]
  fn test_forward_transform_4x4() {
    for &tx_type in get_valid_txfm_types(TxSize::TX_4X4) {
      test_forward_transform_matches(TxSize::TX_4X4, tx_type);
    }
  }

  #[test]
  fn test_forward_transform_8x8() {
    for &tx_type in get_valid_txfm_types(TxSize::TX_8X8) {
      test_forward_transform_matches(TxSize::TX_8X8, tx_type);
    }
  }

  #[test]
  fn test_forward_transform_16x16() {
    for &tx_type in get_valid_txfm_types(TxSize::TX_16X16) {
      test_forward_transform_matches(TxSize::TX_16X16, tx_type);
    }
  }

  #[test]
  fn test_forward_transform_32x32() {
    for &tx_type in get_valid_txfm_types(TxSize::TX_32X32) {
      test_forward_transform_matches(TxSize::TX_32X32, tx_type);
    }
  }

  #[test]
  fn test_forward_transform_4x8() {
    for &tx_type in get_valid_txfm_types(TxSize::TX_4X8) {
      test_forward_transform_matches(TxSize::TX_4X8, tx_type);
    }
  }

  #[test]
  fn test_forward_transform_8x4() {
    for &tx_type in get_valid_txfm_types(TxSize::TX_8X4) {
      test_forward_transform_matches(TxSize::TX_8X4, tx_type);
    }
  }

  #[test]
  fn test_forward_transform_8x16() {
    for &tx_type in get_valid_txfm_types(TxSize::TX_8X16) {
      test_forward_transform_matches(TxSize::TX_8X16, tx_type);
    }
  }

  #[test]
  fn test_forward_transform_16x8() {
    for &tx_type in get_valid_txfm_types(TxSize::TX_16X8) {
      test_forward_transform_matches(TxSize::TX_16X8, tx_type);
    }
  }

  #[test]
  fn test_forward_transform_16x32() {
    for &tx_type in get_valid_txfm_types(TxSize::TX_16X32) {
      test_forward_transform_matches(TxSize::TX_16X32, tx_type);
    }
  }

  #[test]
  fn test_forward_transform_32x16() {
    for &tx_type in get_valid_txfm_types(TxSize::TX_32X16) {
      test_forward_transform_matches(TxSize::TX_32X16, tx_type);
    }
  }
}
