// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! WASM32 SIMD-optimized quantization operations

use std::mem::{size_of, MaybeUninit};

use crate::cpu_features::CpuFeatureLevel;
use crate::quantize::{ac_q, dc_q, get_log_tx_scale, rust};
use crate::transform::TxSize;
use crate::util::{CastFromPrimitive, Coefficient};

use core::arch::wasm32::*;

/// Dequantize function with SIMD acceleration
#[inline(always)]
pub fn dequantize<T: Coefficient>(
  qindex: u8, coeffs: &[T], eob: u16, rcoeffs: &mut [MaybeUninit<T>],
  tx_size: TxSize, bit_depth: usize, dc_delta_q: i8, ac_delta_q: i8,
  cpu: CpuFeatureLevel,
) {
  if cpu >= CpuFeatureLevel::SIMD128 {
    if size_of::<T>() == 2 {
      // SIMD for i16 coefficients (8-bit pixel depth)
      dequantize_simd_i16(
        qindex, coeffs, eob, rcoeffs, tx_size, bit_depth, dc_delta_q, ac_delta_q,
      );
    } else {
      // SIMD for i32 coefficients (HBD / 10-bit pixel depth)
      dequantize_simd_i32(
        qindex, coeffs, eob, rcoeffs, tx_size, bit_depth, dc_delta_q, ac_delta_q,
      );
    }
  } else {
    rust::dequantize(
      qindex, coeffs, eob, rcoeffs, tx_size, bit_depth, dc_delta_q, ac_delta_q,
      cpu,
    );
  }
}

/// SIMD dequantize for i16 coefficients
#[inline(always)]
fn dequantize_simd_i16<T: Coefficient>(
  qindex: u8, coeffs: &[T], _eob: u16, rcoeffs: &mut [MaybeUninit<T>],
  tx_size: TxSize, bit_depth: usize, dc_delta_q: i8, ac_delta_q: i8,
) {
  let log_tx_scale = get_log_tx_scale(tx_size) as i32;
  let offset = (1 << log_tx_scale) - 1;

  let dc_quant = dc_q(qindex, dc_delta_q, bit_depth).get() as i32;
  let ac_quant = ac_q(qindex, ac_delta_q, bit_depth).get() as i32;

  // Process DC coefficient (first element)
  let dc_coeff = i32::cast_from(coeffs[0]);
  let dc_result = (dc_coeff * dc_quant + ((dc_coeff >> 31) & offset)) >> log_tx_scale;
  rcoeffs[0].write(T::cast_from(dc_result));

  let len = coeffs.len().min(rcoeffs.len());
  if len <= 1 {
    return;
  }

  // Convert to pointers for SIMD processing
  let coeffs_ptr = coeffs.as_ptr() as *const i16;
  let rcoeffs_ptr = rcoeffs.as_mut_ptr() as *mut i16;

  let ac_quant_vec = i32x4_splat(ac_quant);
  let offset_vec = i32x4_splat(offset);
  
  let mut i = 1;

  // Process 8 AC coefficients at a time
  while i + 8 <= len {
    unsafe {
      // Load 8 i16 coefficients
      let coeffs_v = v128_load(coeffs_ptr.add(i) as *const v128);
      
      // Extend to i32 (low and high)
      let coeffs_lo = i32x4_extend_low_i16x8(coeffs_v);
      let coeffs_hi = i32x4_extend_high_i16x8(coeffs_v);
      
      // Process low 4 coefficients:
      // result = (c * quant + ((c >> 31) & offset)) >> log_tx_scale
      let sign_lo = i32x4_shr(coeffs_lo, 31);
      let offset_lo = v128_and(sign_lo, offset_vec);
      let prod_lo = i32x4_mul(coeffs_lo, ac_quant_vec);
      let sum_lo = i32x4_add(prod_lo, offset_lo);
      let result_lo = i32x4_shr(sum_lo, log_tx_scale as u32);
      
      // Process high 4 coefficients
      let sign_hi = i32x4_shr(coeffs_hi, 31);
      let offset_hi = v128_and(sign_hi, offset_vec);
      let prod_hi = i32x4_mul(coeffs_hi, ac_quant_vec);
      let sum_hi = i32x4_add(prod_hi, offset_hi);
      let result_hi = i32x4_shr(sum_hi, log_tx_scale as u32);
      
      // Pack back to i16 (truncate)
      let result = i16x8_narrow_i32x4(result_lo, result_hi);
      
      // Store result
      v128_store(rcoeffs_ptr.add(i) as *mut v128, result);
    }
    i += 8;
  }

  // Process remaining 4 coefficients if possible
  if i + 4 <= len {
    unsafe {
      let coeffs_v = v128_load64_zero(coeffs_ptr.add(i) as *const u64);
      let coeffs_32 = i32x4_extend_low_i16x8(coeffs_v);
      
      let sign = i32x4_shr(coeffs_32, 31);
      let offset_masked = v128_and(sign, offset_vec);
      let prod = i32x4_mul(coeffs_32, ac_quant_vec);
      let sum = i32x4_add(prod, offset_masked);
      let result_32 = i32x4_shr(sum, log_tx_scale as u32);
      
      // Pack and store (only lower 64 bits used)
      let result = i16x8_narrow_i32x4(result_32, i32x4_splat(0));
      v128_store64_lane::<0>(result, rcoeffs_ptr.add(i) as *mut u64);
    }
    i += 4;
  }

  // Scalar remainder
  while i < len {
    let coeff = i32::cast_from(coeffs[i]);
    let result = (coeff * ac_quant + ((coeff >> 31) & offset)) >> log_tx_scale;
    rcoeffs[i].write(T::cast_from(result));
    i += 1;
  }
}

/// SIMD dequantize for i32 coefficients (High Bit Depth / 10-bit)
#[inline(always)]
fn dequantize_simd_i32<T: Coefficient>(
  qindex: u8, coeffs: &[T], _eob: u16, rcoeffs: &mut [MaybeUninit<T>],
  tx_size: TxSize, bit_depth: usize, dc_delta_q: i8, ac_delta_q: i8,
) {
  let log_tx_scale = get_log_tx_scale(tx_size) as i32;
  let offset = (1 << log_tx_scale) - 1;

  let dc_quant = dc_q(qindex, dc_delta_q, bit_depth).get() as i32;
  let ac_quant = ac_q(qindex, ac_delta_q, bit_depth).get() as i32;

  // Process DC coefficient (first element) 
  let dc_coeff = i32::cast_from(coeffs[0]);
  let dc_result = (dc_coeff * dc_quant + ((dc_coeff >> 31) & offset)) >> log_tx_scale;
  rcoeffs[0].write(T::cast_from(dc_result));

  let len = coeffs.len().min(rcoeffs.len());
  if len <= 1 {
    return;
  }

  // Convert to pointers for SIMD processing
  let coeffs_ptr = coeffs.as_ptr() as *const i32;
  let rcoeffs_ptr = rcoeffs.as_mut_ptr() as *mut i32;

  let ac_quant_vec = i32x4_splat(ac_quant);
  let offset_vec = i32x4_splat(offset);
  
  let mut i = 1;

  // Process 4 i32 coefficients at a time
  while i + 4 <= len {
    unsafe {
      // Load 4 i32 coefficients
      let coeffs_v = v128_load(coeffs_ptr.add(i) as *const v128);
      
      // result = (c * quant + ((c >> 31) & offset)) >> log_tx_scale
      let sign = i32x4_shr(coeffs_v, 31);
      let offset_masked = v128_and(sign, offset_vec);
      let prod = i32x4_mul(coeffs_v, ac_quant_vec);
      let sum = i32x4_add(prod, offset_masked);
      let result = i32x4_shr(sum, log_tx_scale as u32);
      
      // Store result
      v128_store(rcoeffs_ptr.add(i) as *mut v128, result);
    }
    i += 4;
  }

  // Scalar remainder
  while i < len {
    let coeff = i32::cast_from(coeffs[i]);
    let result = (coeff * ac_quant + ((coeff >> 31) & offset)) >> log_tx_scale;
    rcoeffs[i].write(T::cast_from(result));
    i += 1;
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::context::av1_get_coded_tx_size;
  use crate::quantize::{ac_q, dc_q};
  use crate::transform::TxSize::*;

  #[test]
  fn test_dequantize_8bit() {
    // Test that SIMD matches Rust implementation
    let tx_size = TX_8X8;
    let bd: usize = 8;
    let qindex: u8 = 50;
    let area = av1_get_coded_tx_size(tx_size).area();

    // Create test coefficients
    let mut coeffs = vec![0i16; area];
    for (i, coeff) in coeffs.iter_mut().enumerate() {
      *coeff = ((i as i16 % 20) - 10) * 2;
    }

    // Rust reference
    let mut rcoeffs_rust = vec![MaybeUninit::new(0i16); area];
    rust::dequantize(
      qindex,
      &coeffs,
      area as u16,
      &mut rcoeffs_rust,
      tx_size,
      bd,
      0,
      0,
      CpuFeatureLevel::RUST,
    );

    // SIMD version
    let mut rcoeffs_simd = vec![MaybeUninit::new(0i16); area];
    dequantize(
      qindex,
      &coeffs,
      area as u16,
      &mut rcoeffs_simd,
      tx_size,
      bd,
      0,
      0,
      CpuFeatureLevel::SIMD128,
    );

    // Compare
    for i in 0..area {
      let rust_val = unsafe { rcoeffs_rust[i].assume_init() };
      let simd_val = unsafe { rcoeffs_simd[i].assume_init() };
      assert_eq!(
        rust_val, simd_val,
        "Mismatch at index {}: rust={}, simd={}",
        i, rust_val, simd_val
      );
    }
  }

  #[test]
  fn test_dequantize_various_sizes() {
    for &tx_size in &[TX_4X4, TX_8X8, TX_16X16, TX_4X8, TX_8X4] {
      let bd: usize = 8;
      let qindex: u8 = 100;
      let area = av1_get_coded_tx_size(tx_size).area();

      let mut coeffs = vec![0i16; area];
      for (i, coeff) in coeffs.iter_mut().enumerate() {
        *coeff = if i % 3 == 0 { -(i as i16) } else { i as i16 };
      }

      let mut rcoeffs_rust = vec![MaybeUninit::new(0i16); area];
      rust::dequantize(
        qindex,
        &coeffs,
        area as u16,
        &mut rcoeffs_rust,
        tx_size,
        bd,
        0,
        0,
        CpuFeatureLevel::RUST,
      );

      let mut rcoeffs_simd = vec![MaybeUninit::new(0i16); area];
      dequantize(
        qindex,
        &coeffs,
        area as u16,
        &mut rcoeffs_simd,
        tx_size,
        bd,
        0,
        0,
        CpuFeatureLevel::SIMD128,
      );

      for i in 0..area {
        let rust_val = unsafe { rcoeffs_rust[i].assume_init() };
        let simd_val = unsafe { rcoeffs_simd[i].assume_init() };
        assert_eq!(
          rust_val, simd_val,
          "Mismatch at index {} for {:?}: rust={}, simd={}",
          i, tx_size, rust_val, simd_val
        );
      }
    }
  }
}
