// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! WASM32 SIMD-optimized quantization operations
//!
//! Note: Currently uses Rust fallback. The SIMD implementation for dequantize
//! was found to have subtle differences in saturation behavior vs truncation
//! that could cause bit-exact mismatches. The performance gain was not
//! significant enough to justify the complexity of matching exact behavior.

use std::mem::MaybeUninit;

use crate::cpu_features::CpuFeatureLevel;
use crate::quantize::rust;
use crate::transform::TxSize;
use crate::util::Coefficient;

/// Dequantize function - currently uses Rust fallback
#[inline(always)]
pub fn dequantize<T: Coefficient>(
  qindex: u8, coeffs: &[T], eob: u16, rcoeffs: &mut [MaybeUninit<T>],
  tx_size: TxSize, bit_depth: usize, dc_delta_q: i8, ac_delta_q: i8,
  cpu: CpuFeatureLevel,
) {
  // Use Rust fallback for now
  rust::dequantize(
    qindex, coeffs, eob, rcoeffs, tx_size, bit_depth, dc_delta_q, ac_delta_q,
    cpu,
  );
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::context::av1_get_coded_tx_size;
  use crate::quantize::{ac_q, dc_q};
  use crate::transform::TxSize::*;

  #[test]
  fn test_dequantize_8bit() {
    // Basic smoke test to ensure the module compiles and works
    let tx_size = TX_8X8;
    let bd: usize = 8;
    let qindex: u8 = 50;
    let area = av1_get_coded_tx_size(tx_size).area();

    let dc_quant = dc_q(qindex, 0, bd).get() as i16;
    let ac_quant = ac_q(qindex, 0, bd).get() as i16;

    // Simple test coefficients that won't overflow
    let mut coeffs = vec![0i16; area];
    for (i, coeff) in coeffs.iter_mut().enumerate() {
      let quant = if i == 0 { dc_quant } else { ac_quant };
      *coeff = ((i as i16 % 10) - 5) / quant.max(1);
    }

    let mut rcoeffs = vec![MaybeUninit::new(0i16); area];
    dequantize(
      qindex,
      &coeffs,
      area as u16,
      &mut rcoeffs,
      tx_size,
      bd,
      0,
      0,
      CpuFeatureLevel::SIMD128,
    );

    // Just verify it runs without panicking
    for i in 0..area {
      let _ = unsafe { rcoeffs[i].assume_init() };
    }
  }
}
