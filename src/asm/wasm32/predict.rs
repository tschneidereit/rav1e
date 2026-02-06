// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! SIMD-accelerated intra prediction functions for wasm32.

use crate::cpu_features::CpuFeatureLevel;
use crate::partition::{BlockSize, IntraEdge};
use crate::predict::{
  rust, IntraEdgeFilterParameters, PredictionMode, PredictionVariant,
};
use crate::tiling::{PlaneRegion, PlaneRegionMut};
use crate::transform::TxSize;
use crate::util::Pixel;
use v_frame::pixel::PixelType;
use std::mem::MaybeUninit;

use core::arch::wasm32::*;

use super::simd_helpers::*;

/// Main dispatch function for intra prediction.
#[inline(always)]
pub fn dispatch_predict_intra<T: Pixel>(
  mode: PredictionMode, variant: PredictionVariant,
  dst: &mut PlaneRegionMut<'_, T>, tx_size: TxSize, bit_depth: usize,
  ac: &[i16], angle: isize, ief_params: Option<IntraEdgeFilterParameters>,
  edge_buf: &IntraEdge<T>, cpu: CpuFeatureLevel,
) {
  let width = tx_size.width();
  let height = tx_size.height();

  // For non-SIMD128 or complex cases, fall back to Rust
  if cpu < CpuFeatureLevel::SIMD128 {
    rust::dispatch_predict_intra(
      mode, variant, dst, tx_size, bit_depth, ac, angle, ief_params, edge_buf,
      cpu,
    );
    return;
  }

  let (left, top_left, above) = edge_buf.as_slices();
  let above_slice = above;
  let left_slice = &left[left.len().saturating_sub(height)..];

  match T::type_enum() {
    PixelType::U8 => {
      // 8-bit path with SIMD
      match mode {
        PredictionMode::DC_PRED => {
          match variant {
            PredictionVariant::NONE => {
              pred_dc_128_simd(dst, width, height, bit_depth);
            }
            PredictionVariant::LEFT => {
              pred_dc_left_simd(dst, left_slice, width, height);
            }
            PredictionVariant::TOP => {
              pred_dc_top_simd(dst, above_slice, width, height);
            }
            PredictionVariant::BOTH => {
              pred_dc_simd(dst, above_slice, left_slice, width, height);
            }
          }
        }
        PredictionMode::V_PRED if angle == 90 => {
          pred_v_simd(dst, above_slice, width, height);
        }
        PredictionMode::H_PRED if angle == 180 => {
          pred_h_simd(dst, left_slice, width, height);
        }
        PredictionMode::SMOOTH_PRED => {
          pred_smooth_simd(dst, above_slice, left_slice, width, height);
        }
        PredictionMode::SMOOTH_V_PRED => {
          pred_smooth_v_simd(dst, above_slice, left_slice, width, height);
        }
        PredictionMode::SMOOTH_H_PRED => {
          pred_smooth_h_simd(dst, above_slice, left_slice, width, height);
        }
        PredictionMode::PAETH_PRED => {
          pred_paeth_simd(dst, above_slice, left_slice, top_left[0], width, height);
        }
        // Fall back to Rust for directional and CFL modes
        _ => {
          rust::dispatch_predict_intra(
            mode, variant, dst, tx_size, bit_depth, ac, angle, ief_params,
            edge_buf, cpu,
          );
        }
      }
    }
    PixelType::U16 => {
      // For HBD, fall back to Rust for now
      rust::dispatch_predict_intra(
        mode, variant, dst, tx_size, bit_depth, ac, angle, ief_params, edge_buf,
        cpu,
      );
    }
  }
}

// ============================================================================
// DC Prediction
// ============================================================================

/// DC prediction with 128 as the value (no neighbors available)
#[inline(always)]
fn pred_dc_128_simd<T: Pixel>(
  output: &mut PlaneRegionMut<'_, T>, width: usize, height: usize,
  bit_depth: usize,
) {
  let v = T::cast_from(128u32 << (bit_depth - 8));
  for line in output.rows_iter_mut().take(height) {
    line[..width].fill(v);
  }
}

/// DC prediction using left neighbors only
#[inline(always)]
fn pred_dc_left_simd<T: Pixel>(
  output: &mut PlaneRegionMut<'_, T>, left: &[T], width: usize, height: usize,
) {
  // Sum left pixels using SIMD
  let sum = sum_pixels_simd(left, height);
  let avg = T::cast_from((sum + (height >> 1) as u32) / height as u32);
  
  for line in output.rows_iter_mut().take(height) {
    line[..width].fill(avg);
  }
}

/// DC prediction using top neighbors only
#[inline(always)]
fn pred_dc_top_simd<T: Pixel>(
  output: &mut PlaneRegionMut<'_, T>, above: &[T], width: usize, height: usize,
) {
  // Sum above pixels using SIMD
  let sum = sum_pixels_simd(above, width);
  let avg = T::cast_from((sum + (width >> 1) as u32) / width as u32);
  
  for line in output.rows_iter_mut().take(height) {
    line[..width].fill(avg);
  }
}

/// DC prediction using both left and top neighbors
#[inline(always)]
fn pred_dc_simd<T: Pixel>(
  output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T], width: usize,
  height: usize,
) {
  // Sum both left and above pixels
  let sum_left = sum_pixels_simd(left, height);
  let sum_above = sum_pixels_simd(above, width);
  let sum = sum_left + sum_above;
  let len = (width + height) as u32;
  let avg = T::cast_from((sum + (len >> 1)) / len);
  
  for line in output.rows_iter_mut().take(height) {
    line[..width].fill(avg);
  }
}

/// Sum pixels using SIMD where possible
#[inline(always)]
fn sum_pixels_simd<T: Pixel>(pixels: &[T], count: usize) -> u32 {
  match T::type_enum() {
    PixelType::U8 => {
      let pixels_u8 = unsafe {
        std::slice::from_raw_parts(pixels.as_ptr() as *const u8, count)
      };
      sum_u8_simd(pixels_u8)
    }
    PixelType::U16 => {
      let pixels_u16 = unsafe {
        std::slice::from_raw_parts(pixels.as_ptr() as *const u16, count)
      };
      sum_u16_scalar(pixels_u16)
    }
  }
}

/// Sum u8 values using SIMD
#[inline(always)]
fn sum_u8_simd(pixels: &[u8]) -> u32 {
  let mut sum = 0u32;
  let mut i = 0;
  
  // Process 16 bytes at a time
  while i + 16 <= pixels.len() {
    unsafe {
      let v = v128_load(pixels.as_ptr().add(i) as *const v128);
      sum += horizontal_sum_u8x16(v);
    }
    i += 16;
  }
  
  // Handle remainder
  while i < pixels.len() {
    sum += pixels[i] as u32;
    i += 1;
  }
  
  sum
}

/// Sum u16 values (scalar fallback)
#[inline(always)]
fn sum_u16_scalar(pixels: &[u16]) -> u32 {
  pixels.iter().map(|&p| p as u32).sum()
}

// ============================================================================
// Horizontal and Vertical Prediction
// ============================================================================

/// Vertical prediction: copy above row to all rows
#[inline(always)]
fn pred_v_simd<T: Pixel>(
  output: &mut PlaneRegionMut<'_, T>, above: &[T], width: usize, height: usize,
) {
  for line in output.rows_iter_mut().take(height) {
    line[..width].copy_from_slice(&above[..width]);
  }
}

/// Horizontal prediction: fill each row with the corresponding left pixel
#[inline(always)]
fn pred_h_simd<T: Pixel>(
  output: &mut PlaneRegionMut<'_, T>, left: &[T], width: usize, height: usize,
) {
  for (line, l) in output.rows_iter_mut().zip(left[..height].iter().rev()) {
    line[..width].fill(*l);
  }
}

// ============================================================================
// Smooth Prediction - Compiler-friendly implementation
// ============================================================================

/// Smooth prediction (both horizontal and vertical smoothing)
/// Uses scalar code structured for auto-vectorization by the compiler
#[inline(always)]
fn pred_smooth_simd<T: Pixel>(
  output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T], width: usize,
  height: usize,
) {
  // Delegate to Rust implementation which compiles well with SIMD128
  rust::pred_smooth(output, above, left, width, height);
}

/// Smooth vertical prediction
#[inline(always)]
fn pred_smooth_v_simd<T: Pixel>(
  output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T], width: usize,
  height: usize,
) {
  rust::pred_smooth_v(output, above, left, width, height);
}

/// Smooth horizontal prediction
#[inline(always)]
fn pred_smooth_h_simd<T: Pixel>(
  output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T], width: usize,
  height: usize,
) {
  rust::pred_smooth_h(output, above, left, width, height);
}

// ============================================================================
// Paeth Prediction
// ============================================================================

/// Paeth prediction - chooses between left, top, and top-left based on gradient
/// Uses SIMD for the absolute difference comparisons and conditional selection.
#[inline(always)]
fn pred_paeth_simd<T: Pixel>(
  output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T], above_left: T,
  width: usize, height: usize,
) {
  match T::type_enum() {
    PixelType::U8 => {
      pred_paeth_simd_u8(output, above, left, above_left, width, height);
    }
    PixelType::U16 => {
      rust::pred_paeth(output, above, left, above_left, width, height);
    }
  }
}

/// Paeth prediction for 8-bit pixels using SIMD
#[inline(always)]
fn pred_paeth_simd_u8<T: Pixel>(
  output: &mut PlaneRegionMut<'_, T>, above: &[T], left: &[T], above_left: T,
  width: usize, height: usize,
) {
  let raw_top_left: i32 = above_left.into();
  let top_left_vec = i32x4_splat(raw_top_left);

  for r in 0..height {
    let row = &mut output[r];
    let raw_left: i32 = left[height - 1 - r].into();
    let left_vec = i32x4_splat(raw_left);

    let mut c = 0;

    // Process 4 pixels at a time with SIMD
    while c + 4 <= width {
      // Load 4 above (top) values
      let top_vec = i32x4(
        above[c].into(), above[c+1].into(),
        above[c+2].into(), above[c+3].into()
      );

      // p_base = top + left - top_left
      let p_base = i32x4_sub(i32x4_add(top_vec, left_vec), top_left_vec);

      // Compute absolute differences
      let diff_left = i32x4_sub(p_base, left_vec);
      let p_left = i32x4_abs(diff_left);

      let diff_top = i32x4_sub(p_base, top_vec);
      let p_top = i32x4_abs(diff_top);

      let diff_top_left = i32x4_sub(p_base, top_left_vec);
      let p_top_left = i32x4_abs(diff_top_left);

      // Select the value with minimum distance
      // if p_left <= p_top && p_left <= p_top_left -> left
      // elif p_top <= p_top_left -> top
      // else -> top_left
      
      let left_le_top = i32x4_le(p_left, p_top);
      let left_le_tl = i32x4_le(p_left, p_top_left);
      let select_left = v128_and(left_le_top, left_le_tl);
      
      let top_le_tl = i32x4_le(p_top, p_top_left);
      let not_select_left = v128_not(select_left);
      let select_top = v128_and(not_select_left, top_le_tl);
      
      // Use relaxed laneselect if available for faster blending
      #[cfg(target_feature = "relaxed-simd")]
      let result = {
        // First select between left and top_left based on select_left
        let temp = i32x4_relaxed_laneselect(left_vec, top_left_vec, select_left);
        // Then select between that and top based on select_top
        i32x4_relaxed_laneselect(top_vec, temp, select_top)
      };
      
      #[cfg(not(target_feature = "relaxed-simd"))]
      let result = {
        // Standard bitwise selection
        let select_tl = v128_andnot(v128_or(select_left, select_top), v128_not(i32x4_splat(0)));
        v128_or(
          v128_or(
            v128_and(select_left, left_vec),
            v128_and(select_top, top_vec)
          ),
          v128_and(select_tl, top_left_vec)
        )
      };

      // Extract and store
      row[c] = T::cast_from(i32x4_extract_lane::<0>(result) as u32);
      row[c+1] = T::cast_from(i32x4_extract_lane::<1>(result) as u32);
      row[c+2] = T::cast_from(i32x4_extract_lane::<2>(result) as u32);
      row[c+3] = T::cast_from(i32x4_extract_lane::<3>(result) as u32);

      c += 4;
    }

    // Handle remainder with scalar code
    while c < width {
      let raw_top: i32 = above[c].into();
      let p_base = raw_top + raw_left - raw_top_left;

      let p_left = (p_base - raw_left).abs();
      let p_top = (p_base - raw_top).abs();
      let p_top_left = (p_base - raw_top_left).abs();

      row[c] = if p_left <= p_top && p_left <= p_top_left {
        T::cast_from(raw_left)
      } else if p_top <= p_top_left {
        T::cast_from(raw_top)
      } else {
        T::cast_from(raw_top_left)
      };
      c += 1;
    }
  }
}

// Wrapper for pred_cfl_ac - just delegates to the rust implementation
// This is needed because rust::pred_cfl_ac is pub(crate) and cannot be re-exported
pub fn pred_cfl_ac<T: Pixel, const XDEC: usize, const YDEC: usize>(
  ac: &mut [MaybeUninit<i16>], luma: &PlaneRegion<'_, T>,
  plane_bsize: BlockSize, w_pad: usize, h_pad: usize, cpu: CpuFeatureLevel,
) {
  rust::pred_cfl_ac::<T, XDEC, YDEC>(ac, luma, plane_bsize, w_pad, h_pad, cpu);
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::frame::Plane;
  use crate::partition::IntraEdge;
  use crate::tiling::Area;
  use crate::transform::TxSize;
  use rand::Rng;

  fn create_test_plane<T: Pixel>(width: usize, height: usize) -> Plane<T> {
    Plane::new(width, height, 0, 0, 0, 0)
  }

  // Aligned buffer type for IntraEdge
  #[repr(align(64))]
  struct AlignedEdgeBuf<T> {
    data: [T; 4 * 64 + 1],
  }

  fn test_dc_pred_matches<T: Pixel>(width: usize, height: usize, bit_depth: usize) {
    let mut rng = rand::rng();
    
    // Create edge buffer
    let mut edge_buf: AlignedEdgeBuf<T> = AlignedEdgeBuf {
      data: [T::cast_from(0u16); 4 * 64 + 1],
    };
    
    // Fill with random values
    for p in edge_buf.data.iter_mut() {
      *p = T::cast_from(rng.random_range(0..((1 << bit_depth) - 1) as u16));
    }
    
    let edge = IntraEdge::mock(&crate::util::Aligned::new(edge_buf.data));
    let (left, top_left, above) = edge.as_slices();
    
    // Test DC prediction
    let mut dst_rust = create_test_plane::<T>(width + 16, height + 16);
    let mut dst_simd = create_test_plane::<T>(width + 16, height + 16);
    
    {
      let area = Area::StartingAt { x: 0, y: 0 };
      let mut region = dst_rust.region_mut(area);
      rust::pred_dc(&mut region, above, &left[left.len().saturating_sub(height)..], width, height, bit_depth);
    }
    
    {
      let area = Area::StartingAt { x: 0, y: 0 };
      let mut region = dst_simd.region_mut(area);
      pred_dc_simd(&mut region, above, &left[left.len().saturating_sub(height)..], width, height);
    }
    
    // Compare
    for r in 0..height {
      for c in 0..width {
        let rust_val = dst_rust.data[r * dst_rust.cfg.stride + c];
        let simd_val = dst_simd.data[r * dst_simd.cfg.stride + c];
        assert_eq!(
          rust_val, simd_val,
          "DC pred mismatch at ({}, {}) for {}x{}: rust={:?}, simd={:?}",
          c, r, width, height, rust_val, simd_val
        );
      }
    }
  }

  #[test]
  fn test_dc_pred_4x4() {
    test_dc_pred_matches::<u8>(4, 4, 8);
  }

  #[test]
  fn test_dc_pred_8x8() {
    test_dc_pred_matches::<u8>(8, 8, 8);
  }

  #[test]
  fn test_dc_pred_16x16() {
    test_dc_pred_matches::<u8>(16, 16, 8);
  }

  #[test]
  fn test_dc_pred_32x32() {
    test_dc_pred_matches::<u8>(32, 32, 8);
  }
}
