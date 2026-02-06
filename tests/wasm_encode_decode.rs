// Copyright (c) 2024, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! WASM encode/decode tests that run rav1e.wasm via wasmtime CLI and verify
//! output using native dav1d decoder.
//!
//! These tests mirror the native encode_decode tests but use CLI invocation
//! to test the wasm32-wasip2 target.

#![cfg(feature = "wasm_decode_test")]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Get path to the rav1e.wasm binary (set via RAV1E_WASM_PATH env var or default location)
fn get_wasm_path() -> PathBuf {
  env::var("RAV1E_WASM_PATH")
    .map(PathBuf::from)
    .unwrap_or_else(|_| {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-wasip2/release/rav1e.wasm")
    })
}

/// Create a temporary directory for test files
fn create_temp_dir(test_name: &str) -> PathBuf {
  let dir = env::temp_dir().join(format!("rav1e_wasm_test_{}", test_name));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).expect("Failed to create temp dir");
  dir
}

/// Generate a deterministic Y4M file with random data
fn generate_y4m(
  path: &Path, width: usize, height: usize, frames: usize, bit_depth: usize,
  chroma_sampling: &str, seed: u64,
) {
  let mut file = File::create(path).expect("Failed to create Y4M file");
  let mut ra = ChaChaRng::seed_from_u64(seed);

  // Y4M header
  let colorspace = match (chroma_sampling, bit_depth) {
    ("420", 8) => "C420jpeg",
    ("420", 10) => "C420p10",
    ("420", 12) => "C420p12",
    ("422", 8) => "C422",
    ("422", 10) => "C422p10",
    ("422", 12) => "C422p12",
    ("444", 8) => "C444",
    ("444", 10) => "C444p10",
    ("444", 12) => "C444p12",
    ("400", 8) => "Cmono",
    ("400", 10) => "Cmono10",
    ("400", 12) => "Cmono12",
    _ => panic!("Unsupported format"),
  };

  writeln!(file, "YUV4MPEG2 W{width} H{height} F25:1 Ip A1:1 {colorspace}")
    .unwrap();

  let (chroma_width, chroma_height) = match chroma_sampling {
    // For 420 and 422, use proper rounding for odd dimensions
    "420" => ((width + 1) / 2, (height + 1) / 2),
    "422" => ((width + 1) / 2, height),
    "444" => (width, height),
    "400" => (0, 0),
    _ => panic!("Unsupported chroma sampling"),
  };

  for _ in 0..frames {
    write!(file, "FRAME\n").unwrap();

    // Y plane
    for _ in 0..(width * height) {
      if bit_depth > 8 {
        let v: u16 = ra.random::<u16>() >> (16 - bit_depth);
        file.write_all(&v.to_le_bytes()).unwrap();
      } else {
        let v: u8 = ra.random();
        file.write_all(&[v]).unwrap();
      }
    }

    // U and V planes (if not monochrome)
    if chroma_sampling != "400" {
      for _ in 0..2 {
        for _ in 0..(chroma_width * chroma_height) {
          if bit_depth > 8 {
            let v: u16 = ra.random::<u16>() >> (16 - bit_depth);
            file.write_all(&v.to_le_bytes()).unwrap();
          } else {
            let v: u8 = ra.random();
            file.write_all(&[v]).unwrap();
          }
        }
      }
    }
  }
}

/// Run rav1e.wasm via wasmtime to encode video
fn run_wasm_encode(
  wasm_path: &Path, input: &Path, output: &Path, reconstruction: &Path,
  speed: u8, quantizer: u8, low_latency: bool, min_keyint: u64,
  max_keyint: u64, tile_rows: usize, tile_cols: usize, still_picture: bool,
  high_bitdepth: bool,
) -> bool {
  let workdir =
    input.parent().unwrap_or(Path::new(".")).to_string_lossy().to_string();

  let mut cmd = Command::new("wasmtime");
  cmd
    .arg("run")
    .arg("-S")
    .arg("cli")
    .arg(format!("--dir={}", workdir))
    .arg(wasm_path)
    .arg(input)
    .arg("--output")
    .arg(output)
    .arg("--reconstruction")
    .arg(reconstruction)
    .arg("--speed")
    .arg(speed.to_string())
    .arg("--quantizer")
    .arg(quantizer.to_string())
    .arg("--min-keyint")
    .arg(min_keyint.to_string())
    .arg("--keyint")
    .arg(max_keyint.to_string());

  if low_latency {
    cmd.arg("--low-latency");
  }

  if tile_rows > 0 {
    cmd.arg("--tile-rows").arg(tile_rows.to_string());
  }

  if tile_cols > 0 {
    cmd.arg("--tile-cols").arg(tile_cols.to_string());
  }

  if still_picture {
    cmd.arg("--still-picture");
  }

  if high_bitdepth {
    cmd.arg("--high-bitdepth");
  }

  let output = cmd.output().expect("Failed to run wasmtime");

  if !output.status.success() {
    eprintln!(
      "wasmtime stderr: {}",
      String::from_utf8_lossy(&output.stderr)
    );
    return false;
  }
  true
}

/// Run dav1d to decode IVF to Y4M
fn run_dav1d_decode(input: &Path, output: &Path) -> bool {
  let result = Command::new("dav1d")
    .arg("-i")
    .arg(input)
    .arg("-o")
    .arg(output)
    .output()
    .expect("Failed to run dav1d");

  if !result.status.success() {
    eprintln!("dav1d stderr: {}", String::from_utf8_lossy(&result.stderr));
    return false;
  }
  true
}

/// Parse Y4M header and return (width, height, bit_depth, chroma_sampling)
/// chroma_sampling is "420", "422", "444", or "400" (mono)
fn parse_y4m_header(path: &Path) -> Option<(usize, usize, usize, String)> {
  let file = File::open(path).ok()?;
  let mut reader = BufReader::new(file);
  let mut header = String::new();
  reader.read_line(&mut header).ok()?;

  let mut width = 0;
  let mut height = 0;
  let mut bit_depth = 8;
  let mut chroma = "420".to_string(); // default

  for part in header.split_whitespace() {
    if let Some(w) = part.strip_prefix('W') {
      width = w.parse().ok()?;
    } else if let Some(h) = part.strip_prefix('H') {
      height = h.parse().ok()?;
    } else if let Some(c) = part.strip_prefix('C') {
      if c.contains("p10") || c.contains("mono10") {
        bit_depth = 10;
      } else if c.contains("p12") || c.contains("mono12") {
        bit_depth = 12;
      }
      // Parse chroma subsampling
      if c.starts_with("444") {
        chroma = "444".to_string();
      } else if c.starts_with("422") {
        chroma = "422".to_string();
      } else if c.starts_with("mono") {
        chroma = "400".to_string();
      }
      // else default 420
    }
  }

  if width > 0 && height > 0 {
    Some((width, height, bit_depth, chroma))
  } else {
    None
  }
}

/// Compare two Y4M files frame by frame
fn compare_y4m_files(file1: &Path, file2: &Path) -> Result<(), String> {
  let mut f1 = BufReader::new(
    File::open(file1).map_err(|e| format!("Failed to open {file1:?}: {e}"))?,
  );
  let mut f2 = BufReader::new(
    File::open(file2).map_err(|e| format!("Failed to open {file2:?}: {e}"))?,
  );

  // Read and compare headers
  let mut header1 = String::new();
  let mut header2 = String::new();
  f1.read_line(&mut header1)
    .map_err(|e| format!("Failed to read header1: {e}"))?;
  f2.read_line(&mut header2)
    .map_err(|e| format!("Failed to read header2: {e}"))?;

  // Parse dimensions from headers (they might differ slightly in colorspace notation)
  let (w1, h1, bd1, cs1) = parse_y4m_header(file1)
    .ok_or_else(|| "Failed to parse file1 header".to_string())?;
  let (w2, h2, bd2, cs2) = parse_y4m_header(file2)
    .ok_or_else(|| "Failed to parse file2 header".to_string())?;

  if w1 != w2 || h1 != h2 {
    return Err(format!(
      "Dimension mismatch: {}x{} vs {}x{}",
      w1, h1, w2, h2
    ));
  }

  if bd1 != bd2 {
    return Err(format!("Bit depth mismatch: {} vs {}", bd1, bd2));
  }

  if cs1 != cs2 {
    return Err(format!("Chroma sampling mismatch: {} vs {}", cs1, cs2));
  }

  // Compare frame data
  let mut frame_num = 0;
  let mut frame_marker1 = vec![0u8; 6]; // "FRAME\n"
  let mut frame_marker2 = vec![0u8; 6];

  loop {
    let r1 = f1.read_exact(&mut frame_marker1);
    let r2 = f2.read_exact(&mut frame_marker2);

    match (r1, r2) {
      (Err(_), Err(_)) => break, // Both files ended
      (Ok(_), Err(_)) => {
        return Err(format!("File2 ended early at frame {frame_num}"))
      }
      (Err(_), Ok(_)) => {
        return Err(format!("File1 ended early at frame {frame_num}"))
      }
      (Ok(_), Ok(_)) => {}
    }

    // Calculate frame size based on chroma subsampling
    let bytes_per_sample = if bd1 > 8 { 2 } else { 1 };
    let y_size = w1 * h1 * bytes_per_sample;
    let uv_size = match cs1.as_str() {
      "444" => w1 * h1 * bytes_per_sample * 2,
      "422" => ((w1 + 1) / 2) * h1 * bytes_per_sample * 2,
      "420" => ((w1 + 1) / 2) * ((h1 + 1) / 2) * bytes_per_sample * 2,
      "400" => 0, // monochrome
      _ => ((w1 + 1) / 2) * ((h1 + 1) / 2) * bytes_per_sample * 2, // default 420
    };
    let frame_size = y_size + uv_size;

    let mut data1 = vec![0u8; frame_size];
    let mut data2 = vec![0u8; frame_size];

    f1.read_exact(&mut data1)
      .map_err(|e| format!("Failed to read frame {frame_num} from file1: {e}"))?;
    f2.read_exact(&mut data2)
      .map_err(|e| format!("Failed to read frame {frame_num} from file2: {e}"))?;

    if data1 != data2 {
      // Find first differing byte for better error message
      for (i, (b1, b2)) in data1.iter().zip(data2.iter()).enumerate() {
        if b1 != b2 {
          return Err(format!(
            "Frame {frame_num} differs at byte {i}: {b1} vs {b2}"
          ));
        }
      }
    }

    frame_num += 1;
  }

  Ok(())
}

/// Run a complete encode-decode-compare test
fn run_test(
  test_name: &str, width: usize, height: usize, frames: usize, speed: u8,
  quantizer: u8, bit_depth: usize, chroma_sampling: &str, min_keyint: u64,
  max_keyint: u64, low_latency: bool, tile_rows: usize, tile_cols: usize,
  still_picture: bool, seed: u64,
) {
  let wasm_path = get_wasm_path();
  if !wasm_path.exists() {
    panic!(
      "rav1e.wasm not found at {:?}. Build with: cargo build --target wasm32-wasip2 --release --no-default-features --features binaries",
      wasm_path
    );
  }

  let temp_dir = create_temp_dir(test_name);
  let input_path = temp_dir.join("input.y4m");
  let output_path = temp_dir.join("output.ivf");
  let rec_path = temp_dir.join("reconstruction.y4m");
  let dec_path = temp_dir.join("decoded.y4m");

  // Generate input
  generate_y4m(
    &input_path,
    width,
    height,
    frames,
    bit_depth,
    chroma_sampling,
    seed,
  );

  // Encode with rav1e.wasm
  let high_bitdepth = bit_depth > 8;
  assert!(
    run_wasm_encode(
      &wasm_path,
      &input_path,
      &output_path,
      &rec_path,
      speed,
      quantizer,
      low_latency,
      min_keyint,
      max_keyint,
      tile_rows,
      tile_cols,
      still_picture,
      high_bitdepth,
    ),
    "Encoding failed for test {test_name}"
  );

  // Decode with dav1d
  assert!(
    run_dav1d_decode(&output_path, &dec_path),
    "Decoding failed for test {test_name}"
  );

  // Compare reconstruction with decoded output
  match compare_y4m_files(&rec_path, &dec_path) {
    Ok(()) => {
      // Clean up on success
      let _ = fs::remove_dir_all(&temp_dir);
    }
    Err(e) => {
      panic!(
        "Comparison failed for test {test_name}: {e}\nFiles preserved in {temp_dir:?}"
      );
    }
  }
}

// ============================================================================
// Test cases mirroring native encode_decode tests
// ============================================================================

mod speed_tests {
  use super::*;

  macro_rules! speed_test {
    ($name:ident, $speed:expr) => {
      #[test]
      fn $name() {
        // Test multiple dimension offsets like native tests
        for (dx, dy) in &[(0, 0), (4, 4), (8, 8), (16, 16)] {
          let w = 64 + dx;
          let h = 80 + dy;
          run_test(
            &format!("speed_{}_{}x{}", $speed, w, h),
            w,
            h,
            5,       // frames
            $speed,  // speed
            100,     // quantizer
            8,       // bit_depth
            "420",   // chroma_sampling
            15,      // min_keyint
            15,      // max_keyint
            true,    // low_latency
            0,       // tile_rows
            0,       // tile_cols
            false,   // still_picture
            0,       // seed
          );
        }
      }
    };
  }

  speed_test!(speed_10, 10);
  speed_test!(speed_9, 9);
  speed_test!(speed_8, 8);
  speed_test!(speed_7, 7);
  speed_test!(speed_6, 6);

  // Slower speeds are marked as ignored (like native tests)
  #[test]
  #[ignore]
  fn speed_5() {
    for (dx, dy) in &[(0, 0), (4, 4)] {
      run_test(
        &format!("speed_5_{}x{}", 64 + dx, 80 + dy),
        64 + dx,
        80 + dy,
        5,
        5,
        100,
        8,
        "420",
        15,
        15,
        true,
        0,
        0,
        false,
        0,
      );
    }
  }

  #[test]
  #[ignore]
  fn speed_4() {
    run_test("speed_4", 64, 80, 5, 4, 100, 8, "420", 15, 15, true, 0, 0, false, 0);
  }

  #[test]
  #[ignore]
  fn speed_3() {
    run_test("speed_3", 64, 80, 5, 3, 100, 8, "420", 15, 15, true, 0, 0, false, 0);
  }

  #[test]
  #[ignore]
  fn speed_2() {
    run_test("speed_2", 64, 80, 5, 2, 100, 8, "420", 15, 15, true, 0, 0, false, 0);
  }

  #[test]
  #[ignore]
  fn speed_1() {
    run_test("speed_1", 64, 80, 5, 1, 100, 8, "420", 15, 15, true, 0, 0, false, 0);
  }

  #[test]
  #[ignore]
  fn speed_0() {
    run_test("speed_0", 64, 80, 5, 0, 100, 8, "420", 15, 15, true, 0, 0, false, 0);
  }
}

mod dimension_tests {
  use super::*;

  macro_rules! dimension_test {
    ($name:ident, $w:expr, $h:expr) => {
      #[test]
      fn $name() {
        let still_picture = $w < 16 || $h < 16;
        run_test(
          stringify!($name),
          $w,
          $h,
          1,     // frames
          10,    // speed
          100,   // quantizer
          8,     // bit_depth
          "420", // chroma_sampling
          15,    // min_keyint
          15,    // max_keyint
          true,  // low_latency
          0,     // tile_rows
          0,     // tile_cols
          still_picture,
          0, // seed
        );
      }
    };
  }

  // Small dimensions
  dimension_test!(dimension_256x256, 256, 256);
  dimension_test!(dimension_258x258, 258, 258);
  dimension_test!(dimension_260x260, 260, 260);
  dimension_test!(dimension_262x262, 262, 262);
  dimension_test!(dimension_264x264, 264, 264);
  dimension_test!(dimension_265x265, 265, 265);

  // Tiny dimensions
  dimension_test!(dimension_16x16, 16, 16);
  dimension_test!(dimension_32x32, 32, 32);
  dimension_test!(dimension_64x64, 64, 64);
  dimension_test!(dimension_128x128, 128, 128);

  // Large dimensions (ignored by default)
  #[test]
  #[ignore]
  fn dimension_512x512() {
    run_test(
      "dimension_512x512",
      512,
      512,
      1,
      10,
      100,
      8,
      "420",
      15,
      15,
      true,
      0,
      0,
      false,
      0,
    );
  }

  #[test]
  #[ignore]
  fn dimension_1024x1024() {
    run_test(
      "dimension_1024x1024",
      1024,
      1024,
      1,
      10,
      100,
      8,
      "420",
      15,
      15,
      true,
      0,
      0,
      false,
      0,
    );
  }
}

mod quantizer_tests {
  use super::*;

  macro_rules! quantizer_test {
    ($name:ident, $q:expr) => {
      #[test]
      fn $name() {
        for (dx, dy) in &[(0, 0), (4, 4), (8, 8), (16, 16)] {
          run_test(
            &format!("quantizer_{}_{}x{}", $q, 64 + dx, 80 + dy),
            64 + dx,
            80 + dy,
            5,     // frames
            10,    // speed
            $q,    // quantizer
            8,     // bit_depth
            "420", // chroma_sampling
            15,    // min_keyint
            15,    // max_keyint
            true,  // low_latency
            0,     // tile_rows
            0,     // tile_cols
            false, // still_picture
            0,     // seed
          );
        }
      }
    };
  }

  quantizer_test!(quantizer_60, 60);
  quantizer_test!(quantizer_80, 80);
  quantizer_test!(quantizer_100, 100);
  quantizer_test!(quantizer_120, 120);
}

mod keyframe_tests {
  use super::*;

  #[test]
  fn keyframes() {
    run_test(
      "keyframes",
      64,
      80,
      12,    // frames
      9,     // speed
      100,   // quantizer
      8,     // bit_depth
      "420", // chroma_sampling
      6,     // min_keyint
      6,     // max_keyint
      true,  // low_latency
      0,     // tile_rows
      0,     // tile_cols
      false, // still_picture
      0,     // seed
    );
  }

  #[test]
  fn reordering() {
    for keyint in &[4, 5, 6] {
      run_test(
        &format!("reordering_keyint_{keyint}"),
        64,
        80,
        12,       // frames
        10,       // speed
        100,      // quantizer
        8,        // bit_depth
        "420",    // chroma_sampling
        *keyint,  // min_keyint
        *keyint,  // max_keyint
        false,    // low_latency (reordering enabled)
        0,        // tile_rows
        0,        // tile_cols
        false,    // still_picture
        0,        // seed
      );
    }
  }

  #[test]
  fn reordering_short_video() {
    run_test(
      "reordering_short",
      64,
      80,
      2,     // frames
      10,    // speed
      100,   // quantizer
      8,     // bit_depth
      "420", // chroma_sampling
      12,    // min_keyint
      12,    // max_keyint
      false, // low_latency (reordering enabled)
      0,     // tile_rows
      0,     // tile_cols
      false, // still_picture
      0,     // seed
    );
  }
}

mod bit_depth_tests {
  use super::*;

  #[test]
  #[ignore]
  fn high_bit_depth_10() {
    run_test(
      "hbd_10",
      64,
      80,
      3,     // frames
      10,    // speed (using faster speed for HBD)
      100,   // quantizer
      10,    // bit_depth
      "420", // chroma_sampling
      15,    // min_keyint
      15,    // max_keyint
      true,  // low_latency
      0,     // tile_rows
      0,     // tile_cols
      false, // still_picture
      0,     // seed
    );
  }

  #[test]
  #[ignore]
  fn high_bit_depth_12() {
    run_test(
      "hbd_12",
      64,
      80,
      3,     // frames
      10,    // speed
      100,   // quantizer
      12,    // bit_depth
      "420", // chroma_sampling
      15,    // min_keyint
      15,    // max_keyint
      true,  // low_latency
      0,     // tile_rows
      0,     // tile_cols
      false, // still_picture
      0,     // seed
    );
  }
}

mod chroma_sampling_tests {
  use super::*;

  #[test]
  #[ignore]
  fn chroma_sampling_420() {
    run_test(
      "cs_420",
      64,
      80,
      3,     // frames
      10,    // speed
      100,   // quantizer
      8,     // bit_depth
      "420", // chroma_sampling
      15,    // min_keyint
      15,    // max_keyint
      true,  // low_latency
      0,     // tile_rows
      0,     // tile_cols
      false, // still_picture
      0,     // seed
    );
  }

  #[test]
  #[ignore]
  fn chroma_sampling_422() {
    run_test(
      "cs_422",
      64,
      80,
      3,     // frames
      10,    // speed
      100,   // quantizer
      8,     // bit_depth
      "422", // chroma_sampling
      15,    // min_keyint
      15,    // max_keyint
      true,  // low_latency
      0,     // tile_rows
      0,     // tile_cols
      false, // still_picture
      0,     // seed
    );
  }

  #[test]
  #[ignore]
  fn chroma_sampling_444() {
    run_test(
      "cs_444",
      64,
      80,
      3,     // frames
      10,    // speed
      100,   // quantizer
      8,     // bit_depth
      "444", // chroma_sampling
      15,    // min_keyint
      15,    // max_keyint
      true,  // low_latency
      0,     // tile_rows
      0,     // tile_cols
      false, // still_picture
      0,     // seed
    );
  }
}

mod tile_tests {
  use super::*;

  #[test]
  fn tile_encoding() {
    run_test(
      "tiles_2x2",
      256,
      140,   // height chosen to test stretched restoration units
      5,     // frames
      10,    // speed
      100,   // quantizer
      8,     // bit_depth
      "420", // chroma_sampling
      15,    // min_keyint
      15,    // max_keyint
      true,  // low_latency
      2,     // tile_rows
      2,     // tile_cols
      false, // still_picture
      0,     // seed
    );
  }
}

mod still_picture_tests {
  use super::*;

  #[test]
  fn still_picture_mode() {
    run_test(
      "still_picture",
      480,
      304,
      1,     // frames
      6,     // speed
      100,   // quantizer
      8,     // bit_depth
      "420", // chroma_sampling
      0,     // min_keyint
      0,     // max_keyint
      false, // low_latency
      0,     // tile_rows
      0,     // tile_cols
      true,  // still_picture
      0,     // seed
    );
  }
}

// ============================================================================
// Binary/CLI tests mirroring tests/binary.rs
// These test the CLI interface itself (rate control, multi-pass, etc.)
// ============================================================================

mod binary_tests {
  use super::*;

  /// Get path to the small test input file
  fn get_small_input_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/small_input.y4m")
  }

  /// Copy the test input to a work directory and return the local path
  fn setup_input(work_dir: &Path) -> PathBuf {
    let src = get_small_input_path();
    let dest = work_dir.join("input.y4m");
    fs::copy(&src, &dest).expect("Failed to copy test input");
    dest
  }

  /// Run a wasmtime command with the given arguments
  /// Returns (success, stdout, stderr)
  fn run_wasm_cli(
    wasm_path: &Path,
    work_dir: &Path,
    args: &[&str],
  ) -> (bool, String, String) {
    let mut cmd = Command::new("wasmtime");
    cmd
      .arg("run")
      .arg("-S")
      .arg("cli")
      .arg(format!("--dir={}", work_dir.to_string_lossy()));

    cmd.arg(wasm_path);
    for arg in args {
      cmd.arg(arg);
    }

    let output = cmd.output().expect("Failed to run wasmtime");

    (
      output.status.success(),
      String::from_utf8_lossy(&output.stdout).to_string(),
      String::from_utf8_lossy(&output.stderr).to_string(),
    )
  }

  /// Run an encode with flexible options
  /// input_name and output_name are relative to work_dir (will be converted to absolute paths)
  fn run_encode(
    test_name: &str, work_dir: &Path, input_name: &str, output_name: &str,
    high_bitdepth: bool, extra_args: &[&str],
  ) -> bool {
    let wasm_path = get_wasm_path();
    if !wasm_path.exists() {
      panic!(
        "rav1e.wasm not found at {:?}. Build with: cargo build --target wasm32-wasip2 --release --no-default-features --features binaries",
        wasm_path
      );
    }

    // Convert relative names to absolute paths within work_dir
    let input_path = work_dir.join(input_name);
    let output_path = work_dir.join(output_name);
    let input_str = input_path.to_string_lossy();
    let output_str = output_path.to_string_lossy();

    // Always include -y (overwrite) since WASI can't handle interactive prompts
    let mut args: Vec<&str> = vec![&input_str, "--output", &output_str, "-y"];

    if high_bitdepth {
      args.push("--high-bitdepth");
    }

    // Handle extra_args - convert relative paths for pass files
    let extra_args_expanded: Vec<String> = extra_args
      .iter()
      .map(|&arg| {
        // If it looks like a relative path (contains . but not flag), expand it
        if !arg.starts_with('-') && (arg.ends_with(".dat") || arg.ends_with(".ivf") || arg.ends_with(".y4m")) {
          work_dir.join(arg).to_string_lossy().to_string()
        } else {
          arg.to_string()
        }
      })
      .collect();
    let extra_args_refs: Vec<&str> = extra_args_expanded.iter().map(|s| s.as_str()).collect();
    args.extend(&extra_args_refs);

    let (success, _stdout, stderr) = run_wasm_cli(&wasm_path, work_dir, &args);

    if !success {
      eprintln!("Encode failed for {test_name}: {stderr}");
    }

    success
  }

  // -------------------------------------------------------------------------
  // One-pass QP-based encoding tests
  // -------------------------------------------------------------------------

  #[test]
  fn one_pass_qp_based_low_bitdepth() {
    let temp_dir = create_temp_dir("binary_1pass_qp_low");
    let _input = setup_input(&temp_dir);

    assert!(
      run_encode(
        "one_pass_qp_based_low_bitdepth",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        false, // low bitdepth
        &["--quantizer", "100"],
      ),
      "One-pass QP-based encoding (low bitdepth) failed"
    );

    assert!(temp_dir.join("output.ivf").exists(), "Output file was not created");
    let _ = fs::remove_dir_all(&temp_dir);
  }

  #[test]
  fn one_pass_qp_based_high_bitdepth() {
    let temp_dir = create_temp_dir("binary_1pass_qp_high");
    let _input = setup_input(&temp_dir);

    assert!(
      run_encode(
        "one_pass_qp_based_high_bitdepth",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        true, // high bitdepth
        &["--quantizer", "100"],
      ),
      "One-pass QP-based encoding (high bitdepth) failed"
    );

    assert!(temp_dir.join("output.ivf").exists(), "Output file was not created");
    let _ = fs::remove_dir_all(&temp_dir);
  }

  // -------------------------------------------------------------------------
  // One-pass bitrate-based encoding tests
  // -------------------------------------------------------------------------

  #[test]
  fn one_pass_bitrate_based_low_bitdepth() {
    let temp_dir = create_temp_dir("binary_1pass_br_low");
    let _input = setup_input(&temp_dir);

    assert!(
      run_encode(
        "one_pass_bitrate_based_low_bitdepth",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        false,
        &["--bitrate", "1000"],
      ),
      "One-pass bitrate-based encoding (low bitdepth) failed"
    );

    assert!(temp_dir.join("output.ivf").exists(), "Output file was not created");
    let _ = fs::remove_dir_all(&temp_dir);
  }

  #[test]
  fn one_pass_bitrate_based_high_bitdepth() {
    let temp_dir = create_temp_dir("binary_1pass_br_high");
    let _input = setup_input(&temp_dir);

    assert!(
      run_encode(
        "one_pass_bitrate_based_high_bitdepth",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        true,
        &["--bitrate", "1000"],
      ),
      "One-pass bitrate-based encoding (high bitdepth) failed"
    );

    assert!(temp_dir.join("output.ivf").exists(), "Output file was not created");
    let _ = fs::remove_dir_all(&temp_dir);
  }

  // -------------------------------------------------------------------------
  // Two-pass bitrate-based encoding tests
  // -------------------------------------------------------------------------

  #[test]
  fn two_pass_bitrate_based_low_bitdepth() {
    let temp_dir = create_temp_dir("binary_2pass_low");
    let _input = setup_input(&temp_dir);

    // First pass
    assert!(
      run_encode(
        "two_pass_bitrate_based_low_bitdepth_pass1",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        false,
        &["--bitrate", "1000", "--first-pass", "pass.dat"],
      ),
      "Two-pass first pass (low bitdepth) failed"
    );

    assert!(temp_dir.join("pass.dat").exists(), "Pass file was not created");

    // Second pass
    assert!(
      run_encode(
        "two_pass_bitrate_based_low_bitdepth_pass2",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        false,
        &["--bitrate", "1000", "--second-pass", "pass.dat"],
      ),
      "Two-pass second pass (low bitdepth) failed"
    );

    assert!(temp_dir.join("output.ivf").exists(), "Output file was not created");
    let _ = fs::remove_dir_all(&temp_dir);
  }

  #[test]
  fn two_pass_bitrate_based_high_bitdepth() {
    let temp_dir = create_temp_dir("binary_2pass_high");
    let _input = setup_input(&temp_dir);

    // First pass
    assert!(
      run_encode(
        "two_pass_bitrate_based_high_bitdepth_pass1",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        true,
        &["--bitrate", "1000", "--first-pass", "pass.dat"],
      ),
      "Two-pass first pass (high bitdepth) failed"
    );

    // Second pass
    assert!(
      run_encode(
        "two_pass_bitrate_based_high_bitdepth_pass2",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        true,
        &["--bitrate", "1000", "--second-pass", "pass.dat"],
      ),
      "Two-pass second pass (high bitdepth) failed"
    );

    let _ = fs::remove_dir_all(&temp_dir);
  }

  // -------------------------------------------------------------------------
  // Two-pass bitrate-based constrained encoding tests
  // -------------------------------------------------------------------------

  #[test]
  fn two_pass_bitrate_based_constrained_low_bitdepth() {
    let temp_dir = create_temp_dir("binary_2pass_cons_low");
    let _input = setup_input(&temp_dir);

    // First pass
    assert!(
      run_encode(
        "two_pass_constrained_low_pass1",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        false,
        &[
          "--bitrate",
          "1000",
          "--reservoir-frame-delay",
          "14",
          "--first-pass",
          "pass.dat"
        ],
      ),
      "Two-pass constrained first pass (low bitdepth) failed"
    );

    // Second pass
    assert!(
      run_encode(
        "two_pass_constrained_low_pass2",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        false,
        &[
          "--bitrate",
          "1000",
          "--reservoir-frame-delay",
          "14",
          "--second-pass",
          "pass.dat"
        ],
      ),
      "Two-pass constrained second pass (low bitdepth) failed"
    );

    let _ = fs::remove_dir_all(&temp_dir);
  }

  #[test]
  fn two_pass_bitrate_based_constrained_high_bitdepth() {
    let temp_dir = create_temp_dir("binary_2pass_cons_high");
    let _input = setup_input(&temp_dir);

    // First pass
    assert!(
      run_encode(
        "two_pass_constrained_high_pass1",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        true,
        &[
          "--bitrate",
          "1000",
          "--reservoir-frame-delay",
          "14",
          "--first-pass",
          "pass.dat"
        ],
      ),
      "Two-pass constrained first pass (high bitdepth) failed"
    );

    // Second pass
    assert!(
      run_encode(
        "two_pass_constrained_high_pass2",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        true,
        &[
          "--bitrate",
          "1000",
          "--reservoir-frame-delay",
          "14",
          "--second-pass",
          "pass.dat"
        ],
      ),
      "Two-pass constrained second pass (high bitdepth) failed"
    );

    let _ = fs::remove_dir_all(&temp_dir);
  }

  // -------------------------------------------------------------------------
  // Three-pass bitrate-based encoding tests
  // -------------------------------------------------------------------------

  #[test]
  fn three_pass_bitrate_based_low_bitdepth() {
    let temp_dir = create_temp_dir("binary_3pass_low");
    let _input = setup_input(&temp_dir);

    // First pass
    assert!(
      run_encode(
        "three_pass_low_pass1",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        false,
        &["--bitrate", "1000", "--first-pass", "pass1.dat"],
      ),
      "Three-pass first pass (low bitdepth) failed"
    );

    // Second pass (read pass1, write pass2)
    assert!(
      run_encode(
        "three_pass_low_pass2",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        false,
        &[
          "--bitrate",
          "1000",
          "--second-pass",
          "pass1.dat",
          "--first-pass",
          "pass2.dat"
        ],
      ),
      "Three-pass second pass (low bitdepth) failed"
    );

    // Third pass (read pass2)
    assert!(
      run_encode(
        "three_pass_low_pass3",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        false,
        &["--bitrate", "1000", "--second-pass", "pass2.dat"],
      ),
      "Three-pass third pass (low bitdepth) failed"
    );

    let _ = fs::remove_dir_all(&temp_dir);
  }

  #[test]
  fn three_pass_bitrate_based_high_bitdepth() {
    let temp_dir = create_temp_dir("binary_3pass_high");
    let _input = setup_input(&temp_dir);

    // First pass
    assert!(
      run_encode(
        "three_pass_high_pass1",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        true,
        &["--bitrate", "1000", "--first-pass", "pass1.dat"],
      ),
      "Three-pass first pass (high bitdepth) failed"
    );

    // Second pass (read pass1, write pass2)
    assert!(
      run_encode(
        "three_pass_high_pass2",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        true,
        &[
          "--bitrate",
          "1000",
          "--second-pass",
          "pass1.dat",
          "--first-pass",
          "pass2.dat"
        ],
      ),
      "Three-pass second pass (high bitdepth) failed"
    );

    // Third pass (read pass2)
    assert!(
      run_encode(
        "three_pass_high_pass3",
        &temp_dir,
        "input.y4m",
        "output.ivf",
        true,
        &["--bitrate", "1000", "--second-pass", "pass2.dat"],
      ),
      "Three-pass third pass (high bitdepth) failed"
    );

    let _ = fs::remove_dir_all(&temp_dir);
  }
}
