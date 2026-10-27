// RUN: triton-shared-opt --triton-to-linalg-incubated --split-input-file %s | FileCheck %s
// RUN: triton-shared-opt --triton-to-linalg-incubated=compile-on-910-95=true --split-input-file %s | FileCheck %s --check-prefix=LEGACY
// REQUIRES: flir-incubated

// LEGACY: module
// LEGACY-NOT: custom_flagtree_

// Exact inverses must fold before splitting into physical channel buffers.
// CHECK-LABEL: func.func @inverse(
// CHECK-NOT: hivm.hir.custom
// CHECK-NOT: tensor.extract_slice
// CHECK: return %arg0
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @inverse(%x: tensor<32x2xf32>) -> tensor<32x2xf32> {
    %a, %b = tt.split %x : tensor<32x2xf32> -> tensor<32xf32>
    %y = tt.join %a, %b : tensor<32xf32> -> tensor<32x2xf32>
    return %y : tensor<32x2xf32>
  }
}

// -----

// Independent inputs cannot be folded away.
// CHECK-LABEL: func.func @independent_join(
// CHECK: tensor.concat
// CHECK: symbol = "custom_flagtree_stride_gather_int32"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @independent_join(%a: tensor<32xf32>, %b: tensor<32xf32>) -> tensor<32x2xf32> {
    %y = tt.join %a, %b : tensor<32xf32> -> tensor<32x2xf32>
    return %y : tensor<32x2xf32>
  }
}

// -----

// CHECK-LABEL: func.func @static_transpose(
// CHECK: hivm.hir.varange
// CHECK: arith.andi
// CHECK: symbol = "custom_flagtree_stride_gather_half"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @static_transpose(%x: tensor<4x32xf16>) -> tensor<32x4xf16> {
    %y = tt.trans %x {order = array<i32: 1, 0>} : tensor<4x32xf16> -> tensor<32x4xf16>
    return %y : tensor<32x4xf16>
  }
}

// -----

// The static permutation's 16 KiB index budget must not capture large tiles.
// CHECK-LABEL: func.func @large_transpose(
// CHECK-NOT: hivm.hir.custom
// CHECK: linalg.transpose
// CHECK-NOT: hivm.hir.custom
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @large_transpose(%x: tensor<64x256xf32>) -> tensor<256x64xf32> {
    %y = tt.trans %x {order = array<i32: 1, 0>} : tensor<64x256xf32> -> tensor<256x64xf32>
    return %y : tensor<256x64xf32>
  }
}

// -----

// A computed/input value cannot assume the layout of a dense single-use load.
// CHECK-LABEL: func.func @general_split(
// CHECK: symbol = "custom_flagtree_split_float"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @general_split(%x: tensor<32x2xf32>) -> (tensor<32xf32>, tensor<32xf32>) {
    %a, %b = tt.split %x : tensor<32x2xf32> -> tensor<32xf32>
    return %a, %b : tensor<32xf32>, tensor<32xf32>
  }
}

// -----

// Narrow casts must be flat: old CANN's sub-block 2D casts mishandle row strides.
// CHECK-LABEL: func.func @narrow_split(
// CHECK: symbol = "custom_flagtree_byte_to_word"
// CHECK: symbol = "custom_flagtree_split_int16"
// CHECK: symbol = "custom_flagtree_word_to_byte"
// CHECK: symbol = "custom_flagtree_word_to_byte"
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @narrow_split(%x: tensor<8x8x2xi8>) -> (tensor<8x8xi8>, tensor<8x8xi8>) {
    %a, %b = tt.split %x : tensor<8x8x2xi8> -> tensor<8x8xi8>
    return %a, %b : tensor<8x8xi8>, tensor<8x8xi8>
  }
}
