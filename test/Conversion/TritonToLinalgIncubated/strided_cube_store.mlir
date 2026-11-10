// RUN: triton-shared-opt --triton-to-linalg-incubated='named-ops=true' --split-input-file %s | FileCheck %s
// RUN: triton-shared-opt --triton-to-linalg-incubated="named-ops=true compile-on-910-95=true" --split-input-file %s | FileCheck %s --check-prefix=LEGACY --implicit-check-not=flagtree.cube_to_vector
// LEGACY: module
// REQUIRES: flir-incubated

// CHECK-LABEL: func.func @strided_cube_writeback(
// CHECK: flagtree.cube_to_vector
// CHECK: custom_flagtree_strided_store_float
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @strided_cube_writeback(%a: tensor<32x16xf32>, %b: tensor<16x16xf32>, %out: memref<32x16xf32, strided<[256, 16], offset: ?>>) {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x16xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x16xf32>) -> tensor<32x16xf32>
    %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
    bufferization.materialize_in_destination %dot in writable %out : (tensor<32x16xf32>, memref<32x16xf32, strided<[256, 16], offset: ?>>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: func.func @dense_cube_writeback(
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: bufferization.materialize_in_destination
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @dense_cube_writeback(%a: tensor<32x16xf32>, %b: tensor<16x16xf32>, %out: memref<32x16xf32, strided<[16, 1], offset: ?>>) {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x16xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x16xf32>) -> tensor<32x16xf32>
    %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
    bufferization.materialize_in_destination %dot in writable %out : (tensor<32x16xf32>, memref<32x16xf32, strided<[16, 1], offset: ?>>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: func.func @dynamic_cube_writeback(
// CHECK-NOT: flagtree.cube_to_vector
// CHECK-NOT: custom_flagtree_strided_store_float
// CHECK: bufferization.materialize_in_destination
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @dynamic_cube_writeback(%a: tensor<32x16xf32>, %b: tensor<16x16xf32>, %out: memref<32x16xf32, strided<[?, ?], offset: ?>>) {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x16xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x16xf32>) -> tensor<32x16xf32>
    %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
    bufferization.materialize_in_destination %dot in writable %out : (tensor<32x16xf32>, memref<32x16xf32, strided<[?, ?], offset: ?>>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: func.func @negative_cube_writeback(
// CHECK-NOT: flagtree.cube_to_vector
// CHECK: bufferization.materialize_in_destination
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @negative_cube_writeback(%a: tensor<32x16xf32>, %b: tensor<16x16xf32>, %out: memref<32x16xf32, strided<[-256, 16], offset: ?>>) {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x16xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x16xf32>) -> tensor<32x16xf32>
    %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
    bufferization.materialize_in_destination %dot in writable %out : (tensor<32x16xf32>, memref<32x16xf32, strided<[-256, 16], offset: ?>>) -> ()
    return
  }
}


// -----
// CHECK-LABEL: func.func @vector_store_unchanged(
// CHECK-NOT: flagtree.cube_to_vector
// CHECK-NOT: custom_flagtree_strided_store
// CHECK: bufferization.materialize_in_destination
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @vector_store_unchanged(%v: tensor<32x16xf32>, %out: memref<32x16xf32, strided<[256, 16], offset: ?>>) {
    bufferization.materialize_in_destination %v in writable %out : (tensor<32x16xf32>, memref<32x16xf32, strided<[256, 16], offset: ?>>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: func.func @oversize_dma_pitch(
// CHECK-NOT: flagtree.cube_to_vector
// CHECK-NOT: custom_flagtree_strided_store
// CHECK: bufferization.materialize_in_destination
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @oversize_dma_pitch(%a: tensor<32x16xf32>, %b: tensor<16x16xf32>, %out: memref<32x16xf32, strided<[2147483648, 16], offset: ?>>) {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x16xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x16xf32>) -> tensor<32x16xf32>
    %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
    bufferization.materialize_in_destination %dot in writable %out : (tensor<32x16xf32>, memref<32x16xf32, strided<[2147483648, 16], offset: ?>>) -> ()
    return
  }
}


// -----
// CHECK-LABEL: func.func @control_flow_store_unchanged(
// CHECK-NOT: flagtree.cube_to_vector
// CHECK-NOT: custom_flagtree_strided_store
// CHECK: bufferization.materialize_in_destination
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @control_flow_store_unchanged(%cond: i1, %a: tensor<32x16xf32>, %b: tensor<16x16xf32>, %v: tensor<32x16xf32>, %out: memref<32x16xf32, strided<[256, 16], offset: ?>>) {
    %result = scf.if %cond -> tensor<32x16xf32> {
      %zero = arith.constant dense<0.0> : tensor<32x16xf32>
      %dot = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
      scf.yield %dot : tensor<32x16xf32>
    } else {
      scf.yield %v : tensor<32x16xf32>
    }
    bufferization.materialize_in_destination %result in writable %out : (tensor<32x16xf32>, memref<32x16xf32, strided<[256, 16], offset: ?>>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: func.func @vector_arithmetic_store_unchanged(
// CHECK-NOT: custom_flagtree_strided_store
// CHECK: bufferization.materialize_in_destination
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @vector_arithmetic_store_unchanged(%a: tensor<32x16xf32>, %b: tensor<16x16xf32>, %v: tensor<32x16xf32>, %out: memref<32x16xf32, strided<[256, 16], offset: ?>>) {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x16xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x16xf32>) -> tensor<32x16xf32>
    %dot = tt.dot %a, %b, %init, inputPrecision = ieee : tensor<32x16xf32> * tensor<16x16xf32> -> tensor<32x16xf32>
    %sum = arith.addf %dot, %v : tensor<32x16xf32>
    bufferization.materialize_in_destination %sum in writable %out : (tensor<32x16xf32>, memref<32x16xf32, strided<[256, 16], offset: ?>>) -> ()
    return
  }
}
