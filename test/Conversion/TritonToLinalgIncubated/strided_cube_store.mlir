// RUN: triton-shared-opt --triton-to-linalg='named-ops=true' --split-input-file %s | FileCheck %s
// REQUIRES: flir-incubated

// CHECK-LABEL: func.func @strided_cube_writeback(
// CHECK: flagtree.cube_to_vector
// CHECK: custom_flagtree_strided_store_float
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @strided_cube_writeback(%a: tensor<32x16xf32>, %b: tensor<16x16xf32>, %out: memref<32x16xf32, strided<[256, 16], offset: ?>>) {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x16xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x16xf32>) -> tensor<32x16xf32>
    %dot = linalg.matmul {input_precision = "ieee"} ins(%a, %b : tensor<32x16xf32>, tensor<16x16xf32>) outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
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
    %dot = linalg.matmul {input_precision = "ieee"} ins(%a, %b : tensor<32x16xf32>, tensor<16x16xf32>) outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
    bufferization.materialize_in_destination %dot in writable %out : (tensor<32x16xf32>, memref<32x16xf32, strided<[16, 1], offset: ?>>) -> ()
    return
  }
}

// -----
// CHECK-LABEL: func.func @dynamic_cube_writeback(
// CHECK: flagtree.cube_to_vector
// CHECK: custom_flagtree_strided_store_float
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @dynamic_cube_writeback(%a: tensor<32x16xf32>, %b: tensor<16x16xf32>, %out: memref<32x16xf32, strided<[?, ?], offset: ?>>) {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x16xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x16xf32>) -> tensor<32x16xf32>
    %dot = linalg.matmul {input_precision = "ieee"} ins(%a, %b : tensor<32x16xf32>, tensor<16x16xf32>) outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
    bufferization.materialize_in_destination %dot in writable %out : (tensor<32x16xf32>, memref<32x16xf32, strided<[?, ?], offset: ?>>) -> ()
    return
  }
}
