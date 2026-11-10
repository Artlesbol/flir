// RUN: triton-shared-opt --triton-to-linalg='named-ops=true' %s | FileCheck %s
// REQUIRES: flir-incubated
// CHECK-LABEL: func.func @join_permute_chain(
// CHECK: arith.bitcast {{.*}} : tensor<1024xf32> to tensor<1024xi32>
// CHECK: arith.bitcast {{.*}} : tensor<1024xf32> to tensor<1024xi32>
// CHECK: custom_flagtree_stride_gather_int32
// CHECK: arith.bitcast {{.*}} : tensor<2048xi32> to tensor<2048xf32>
// CHECK: custom_flagtree_stride_gather_float
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @join_permute_chain(%a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<2048xf32> {
    %ra = tt.reshape %a : tensor<1024xf32> -> tensor<512x2xf32>
    %rb = tt.reshape %b : tensor<1024xf32> -> tensor<512x2xf32>
    %j = tt.join %ra, %rb : tensor<512x2xf32> -> tensor<512x2x2xf32>
    %t = tt.trans %j {order = array<i32: 0, 2, 1>} : tensor<512x2x2xf32> -> tensor<512x2x2xf32>
    %o = tt.reshape %t : tensor<512x2x2xf32> -> tensor<2048xf32>
    return %o : tensor<2048xf32>
  }
}
