// RUN: triton-shared-opt %s --triton-to-structured-incubated --triton-to-linalg-incubated='named-ops=true' --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @unit_views
// CHECK-NOT: DiscreteMemAccess
// CHECK-NOT: memref.load
// CHECK: memref.copy
// CHECK-NOT: DiscreteMemAccess
// CHECK-NOT: memref.load
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @unit_views(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32) {
    %q = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %u = tt.expand_dims %q {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %v = tt.expand_dims %u {axis = 2 : i32} : tensor<1x32xi32> -> tensor<1x32x1xi32>
    %off = tt.reshape %v : tensor<1x32x1xi32> -> tensor<32xi32>
    %ns = tt.splat %n : i32 -> tensor<32xi32>
    %mask = arith.cmpi slt, %q, %ns : tensor<32xi32>
    %m0 = tt.expand_dims %mask {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
    %mf = tt.reshape %m0 : tensor<1x32xi1> -> tensor<32xi1>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %xp = tt.addptr %xs, %off : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %zero = arith.constant dense<0.0> : tensor<32xf32>
    %value = tt.load %xp, %mf, %zero : tensor<32x!tt.ptr<f32>>
    %ys = tt.splat %y : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %yp = tt.addptr %ys, %q : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %yp, %value, %mask : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// A real dimension product is not a unit-only view and is not erased.
// CHECK-LABEL: func.func @real_product
// CHECK: tensor.reshape
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @real_product(%x: tensor<2x16xi32>) -> tensor<32xi32> {
    %v = tt.reshape %x : tensor<2x16xi32> -> tensor<32xi32>
    return %v : tensor<32xi32>
  }
}

// -----
// allow_reorder has a distinct element-mapping contract.
// CHECK-LABEL: func.func @reordered
// CHECK: tensor.reshape
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @reordered(%x: tensor<1x32xi32>) -> tensor<32xi32> {
    %v = tt.reshape %x allow_reorder : tensor<1x32xi32> -> tensor<32xi32>
    return %v : tensor<32xi32>
  }
}
