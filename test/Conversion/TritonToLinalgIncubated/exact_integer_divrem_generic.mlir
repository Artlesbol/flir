// RUN: triton-shared-opt %s --triton-to-linalg-incubated --split-input-file | FileCheck %s
// REQUIRES: flir-incubated

// CHECK-LABEL: func.func @generic_constant
// CHECK: linalg.generic
// CHECK-NOT: arith.divsi
// CHECK-NOT: arith.divui
// CHECK-NOT: arith.sitofp
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @generic_constant(%x: tensor<8xi32>) -> tensor<8xi32> {
    %d = arith.constant dense<15> : tensor<8xi32>
    %r = arith.remsi %x, %d : tensor<8xi32>
    return %r : tensor<8xi32>
  }
}

// -----
// CHECK-LABEL: func.func @generic_dynamic
// CHECK: arith.extsi {{.*}} : i32 to i64
// CHECK: arith.divsi {{.*}} : i64
// CHECK: arith.trunci {{.*}} : i64 to i32
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @generic_dynamic(%x: tensor<8xi32>, %d: tensor<8xi32>) -> tensor<8xi32> {
    %q = arith.divsi %x, %d : tensor<8xi32>
    return %q : tensor<8xi32>
  }
}
