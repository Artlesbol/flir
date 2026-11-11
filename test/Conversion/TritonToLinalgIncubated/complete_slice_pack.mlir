// RUN: triton-shared-opt --triton-to-linalg-incubated="named-ops=true" --split-input-file %s | FileCheck %s
// REQUIRES: flir-incubated

// CHECK-LABEL: func.func @complete(
// CHECK: tensor.concat dim(0)
// CHECK-NOT: tensor.insert_slice
module attributes {hacc.target = #hacc.target<"Ascend910B4">, flagtree.multibuffer = true} {
  func.func @complete(%a: tensor<4xf32>, %b: tensor<4xf32>, %base: tensor<8xf32>) -> tensor<8xf32> {
    %x = tensor.insert_slice %a into %base[0] [4] [1] : tensor<4xf32> into tensor<8xf32>
    %y = tensor.insert_slice %b into %x[4] [4] [1] : tensor<4xf32> into tensor<8xf32>
    return %y : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func @reverse(
// CHECK: tensor.concat dim(0)
// CHECK-NOT: tensor.insert_slice
module attributes {hacc.target = #hacc.target<"Ascend910B4">, flagtree.multibuffer = true} {
  func.func @reverse(%a: tensor<4xf32>, %b: tensor<4xf32>, %base: tensor<8xf32>) -> tensor<8xf32> {
    %x = tensor.insert_slice %a into %base[4] [4] [1] : tensor<4xf32> into tensor<8xf32>
    %y = tensor.insert_slice %b into %x[0] [4] [1] : tensor<4xf32> into tensor<8xf32>
    return %y : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func @single_buffer(
// CHECK-NOT: tensor.concat
// CHECK: tensor.insert_slice
module attributes {hacc.target = #hacc.target<"Ascend910B4">, flagtree.multibuffer = false} {
  func.func @single_buffer(%a: tensor<4xf32>, %b: tensor<4xf32>, %base: tensor<8xf32>) -> tensor<8xf32> {
    %x = tensor.insert_slice %a into %base[0] [4] [1] : tensor<4xf32> into tensor<8xf32>
    %y = tensor.insert_slice %b into %x[4] [4] [1] : tensor<4xf32> into tensor<8xf32>
    return %y : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func @other_target(
// CHECK-NOT: tensor.concat
// CHECK: tensor.insert_slice
module attributes {hacc.target = #hacc.target<"Ascend310P">, flagtree.multibuffer = true} {
  func.func @other_target(%a: tensor<4xf32>, %b: tensor<4xf32>, %base: tensor<8xf32>) -> tensor<8xf32> {
    %x = tensor.insert_slice %a into %base[0] [4] [1] : tensor<4xf32> into tensor<8xf32>
    %y = tensor.insert_slice %b into %x[4] [4] [1] : tensor<4xf32> into tensor<8xf32>
    return %y : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func @overlap(
// CHECK-NOT: tensor.concat
// CHECK: tensor.insert_slice
module attributes {hacc.target = #hacc.target<"Ascend910B4">, flagtree.multibuffer = true} {
  func.func @overlap(%a: tensor<4xf32>, %b: tensor<4xf32>, %base: tensor<8xf32>) -> tensor<8xf32> {
    %x = tensor.insert_slice %a into %base[0] [4] [1] : tensor<4xf32> into tensor<8xf32>
    %y = tensor.insert_slice %b into %x[3] [4] [1] : tensor<4xf32> into tensor<8xf32>
    return %y : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func @gap(
// CHECK-NOT: tensor.concat
// CHECK: tensor.insert_slice
module attributes {hacc.target = #hacc.target<"Ascend910B4">, flagtree.multibuffer = true} {
  func.func @gap(%a: tensor<3xf32>, %b: tensor<3xf32>, %base: tensor<8xf32>) -> tensor<8xf32> {
    %x = tensor.insert_slice %a into %base[0] [3] [1] : tensor<3xf32> into tensor<8xf32>
    %y = tensor.insert_slice %b into %x[4] [3] [1] : tensor<3xf32> into tensor<8xf32>
    return %y : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func @dynamic_offset(
// CHECK-NOT: tensor.concat
// CHECK: tensor.insert_slice
module attributes {hacc.target = #hacc.target<"Ascend910B4">, flagtree.multibuffer = true} {
  func.func @dynamic_offset(%a: tensor<4xf32>, %b: tensor<4xf32>, %base: tensor<8xf32>, %off: index) -> tensor<8xf32> {
    %x = tensor.insert_slice %a into %base[0] [4] [1] : tensor<4xf32> into tensor<8xf32>
    %y = tensor.insert_slice %b into %x[%off] [4] [1] : tensor<4xf32> into tensor<8xf32>
    return %y : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func @intermediate_consumer(
// CHECK-NOT: tensor.concat
// CHECK: tensor.insert_slice
module attributes {hacc.target = #hacc.target<"Ascend910B4">, flagtree.multibuffer = true} {
  func.func @intermediate_consumer(%a: tensor<4xf32>, %b: tensor<4xf32>, %base: tensor<8xf32>, %out: memref<1xf32>, %idx: index) -> tensor<8xf32> {
    %x = tensor.insert_slice %a into %base[0] [4] [1] : tensor<4xf32> into tensor<8xf32>
    %y = tensor.insert_slice %b into %x[4] [4] [1] : tensor<4xf32> into tensor<8xf32>
    %old = tensor.extract %x[%idx] : tensor<8xf32>
    memref.store %old, %out[%idx] : memref<1xf32>
    return %y : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func @strided(
// CHECK-NOT: tensor.concat
// CHECK: tensor.insert_slice
module attributes {hacc.target = #hacc.target<"Ascend910B4">, flagtree.multibuffer = true} {
  func.func @strided(%a: tensor<4xf32>, %b: tensor<4xf32>, %base: tensor<8xf32>) -> tensor<8xf32> {
    %x = tensor.insert_slice %a into %base[0] [4] [2] : tensor<4xf32> into tensor<8xf32>
    %y = tensor.insert_slice %b into %x[1] [4] [2] : tensor<4xf32> into tensor<8xf32>
    return %y : tensor<8xf32>
  }
}
