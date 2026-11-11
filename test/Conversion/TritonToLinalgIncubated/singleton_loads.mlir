// RUN: triton-shared-opt --triton-to-structured-incubated --split-input-file %s | FileCheck %s
// REQUIRES: flir-incubated

// CHECK-LABEL: func.func @project(
// CHECK-COUNT-2: tt.load {{.*}} : !tt.ptr<f32>
// CHECK-NOT: tensor.extract_slice
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @project(%p: !tt.ptr<f32>, %a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>) {
 %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
 %ptrs = tt.splat %p : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
 %ptr = tt.addptr %ptrs, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
 %v = tt.load %ptr : tensor<32x!tt.ptr<f32>>
 %x = tensor.extract_slice %v[3] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %y = tensor.extract_slice %v[7] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %zero = arith.constant dense<0.0> : tensor<16x16xf32>
 %d = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
 return %x, %y, %d : tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @cache_policy(
// CHECK-COUNT-2: tt.load {{.*}} : !tt.ptr<f32>
// CHECK-NOT: tensor.extract_slice
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @cache_policy(%p: !tt.ptr<f32>, %a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>) {
 %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
 %ptrs = tt.splat %p : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
 %ptr = tt.addptr %ptrs, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
 %v = tt.load %ptr {cache = 1 : i32, evict = 1 : i32} : tensor<32x!tt.ptr<f32>>
 %x = tensor.extract_slice %v[3] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %y = tensor.extract_slice %v[7] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %zero = arith.constant dense<0.0> : tensor<16x16xf32>
 %d = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
 return %x, %y, %d : tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @volatile_read(
// CHECK: tensor.extract_slice
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @volatile_read(%p: !tt.ptr<f32>, %a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>) {
 %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
 %ptrs = tt.splat %p : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
 %ptr = tt.addptr %ptrs, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
 %v = tt.load %ptr {isVolatile = true} : tensor<32x!tt.ptr<f32>>
 %x = tensor.extract_slice %v[3] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %y = tensor.extract_slice %v[7] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %zero = arith.constant dense<0.0> : tensor<16x16xf32>
 %d = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
 return %x, %y, %d : tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @masked_read(
// CHECK: tensor.extract_slice
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @masked_read(%p: !tt.ptr<f32>, %a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %mask: tensor<32xi1>) -> (tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>) {
 %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
 %ptrs = tt.splat %p : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
 %ptr = tt.addptr %ptrs, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
 %v = tt.load %ptr, %mask : tensor<32x!tt.ptr<f32>>
 %x = tensor.extract_slice %v[3] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %y = tensor.extract_slice %v[7] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %zero = arith.constant dense<0.0> : tensor<16x16xf32>
 %d = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
 return %x, %y, %d : tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @pure_vector(
// CHECK: tensor.extract_slice
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @pure_vector(%p: !tt.ptr<f32>, %a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>) {
 %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
 %ptrs = tt.splat %p : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
 %ptr = tt.addptr %ptrs, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
 %v = tt.load %ptr : tensor<32x!tt.ptr<f32>>
 %x = tensor.extract_slice %v[3] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %y = tensor.extract_slice %v[7] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %zero = arith.constant dense<0.0> : tensor<16x16xf32>
 %d = arith.addf %a, %b : tensor<16x16xf32>
 return %x, %y, %d : tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @other_arch(
// CHECK: tensor.extract_slice
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910_95">} {
func.func @other_arch(%p: !tt.ptr<f32>, %a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>) {
 %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
 %ptrs = tt.splat %p : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
 %ptr = tt.addptr %ptrs, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
 %v = tt.load %ptr : tensor<32x!tt.ptr<f32>>
 %x = tensor.extract_slice %v[3] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %y = tensor.extract_slice %v[7] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %zero = arith.constant dense<0.0> : tensor<16x16xf32>
 %d = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
 return %x, %y, %d : tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @extra_vector_consumer(
// CHECK: tensor.extract_slice
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @extra_vector_consumer(%p: !tt.ptr<f32>, %a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>, tensor<32xf32>) {
 %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
 %ptrs = tt.splat %p : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
 %ptr = tt.addptr %ptrs, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
 %v = tt.load %ptr : tensor<32x!tt.ptr<f32>>
 %x = tensor.extract_slice %v[3] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %y = tensor.extract_slice %v[7] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %zero = arith.constant dense<0.0> : tensor<16x16xf32>
 %d = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
 return %x, %y, %d, %v : tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>, tensor<32xf32>
}
}

// -----

// CHECK-LABEL: func.func @dynamic_projection(
// CHECK: tensor.extract_slice
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @dynamic_projection(%p: !tt.ptr<f32>, %a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %offset: index) -> (tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>) {
 %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
 %ptrs = tt.splat %p : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
 %ptr = tt.addptr %ptrs, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
 %v = tt.load %ptr : tensor<32x!tt.ptr<f32>>
 %x = tensor.extract_slice %v[%offset] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %y = tensor.extract_slice %v[7] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %zero = arith.constant dense<0.0> : tensor<16x16xf32>
 %d = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
 return %x, %y, %d : tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @load_hints(
// CHECK: tensor.extract_slice
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @load_hints(%p: !tt.ptr<f32>, %a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>) {
 %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
 %ptrs = tt.splat %p : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
 %ptr = tt.addptr %ptrs, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
 %v = tt.load %ptr {flagtree_hints = "retain"} : tensor<32x!tt.ptr<f32>>
 %x = tensor.extract_slice %v[3] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %y = tensor.extract_slice %v[7] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %zero = arith.constant dense<0.0> : tensor<16x16xf32>
 %d = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
 return %x, %y, %d : tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>
}
}

// -----

// CHECK-LABEL: func.func @preserve_read_order(
// CHECK-COUNT-2: tt.load {{.*}} : !tt.ptr<f32>
// CHECK: tt.store
// CHECK-NOT: tensor.extract_slice
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @preserve_read_order(%p: !tt.ptr<f32>, %a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>) {
 %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
 %ptrs = tt.splat %p : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
 %ptr = tt.addptr %ptrs, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
 %v = tt.load %ptr : tensor<32x!tt.ptr<f32>>
 %c3 = arith.constant 3 : i32
 %p3 = tt.addptr %p, %c3 : !tt.ptr<f32>, i32
 %new = arith.constant 31.0 : f32
 tt.store %p3, %new : !tt.ptr<f32>
 %x = tensor.extract_slice %v[3] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %y = tensor.extract_slice %v[7] [1] [1] : tensor<32xf32> to tensor<1xf32>
 %zero = arith.constant dense<0.0> : tensor<16x16xf32>
 %d = tt.dot %a, %b, %zero, inputPrecision = ieee : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
 return %x, %y, %d : tensor<1xf32>, tensor<1xf32>, tensor<16x16xf32>
}
}
