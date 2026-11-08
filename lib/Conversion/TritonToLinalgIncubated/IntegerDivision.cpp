// Copyright (c) 2026 FlagTree contributors.
// SPDX-License-Identifier: MIT

#include "incubated/Conversion/TritonToLinalgIncubated/TritonOpConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <cstdint>
#include <optional>
#include <type_traits>

using namespace mlir;

namespace TTOpConverters {
namespace {
std::optional<APInt> splatInteger(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto scalar = dyn_cast<IntegerAttr>(constant.getValue()))
      return scalar.getValue();
    if (auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue());
        dense && dense.isSplat())
      return dense.getSplatValue<APInt>();
  }
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    return splatInteger(splat.getSrc());
  if (auto fill = value.getDefiningOp<linalg::FillOp>())
    return splatInteger(fill.getInputs()[0]);
  return std::nullopt;
}

// Use only 32-bit integer vector operations. In particular, neither the
// product's high half nor the correction may pass through an FP32 cast.
class ConstantQuotient {
  ConversionPatternRewriter &rewriter;
  Location loc;
  RankedTensorType type;

public:
  ConstantQuotient(ConversionPatternRewriter &rewriter, Location loc,
                   RankedTensorType type)
      : rewriter(rewriter), loc(loc), type(type) {}
  Value constant(uint32_t bits) {
    return rewriter.create<arith::ConstantOp>(
        loc, DenseIntElementsAttr::get(type, APInt(32, bits)));
  }
  template <typename Op> Value binary(Value a, Value b) {
    return rewriter.create<Op>(loc, a, b);
  }
  Value shift(Value a, unsigned n) {
    return n ? binary<arith::ShRUIOp>(a, constant(n)) : a;
  }
  Value highProduct(Value a, uint32_t multiplier) {
    Value low = binary<arith::AndIOp>(a, constant(0xffff));
    Value high = shift(a, 16);
    Value loMultiplier = constant(multiplier & 0xffff);
    Value hiMultiplier = constant(multiplier >> 16);
    Value ll = binary<arith::MulIOp>(low, loMultiplier);
    Value hl = binary<arith::MulIOp>(high, loMultiplier);
    Value lh = binary<arith::MulIOp>(low, hiMultiplier);
    Value hh = binary<arith::MulIOp>(high, hiMultiplier);
    Value carry = binary<arith::AddIOp>(
        shift(ll, 16), binary<arith::AndIOp>(hl, constant(0xffff)));
    carry = binary<arith::AddIOp>(carry,
                                  binary<arith::AndIOp>(lh, constant(0xffff)));
    Value result = binary<arith::AddIOp>(hh, shift(hl, 16));
    result = binary<arith::AddIOp>(result, shift(lh, 16));
    return binary<arith::AddIOp>(result, shift(carry, 16));
  }
  Value unsignedQuotient(Value n, uint32_t divisor) {
    if (divisor == 1)
      return n;
    APInt d(32, divisor);
    if (d.isPowerOf2())
      return shift(n, d.logBase2());
    // m=floor(2^32/d): q0=high(n*m) is q or q-1 for every uint32 n.
    // Thus r=n-q0*d never overflows, and one exact correction suffices.
    uint32_t multiplier =
        (APInt(64, 1).shl(32).udiv(d.zext(64))).getZExtValue();
    Value q = highProduct(n, multiplier);
    Value denominator = constant(divisor);
    Value r = binary<arith::SubIOp>(n, binary<arith::MulIOp>(q, denominator));
    Value difference = binary<arith::SubIOp>(r, denominator);
    Value notR = binary<arith::XOrIOp>(r, constant(0xffffffff));
    Value notXor = binary<arith::XOrIOp>(binary<arith::XOrIOp>(r, denominator),
                                         constant(0xffffffff));
    Value borrow =
        shift(binary<arith::OrIOp>(binary<arith::AndIOp>(notR, denominator),
                                   binary<arith::AndIOp>(notXor, difference)),
              31);
    Value correction = binary<arith::XOrIOp>(borrow, constant(1));
    return binary<arith::AddIOp>(q, correction);
  }
  Value quotient(Value n, APInt divisor, bool isSigned) {
    if (!isSigned)
      return unsignedQuotient(n, divisor.getZExtValue());
    Value sign = binary<arith::ShRSIOp>(n, constant(31));
    Value magnitude =
        binary<arith::SubIOp>(binary<arith::XOrIOp>(n, sign), sign);
    Value q = unsignedQuotient(magnitude, divisor.abs().getZExtValue());
    Value resultSign = divisor.isNegative()
                           ? binary<arith::XOrIOp>(sign, constant(0xffffffff))
                           : sign;
    return binary<arith::SubIOp>(binary<arith::XOrIOp>(q, resultSign),
                                 resultSign);
  }
};

template <typename Op> struct ExactIntegerDivision : OpConversionPattern<Op> {
  ExactIntegerDivision(MLIRContext *context)
      : OpConversionPattern<Op>(context, PatternBenefit(10)) {}
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!needsExactIntegerDivision(op))
      return failure();
    constexpr bool isSigned = std::is_same_v<Op, arith::DivSIOp> ||
                              std::is_same_v<Op, arith::RemSIOp>;
    constexpr bool isRemainder = std::is_same_v<Op, arith::RemSIOp> ||
                                 std::is_same_v<Op, arith::RemUIOp>;
    auto type = cast<RankedTensorType>(op.getType());
    auto loc = op.getLoc();
    auto divisor = splatInteger(op.getRhs());
    Value lhs = adaptor.getLhs(), rhs = adaptor.getRhs(), result;
    auto extend = [&](Value value, RankedTensorType wideType) -> Value {
      if (value.getType() == wideType)
        return value;
      if constexpr (isSigned)
        return rewriter.create<arith::ExtSIOp>(loc, wideType, value);
      return rewriter.create<arith::ExtUIOp>(loc, wideType, value);
    };
    if (divisor && !divisor->isZero()) {
      auto i32Type = type.clone(rewriter.getI32Type());
      Value n = extend(lhs, i32Type);
      APInt d = isSigned ? divisor->sextOrTrunc(32) : divisor->zextOrTrunc(32);
      ConstantQuotient arithmetic(rewriter, loc, i32Type);
      result = arithmetic.quotient(n, d, isSigned);
      if constexpr (isRemainder)
        result = arithmetic.binary<arith::SubIOp>(
            n, arithmetic.binary<arith::MulIOp>(
                   result, arithmetic.constant(d.getZExtValue())));
    } else {
      // CANN's tensor i32 division normalizes through FP32. Its signed i64
      // path is exact. Build uint32 magnitudes from positive 16-bit limbs:
      // this SDK also mis-lowers a direct tensor extui i32 -> i64 as extsi.
      auto wideType = type.clone(rewriter.getI64Type());
      auto exactExtend = [&](Value value) -> Value {
        if constexpr (isSigned)
          return extend(value, wideType);
        ConstantQuotient arithmetic(rewriter, loc, type);
        Value low = arithmetic.binary<arith::AndIOp>(
            value, arithmetic.constant(0xffff));
        Value high = arithmetic.shift(value, 16);
        low = rewriter.create<arith::ExtSIOp>(loc, wideType, low);
        high = rewriter.create<arith::ExtSIOp>(loc, wideType, high);
        Value shift = rewriter.create<arith::ConstantOp>(
            loc, DenseIntElementsAttr::get(wideType, APInt(64, 16)));
        high = rewriter.create<arith::ShLIOp>(loc, high, shift);
        return rewriter.create<arith::AddIOp>(loc, high, low);
      };
      result = rewriter.create<arith::DivSIOp>(loc, exactExtend(lhs),
                                               exactExtend(rhs));
    }
    if (result.getType() != type)
      result = rewriter.create<arith::TruncIOp>(loc, type, result);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

bool needsExactIntegerDivision(Operation *op) {
  if (!isa<arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp>(op))
    return false;
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!type || type.getEncoding() || !type.hasStaticShape() ||
      !isa<IntegerType>(type.getElementType()) ||
      type.getElementTypeBitWidth() != 32)
    return false;
  auto divisor = splatInteger(op->getOperand(1));
  if (divisor && divisor->isZero())
    return false; // Preserve source undefined behavior for division by zero.
  // Dynamic rem already uses exact scalar integer arithmetic in this SDK.
  return !isa<arith::RemSIOp, arith::RemUIOp>(op) || divisor.has_value();
}

void populateExactIntegerDivisionPatterns(RewritePatternSet &patterns) {
  patterns.add<ExactIntegerDivision<arith::DivSIOp>,
               ExactIntegerDivision<arith::DivUIOp>,
               ExactIntegerDivision<arith::RemSIOp>,
               ExactIntegerDivision<arith::RemUIOp>>(patterns.getContext());
}
} // namespace TTOpConverters
