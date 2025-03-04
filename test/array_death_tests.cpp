/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

#include <cstdlib>

using af::array;
using af::constant;
using af::dim4;
using af::end;
using af::fft;
using af::info;
using af::randu;
using af::scan;
using af::seq;
using af::setDevice;
using af::sin;
using af::sort;
using af::span;
using af::abs;
using af::dtype_traits;
using af::cdouble;
using af::cfloat;

template<typename T>
class ArrayDeathTest : public ::testing::Test {};

template<typename T>
class ArrayDeathTestType : public ::testing::Test {};

void deathTest() {
    info();
    setDevice(0);

    array A = randu(5, 3, f32);

    array B = sin(A) + 1.5;

    B(seq(0, 2), 1) = B(seq(0, 2), 1) * -1;

    array C = fft(B);

    array c = C.row(end);

    dim4 dims(16, 4, 1, 1);
    array r = constant(2, dims);

    array S = scan(r, 0, AF_BINARY_MUL);

    float d[] = {1, 2, 3, 4, 5, 6};
    array D(2, 3, d, afHost);

    D.col(0) = D.col(end);

    array vals, inds;
    sort(vals, inds, A);

    _exit(0);
}

template<typename T, typename Id>
void arrayIndexProxyIndices() {
    array A1 = randu(100, (af_dtype)dtype_traits<T>::af_type);
    array idx1 = randu(50, u32) % 100;
          idx1 = idx1.as((af_dtype)dtype_traits<Id>::af_type);
    auto X1 = A1(idx1);
    array Y1 = X1;
    Y1.eval();
    af::sync();
    array Z1 = A1(idx1);
    Z1.eval();
    af::sync();

    array A2a = randu(100, 19, (af_dtype)dtype_traits<T>::af_type);
    array idx2a1 = randu(50, u32) % 100;
          idx2a1 = idx2a1.as((af_dtype)dtype_traits<Id>::af_type);
    array idx2a2 = randu(7, u32) % 19;
          idx2a2 = idx2a2.as((af_dtype)dtype_traits<Id>::af_type);
    auto X2a = A2a(idx2a1, idx2a2);
    array Y2a = X2a;
    Y2a.eval();
    af::sync();
    array Z2a = A2a(idx2a1, idx2a2);
    Z2a.eval();
    af::sync();

    array A2b = randu(100, 19, (af_dtype)dtype_traits<T>::af_type);
    array idx2b = randu(50, u32) % 100;
          idx2b = idx2b.as((af_dtype)dtype_traits<Id>::af_type);
    auto X2b = A2b(idx2b, span);
    array Y2b = X2b;
    Y2b.eval();
    af::sync();
    array Z2b = A2b(idx2b, span);
    Z2b.eval();
    af::sync();

    array A2c = randu(100, 19, (af_dtype)dtype_traits<T>::af_type);
    array idx2c = randu(7, u32) % 19;
          idx2c = idx2c.as((af_dtype)dtype_traits<Id>::af_type);
    auto X2c = A2c(seq(3, 75), idx2c);
    array Y2c = X2c;
    Y2c.eval();
    af::sync();
    array Z2c = A2c(seq(3, 75), idx2c);
    Z2c.eval();
    af::sync();

    array A3a = randu(10, 19, 8, (af_dtype)dtype_traits<T>::af_type);
    array idx3a1 = randu(5, u32) % 10;
          idx3a1 = idx3a1.as((af_dtype)dtype_traits<Id>::af_type);
    array idx3a2 = randu(9, u32) % 19;
          idx3a2 = idx3a2.as((af_dtype)dtype_traits<Id>::af_type);
    array idx3a3 = randu(4, u32) % 4;
          idx3a3 = idx3a3.as((af_dtype)dtype_traits<Id>::af_type);
    auto X3a = A3a(idx3a1, idx3a2, idx3a3);
    array Y3a = X3a;
    Y3a.eval();
    af::sync();
    array Z3a = A3a(idx3a1, idx3a2, idx3a3);
    Z3a.eval();
    af::sync();

    array A3b = randu(10, 19, 8, (af_dtype)dtype_traits<T>::af_type);
    array idx3b1 = randu(5, u32) % 10;
          idx3b1 = idx3b1.as((af_dtype)dtype_traits<Id>::af_type);
    array idx3b2 = randu(9, u32) % 19;
          idx3b2 = idx3b2.as((af_dtype)dtype_traits<Id>::af_type);
    auto X3b = A3b(idx3b1, idx3b2, span);
    array Y3b = X3b;
    Y3b.eval();
    af::sync();
    array Z3b = A3b(idx3b1, idx3b2, span);
    Z3b.eval();
    af::sync();

    array A3c = randu(10, 19, 8, (af_dtype)dtype_traits<T>::af_type);
    array idx3c1 = randu(5, u32) % 10;
          idx3c1 = idx3c1.as((af_dtype)dtype_traits<Id>::af_type);
    array idx3c2 = randu(4, u32) % 8;
          idx3c2 = idx3c2.as((af_dtype)dtype_traits<Id>::af_type);
    auto X3c = A3c(idx3c1, seq(0, 10), idx3c2);
    array Y3c = X3c;
    Y3c.eval();
    af::sync();
    array Z3c = A3c(idx3c1, seq(0, 10), idx3c2);
    Z3c.eval();
    af::sync();

    array A3d = randu(10, 19, 8, (af_dtype)dtype_traits<T>::af_type);
    array idx3d = randu(6, u32) % 8;
          idx3d = idx3d.as((af_dtype)dtype_traits<Id>::af_type);
    auto X3d = A3d(span, seq(7, 18), idx3d);
    array Y3d = X3d;
    Y3d.eval();
    af::sync();
    array Z3d = A3d(span, seq(7, 18), idx3d);
    Z3d.eval();
    af::sync();

    array A4a = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array idx4a1 = randu(5, u32) % 10;
          idx4a1 = idx4a1.as((af_dtype)dtype_traits<Id>::af_type);
    array idx4a2 = randu(9, u32) % 19;
          idx4a2 = idx4a2.as((af_dtype)dtype_traits<Id>::af_type);
    array idx4a3 = randu(4, u32) % 8;
          idx4a3 = idx4a3.as((af_dtype)dtype_traits<Id>::af_type);
    array idx4a4 = randu(2, u32) % 3;
          idx4a4 = idx4a4.as((af_dtype)dtype_traits<Id>::af_type);
    auto X4a = A4a(idx4a1, idx4a2, idx4a3, idx4a4);
    array Y4a = X4a;
    Y4a.eval();
    af::sync();
    array Z4a = A4a(idx4a1, idx4a2, idx4a3, idx4a4);
    Z4a.eval();
    af::sync();

    array A4b = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array idx4b1 = randu(9, u32) % 19;
          idx4b1 = idx4b1.as((af_dtype)dtype_traits<Id>::af_type);
    array idx4b2 = randu(4, u32) % 8;
          idx4b2 = idx4b2.as((af_dtype)dtype_traits<Id>::af_type);
    array idx4b3 = randu(2, u32) % 3;
          idx4b3 = idx4b3.as((af_dtype)dtype_traits<Id>::af_type);
    auto X4b = A4b(seq(3, 6), idx4b1, idx4b2, idx4b3);
    array Y4b = X4b;
    Y4b.eval();
    af::sync();
    array Z4b = A4b(seq(3, 6), idx4b1, idx4b2, idx4b3);
    Z4b.eval();
    af::sync();

    array A4c = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array idx4c1 = randu(5, u32) % 10;
          idx4c1 = idx4c1.as((af_dtype)dtype_traits<Id>::af_type);
    array idx4c2 = randu(9, u32) % 19;
          idx4c2 = idx4c2.as((af_dtype)dtype_traits<Id>::af_type);
    array idx4c3 = randu(4, u32) % 8;
          idx4c3 = idx4c3.as((af_dtype)dtype_traits<Id>::af_type);
    auto X4c = A4c(idx4c1, idx4c2, idx4c3, span);
    array Y4c = X4c;
    Y4c.eval();
    af::sync();
    array Z4c = A4c(idx4c1, idx4c2, idx4c3, span);
    Z4c.eval();
    af::sync();

    array A4d = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array idx4d1 = randu(5, u32) % 10;
          idx4d1 = idx4d1.as((af_dtype)dtype_traits<Id>::af_type);
    array idx4d2 = randu(2, u32) % 3;
          idx4d2 = idx4d2.as((af_dtype)dtype_traits<Id>::af_type);
    auto X4d = A4d(idx4d1, span, seq(1,1), idx4d2);
    array Y4d = X4d;
    Y4d.eval();
    af::sync();
    array Z4d = A4d(idx4d1, span, seq(1,1), idx4d2);
    Z4d.eval();
    af::sync();

    array A4e = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array idx4e = randu(5, u32) % 10;
          idx4e = idx4e.as((af_dtype)dtype_traits<Id>::af_type);
    auto X4e = A4e(idx4e, span, seq(1,1), seq(0,2));
    array Y4e = X4e;
    Y4e.eval();
    af::sync();
    array Z4e = A4e(idx4e, span, seq(1,1), seq(0,2));
    Z4e.eval();
    af::sync();
}

template<typename T>
void arrayIndexProxyMask() {
    array A1 = randu(100, (af_dtype)dtype_traits<T>::af_type);
    array cond1 = randu(100, b8);
    auto X1 = A1(cond1);
    array Y1 = X1;
    Y1.eval();
    af::sync();
    array Z1 = A1(cond1);
    Z1.eval();
    af::sync();

    array A2a = randu(100, 19, (af_dtype)dtype_traits<T>::af_type);
    array cond2a1 = randu(100, b8);
    array cond2a2 = randu(19, b8);
    auto X2a = A2a(cond2a1, cond2a2);
    array Y2a = X2a;
    Y2a.eval();
    af::sync();
    array Z2a = A2a(cond2a1, cond2a2);
    Z2a.eval();
    af::sync();

    array A2b = randu(100, 19, (af_dtype)dtype_traits<T>::af_type);
    array cond2b = randu(100, b8);
    auto X2b = A2b(cond2b, span);
    array Y2b = X2b;
    Y2b.eval();
    af::sync();
    array Z2b = A2b(cond2b, span);
    Z2b.eval();
    af::sync();

    array A2c = randu(100, 19, (af_dtype)dtype_traits<T>::af_type);
    array cond2c = randu(19, b8);
    auto X2c = A2c(seq(3, 75), cond2c);
    array Y2c = X2c;
    Y2c.eval();
    af::sync();
    array Z2c = A2c(seq(3, 75), cond2c);
    Z2c.eval();
    af::sync();

    array A3a = randu(10, 19, 8, (af_dtype)dtype_traits<T>::af_type);
    array cond3a1 = randu(10, b8);
    array cond3a2 = randu(19, b8);
    array cond3a3 = randu(8, b8);
    auto X3a = A3a(cond3a1, cond3a2, cond3a3);
    array Y3a = X3a;
    Y3a.eval();
    af::sync();
    array Z3a = A3a(cond3a1, cond3a2, cond3a3);
    Z3a.eval();
    af::sync();

    array A3b = randu(10, 19, 8, (af_dtype)dtype_traits<T>::af_type);
    array cond3b1 = randu(10, b8);
    array cond3b2 = randu(19, b8);
    auto X3b = A3b(cond3b1, cond3b2, span);
    array Y3b = X3b;
    Y3b.eval();
    af::sync();
    array Z3b = A3b(cond3b1, cond3b2, span);
    Z3b.eval();
    af::sync();

    array A3c = randu(10, 19, 8, (af_dtype)dtype_traits<T>::af_type);
    array cond3c1 = randu(10, b8);
    array cond3c2 = randu(8, b8);
    auto X3c = A3c(cond3c1, seq(0, 10), cond3c2);
    array Y3c = X3c;
    Y3c.eval();
    af::sync();
    array Z3c = A3c(cond3c1, seq(0, 10), cond3c2);
    Z3c.eval();
    af::sync();

    array A3d = randu(10, 19, 8, (af_dtype)dtype_traits<T>::af_type);
    array cond3d = randu(8, b8);
    auto X3d = A3d(span, seq(7, 18), cond3d);
    array Y3d = X3d;
    Y3d.eval();
    af::sync();
    array Z3d = A3d(span, seq(7, 18), cond3d);
    Z3d.eval();
    af::sync();

    array A4a = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array cond4a1 = randu(10, b8);
    array cond4a2 = randu(19, b8);
    array cond4a3 = randu(8, b8);
    array cond4a4 = randu(3, b8);
    auto X4a = A4a(cond4a1, cond4a2, cond4a3, cond4a4);
    array Y4a = X4a;
    Y4a.eval();
    af::sync();
    array Z4a = A4a(cond4a1, cond4a2, cond4a3, cond4a4);
    Z4a.eval();
    af::sync();

    array A4b = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array cond4b1 = randu(19, b8);
    array cond4b2 = randu(8, b8);
    array cond4b3 = randu(3, b8);
    auto X4b = A4b(seq(3, 6), cond4b1, cond4b2, cond4b3);
    array Y4b = X4b;
    Y4b.eval();
    af::sync();
    array Z4b = A4b(seq(3, 6), cond4b1, cond4b2, cond4b3);
    Z4b.eval();
    af::sync();

    array A4c = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array cond4c1 = randu(10, b8);
    array cond4c2 = randu(19, b8);
    array cond4c3 = randu(8, b8);
    auto X4c = A4c(cond4c1, cond4c2, cond4c3, span);
    array Y4c = X4c;
    Y4c.eval();
    af::sync();
    array Z4c = A4c(cond4c1, cond4c2, cond4c3, span);
    Z4c.eval();
    af::sync();

    array A4d = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array cond4d1 = randu(10, b8);
    array cond4d2 = randu(3, b8);
    auto X4d = A4d(cond4d1, span, seq(1,1), cond4d2);
    array Y4d = X4d;
    Y4d.eval();
    af::sync();
    array Z4d = A4d(cond4d1, span, seq(1,1), cond4d2);
    Z4d.eval();
    af::sync();

    array A4e = randu(10, 19, 8, 3, (af_dtype)dtype_traits<T>::af_type);
    array cond4e = randu(10, b8);
    auto X4e = A4e(cond4e, span, seq(1,1), seq(0,2));
    array Y4e = X4e;
    Y4e.eval();
    af::sync();
    array Z4e = A4e(cond4e, span, seq(1,1), seq(0,2));
    Z4e.eval();
    af::sync();
}

template<typename T>
void arrayIndexProxy() {
    int dev = af::getDevice();
    if(af::isHalfAvailable(dev)) arrayIndexProxyIndices<T, af_half>();
    if(af::isDoubleAvailable(dev)) arrayIndexProxyIndices<T, double>();
    arrayIndexProxyIndices<T, float>();
    arrayIndexProxyIndices<T, int>();
    arrayIndexProxyIndices<T, intl>();
    arrayIndexProxyIndices<T, short>();
    arrayIndexProxyIndices<T, uint>();
    arrayIndexProxyIndices<T, uintl>();
    arrayIndexProxyIndices<T, ushort>();
    arrayIndexProxyMask<T>();
}

TEST(ArrayDeathTest, ProxyMoveAssignmentOperator) {
    EXPECT_EXIT(deathTest(), ::testing::ExitedWithCode(0), "");
}

typedef ::testing::Types<af_half, cdouble, cfloat, double, float, int, intl, short, uint, uintl, ushort> TestTypes;
TYPED_TEST_SUITE(ArrayDeathTestType, TestTypes);

TYPED_TEST(ArrayDeathTestType, ArrayIndexProxyWorks_ISSUE_1693) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    EXPECT_NO_THROW(arrayIndexProxy<TypeParam>());
}
