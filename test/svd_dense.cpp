/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::iota;
using af::randu;
using af::seq;
using af::span;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class svd : public ::testing::Test {};

typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;
TYPED_TEST_SUITE(svd, TestTypes);

template<typename T>
inline double get_val(T val) {
    return val;
}

template<>
inline double get_val<cfloat>(cfloat val) {
    return abs(val);
}

template<>
double get_val<cdouble>(cdouble val) {
    return abs(val);
}

template<typename T>
void svdTest(const int M, const int N) {
    SUPPORTED_TYPE_CHECK(T);
    if (noLAPACKTests()) return;

    dtype ty = (dtype)dtype_traits<T>::af_type;

    array A = randu(M, N, ty);

    //! [ex_svd_reg]
    array U, S, Vt;
    af::svd(U, S, Vt, A);

    const int MN = std::min(M, N);

    array UU = U(span, seq(MN));
    array SS = diag(S, 0, false).as(ty);
    array VV = Vt(seq(MN), span);

    array AA = matmul(UU, SS, VV);
    //! [ex_svd_reg]

#if defined(OS_MAC)
    ASSERT_ARRAYS_NEAR(A, AA, 3E-3);
#else
    ASSERT_ARRAYS_NEAR(A, AA, 1E-3);
#endif
}

template<typename T>
void svdInPlaceTest(const int M, const int N) {
    SUPPORTED_TYPE_CHECK(T);
    if (noLAPACKTests()) return;

    dtype ty = (dtype)dtype_traits<T>::af_type;

    array A      = randu(M, N, ty);
    array A_copy = A.copy();

    array U, S, Vt;
    af::svdInPlace(U, S, Vt, A);

    const int MN = std::min(M, N);

    array UU = U(span, seq(MN));
    array SS = diag(S, 0, false).as(ty);
    array VV = Vt(seq(MN), span);

    array AA = matmul(UU, SS, VV);

#if defined(OS_MAC)
    ASSERT_ARRAYS_NEAR(A_copy, AA, 3E-3);
#else
    ASSERT_ARRAYS_NEAR(A_copy, AA, 1E-3);
#endif
}

template<typename T>
void checkInPlaceSameResults(const int M, const int N) {
    SUPPORTED_TYPE_CHECK(T);
    if (noLAPACKTests()) return;

    dtype ty = (dtype)dtype_traits<T>::af_type;

    array in = randu(dim4(M, N), ty);
    array u, s, v;
    af::svd(u, s, v, in);

    array uu, ss, vv;
    af::svdInPlace(uu, ss, vv, in);

    ASSERT_ARRAYS_EQ(u, uu);
    ASSERT_ARRAYS_EQ(s, ss);
    ASSERT_ARRAYS_EQ(v, vv);
}

TYPED_TEST(svd, Square) { svdTest<TypeParam>(500, 500); }

TYPED_TEST(svd, Rect0) { svdTest<TypeParam>(500, 300); }

TYPED_TEST(svd, Rect1) { svdTest<TypeParam>(300, 500); }

TYPED_TEST(svd, InPlaceSquare) { svdInPlaceTest<TypeParam>(500, 500); }

TYPED_TEST(svd, InPlaceRect0) { svdInPlaceTest<TypeParam>(500, 300); }

// dim0 < dim1 case not supported for now
// TYPED_TEST(svd, InPlaceRect1)
// {
//     svdInPlaceTest<TypeParam>(300, 500);
// }

TYPED_TEST(svd, InPlaceSameResultsSquare) {
    checkInPlaceSameResults<TypeParam>(10, 10);
}

TYPED_TEST(svd, InPlaceSameResultsRect0) {
    checkInPlaceSameResults<TypeParam>(10, 8);
}

// dim0 < dim1 case not supported for now
// TYPED_TEST(svd, InPlaceSameResultsRect1)
// {
//     checkInPlaceSameResults<TypeParam>(8, 10);
// }

TEST(svd, InPlaceRect0_Exception) {
    array in = randu(3, 5);
    array u, s, v;
    EXPECT_THROW(svdInPlace(u, s, v, in), af::exception);
}
