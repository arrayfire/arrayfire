/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::abs;
using af::array;
using af::cfloat;
using af::cdouble;
using af::dtype;
using af::dtype_traits;
using af::randu;
using af::seq;
using af::span;

template<typename T>
class svd : public ::testing::Test
{
};

typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;
TYPED_TEST_CASE(svd, TestTypes);

template<typename T>
inline double get_val(T val)
{
    return val;
}

template<> inline double get_val<cfloat>(cfloat val)
{
    return abs(val);
}

template<> double get_val<cdouble>(cdouble val)
{
    return abs(val);
}

template<typename T>
void svdTest(const int M, const int N)
{

    if (noDoubleTests<T>()) return;
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

    vector<T> hA(M * N);
    vector<T> hAA(M * N);

    A.host(&hA[0]);
    AA.host(&hAA[0]);

    for (int i = 0; i < M * N; i++) {
#if defined(OS_MAC)
        ASSERT_NEAR(get_val(hA[i]), get_val(hAA[i]), 3E-3);
#else
        ASSERT_NEAR(get_val(hA[i]), get_val(hAA[i]), 1E-3);
#endif
    }
}

TYPED_TEST(svd, Square)
{
    svdTest<TypeParam>(500, 500);
}

TYPED_TEST(svd, Rect0)
{
    svdTest<TypeParam>(500, 300);
}

TYPED_TEST(svd, Rect1)
{
    svdTest<TypeParam>(300, 500);
}
