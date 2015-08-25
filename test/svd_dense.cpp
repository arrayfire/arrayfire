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
using af::cfloat;
using af::cdouble;

template<typename T>
class svd : public ::testing::Test
{
};

typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;
TYPED_TEST_CASE(svd, TestTypes);

template<typename T>
double get_val(T val)
{
    return val;
}

template<> double get_val<cfloat>(cfloat val)
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

    af::dtype ty = (af::dtype)af::dtype_traits<T>::af_type;

    af::array A = af::randu(M, N, ty);

    //! [ex_svd_reg]
    af::array U, S, Vt;
    af::svd(U, S, Vt, A);

    const int MN = std::min(M, N);

    af::array UU = U(af::span, af::seq(MN));
    af::array SS = af::diag(S, 0, false).as(ty);
    af::array VV = Vt(af::seq(MN), af::span);

    af::array AA = matmul(UU, SS, VV);
    //! [ex_svd_reg]

    std::vector<T> hA(M * N);
    std::vector<T> hAA(M * N);

    A.host(&hA[0]);
    AA.host(&hAA[0]);

    for (int i = 0; i < M * N; i++) {
        ASSERT_NEAR(get_val(hA[i]), get_val(hAA[i]), 1E-3);
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
