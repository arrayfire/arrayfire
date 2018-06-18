/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// NOTE: Tests are known to fail on OSX when utilizing the CPU and OpenCL
// backends for sizes larger than 128x128 or more. You can read more about it on
// issue https://github.com/arrayfire/arrayfire/issues/1617

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/traits.hpp>
#include <iostream>
#include <complex>
#include <testHelpers.hpp>

using std::abs;
using af::array;
using af::cfloat;
using af::cdouble;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::identity;
using af::matmul;
using af::max;

template<typename T>
void inverseTester(const int m, const int n, const int k, double eps)
{
    if (noDoubleTests<T>()) return;
    if (noLAPACKTests()) return;
#if 1
    array A  = cpu_randu<T>(dim4(m, n));
#else
    array A  = randu(m, n, (dtype)dtype_traits<T>::af_type);
#endif

    //! [ex_inverse]
    array IA = inverse(A);
    array I = matmul(A, IA);
    //! [ex_inverse]

    array I2 = identity(m, n, (dtype)dtype_traits<T>::af_type);

    ASSERT_NEAR(0, max<typename dtype_traits<T>::base_type>(abs(real(I - I2))), eps);
    ASSERT_NEAR(0, max<typename dtype_traits<T>::base_type>(abs(imag(I - I2))), eps);
}


template<typename T>
class Inverse : public ::testing::Test
{

};

template<typename T>
double eps();

template<>
double eps<float>() {
  return 0.01f;
}

template<>
double eps<double>() {
  return 1e-5;
}

template<>
double eps<cfloat>() {
  return 0.01f;
}

template<>
double eps<cdouble>() {
  return 1e-5;
}

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_CASE(Inverse, TestTypes);

TYPED_TEST(Inverse, Square) {
    inverseTester<TypeParam>(1000, 1000, 100, eps<TypeParam>());
}

TYPED_TEST(Inverse, SquareMultiplePowerOfTwo) {
    inverseTester<TypeParam>(2048, 2048, 512, eps<TypeParam>());
}
