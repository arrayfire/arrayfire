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
using af::cfloat;
using af::cdouble;
using af::dtype_traits;

template<typename T>
void inverseTester(const int m, const int n, const int k, double eps)
{
    if (noDoubleTests<T>()) return;
    if (noLAPACKTests()) return;
#if 1
    af::array A  = cpu_randu<T>(af::dim4(m, n));
#else
    af::array A  = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
#endif

    //! [ex_inverse]
    af::array IA = inverse(A);
    af::array I = af::matmul(A, IA);
    //! [ex_inverse]

    af::array I2 = af::identity(m, n, (af::dtype)af::dtype_traits<T>::af_type);

    ASSERT_NEAR(0, af::max<typename dtype_traits<T>::base_type>(af::abs(real(I - I2))), eps);
    ASSERT_NEAR(0, af::max<typename dtype_traits<T>::base_type>(af::abs(imag(I - I2))), eps);
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
