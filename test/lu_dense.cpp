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

TEST(LU, InPlaceSmall)
{
    if (noDoubleTests<float>()) return;
    if (noLAPACKTests()) return;

    int resultIdx = 0;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, float>(string(TEST_DIR"/lapack/lu.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, &(in[0].front()));
    af::array output, pivot;
    af::lu(output, pivot, input);

    af::dim4 odims = output.dims();

    // Get result
    float* outData = new float[tests[resultIdx].size()];
    output.host((void*)outData);

    // Compare result
    for (int y = 0; y < (int)odims[1]; ++y) {
        for (int x = 0; x < (int)odims[0]; ++x) {
            // Check only upper triangle
            if(x <= y) {
            int elIter = y * odims[0] + x;
            ASSERT_NEAR(tests[resultIdx][elIter], outData[elIter], 0.001) << "at: " << elIter << std::endl;
            }
        }
    }

    // Delete
    delete[] outData;
}

TEST(LU, SplitSmall)
{
    if (noDoubleTests<float>()) return;
    if (noLAPACKTests()) return;

    int resultIdx = 0;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, float>(string(TEST_DIR"/lapack/lufactorized.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, &(in[0].front()));
    af::array l, u, pivot;
    af::lu(l, u, pivot, input);

    af::dim4 ldims = l.dims();
    af::dim4 udims = u.dims();

    // Get result
    float* lData = new float[ldims.elements()];
    l.host((void*)lData);
    float* uData = new float[udims.elements()];
    u.host((void*)uData);

    // Compare result
    for (int y = 0; y < (int)ldims[1]; ++y) {
        for (int x = 0; x < (int)ldims[0]; ++x) {
            if(x < y) {
                int elIter = y * ldims[0] + x;
                ASSERT_NEAR(tests[resultIdx][elIter], lData[elIter], 0.001) << "at: " << elIter << std::endl;
            }
        }
    }

    resultIdx = 1;

    for (int y = 0; y < (int)udims[1]; ++y) {
        for (int x = 0; x < (int)udims[0]; ++x) {
            int elIter = y * (int)udims[0] + x;
            ASSERT_NEAR(tests[resultIdx][elIter], uData[elIter], 0.001) << "at: " << elIter << std::endl;
        }
    }

    // Delete
    delete[] lData;
    delete[] uData;
}

template<typename T>
void luTester(const int m, const int n, double eps)
{
    if (noDoubleTests<T>()) return;
    if (noLAPACKTests()) return;

#if 1
    af::array a_orig = cpu_randu<T>(af::dim4(m, n));
#else
    af::array a_orig = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
#endif


    //! [ex_lu_unpacked]
    af::array l, u, pivot;
    af::lu(l, u, pivot, a_orig);
    //! [ex_lu_unpacked]

    //! [ex_lu_recon]
    af::array a_recon = af::matmul(l, u);
    af::array a_perm = a_orig(pivot, af::span);
    //! [ex_lu_recon]

    ASSERT_NEAR(0, af::max<typename dtype_traits<T>::base_type>(af::abs(real(a_recon - a_perm))), eps);
    ASSERT_NEAR(0, af::max<typename dtype_traits<T>::base_type>(af::abs(imag(a_recon - a_perm))), eps);

    //! [ex_lu_packed]
    af::array out = a_orig.copy();
    af::array pivot2;
    af::luInPlace(pivot2, out, false);
    //! [ex_lu_packed]

    //! [ex_lu_extract]
    af::array l2 = lower(out,  true);
    af::array u2 = upper(out, false);
    //! [ex_lu_extract]

    ASSERT_EQ(af::count<uint>(pivot == pivot2), pivot.elements());

    int mn = std::min(m, n);
    l2 = l2(af::span, af::seq(mn));
    u2 = u2(af::seq(mn), af::span);

    af::array a_recon2 = af::matmul(l2, u2);
    af::array a_perm2 = a_orig(pivot2, af::span);

    ASSERT_NEAR(0, af::max<typename dtype_traits<T>::base_type>(af::abs(real(a_recon2 - a_perm2))), eps);
    ASSERT_NEAR(0, af::max<typename dtype_traits<T>::base_type>(af::abs(imag(a_recon2 - a_perm2))), eps);

}

template<typename T>
double eps();

template<>
double eps<float>() {
    return 1E-3;
}

template<>
double eps<double>() {
    return 1e-8;
}

template<>
double eps<cfloat>() {
    return 1E-3;
}

template<>
double eps<cdouble>() {
    return 1e-8;
}

template<typename T>
class LU : public ::testing::Test
{

};

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_CASE(LU, TestTypes);

TYPED_TEST(LU, SquareLarge) {
    luTester<TypeParam>(500, 500, eps<TypeParam>());
}

TYPED_TEST(LU, SquareMultipleOfTwoLarge) {
    luTester<TypeParam>(512, 512, eps<TypeParam>());
}

TYPED_TEST(LU, RectangularLarge0) {
    luTester<TypeParam>(1000, 500, eps<TypeParam>());
}

TYPED_TEST(LU, RectangularMultipleOfTwoLarge0) {
    luTester<TypeParam>(1024, 512, eps<TypeParam>());
}

TYPED_TEST(LU, RectangularLarge1) {
    luTester<TypeParam>(500, 1000, eps<TypeParam>());
}

TYPED_TEST(LU, RectangularMultipleOfTwoLarge1) {
    luTester<TypeParam>(512, 1024, eps<TypeParam>());
}
