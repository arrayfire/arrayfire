/*******************************************************
 * Copyright (c) 2014, ArrayFire
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

///////////////////////////////// CPP ////////////////////////////////////
//
// TEST(Cholesky, CPP)
// {
//     if (noDoubleTests<float>()) return;

//     int resultIdx = 0;

//     vector<af::dim4> numDims;
//     vector<vector<float> > in;
//     vector<vector<float> > tests;
//     readTests<float, float, float>(string(TEST_DIR"/lapack/cholesky.test"),numDims,in,tests);

//     af::dim4 idims = numDims[0];
//     af::array input(idims, &(in[0].front()));
//     int info;
//     af::array output = af::cholesky(input, &info, false);

//     af::dim4 odims = output.dims();

//     // Get result
//     float* outData = new float[tests[resultIdx].size()];
//     output.host((void*)outData);

//     // Compare result
//     for (int y = 0; y < odims[1]; ++y) {
//         for (int x = 0; x < odims[0]; ++x) {
//             // Test only lower half
//             if( x >= y) {
//                 int elIter = y * odims[0] + x;
//                 ASSERT_NEAR(tests[resultIdx][elIter], outData[elIter], 0.001) << "at: " << elIter << std::endl;
//             }
//         }
//     }

//     // Delete
//     delete[] outData;
// }

template<typename T>
void choleskyTester(const int n, double eps, bool is_upper)
{
    if (noDoubleTests<T>()) return;

    af::dtype ty = (af::dtype)af::dtype_traits<T>::af_type;

    // Prepare positive definite matrix
    af::array a = af::randu(n, n, ty);
    af::array b = 10 * n * af::identity(n, n, ty);
    af::array in = matmul(a.H(), a) + b;

    int info = 0;
    af::array out = cholesky(in, &info, is_upper);

    af::array re = is_upper ? matmul(out.H(), out) : matmul(out, out.H());

    ASSERT_NEAR(0, af::max<double>(af::abs(real(in - re))), eps);
    ASSERT_NEAR(0, af::max<double>(af::abs(imag(in - re))), eps);
}

#define CHOLESKY_BIG_TESTS(T, eps)              \
    TEST(Cholesky, T##Upper)                    \
    {                                           \
        choleskyTester<T>( 500, eps, true );    \
    }                                           \
    TEST(Cholesky, T##Lower)                    \
    {                                           \
        choleskyTester<T>(1000, eps, false);    \
    }                                           \
    TEST(Cholesky, T##UpperMultiple)            \
    {                                           \
        choleskyTester<T>(1024, eps, true );    \
    }                                           \
    TEST(Cholesky, T##LowerMultiple)            \
    {                                           \
        choleskyTester<T>( 512, eps, false);    \
    }                                           \


CHOLESKY_BIG_TESTS(float, 0.05)
CHOLESKY_BIG_TESTS(double, 1E-8)
CHOLESKY_BIG_TESTS(cfloat, 0.05)
CHOLESKY_BIG_TESTS(cdouble, 1E-8)
