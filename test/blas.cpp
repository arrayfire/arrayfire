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
#include <af/blas.h>
#include <af/traits.hpp>
#include <af/defines.h>
#include <testHelpers.hpp>
#include <string>

using std::string;
using std::cout;
using std::endl;
using std::ostream_iterator;
using std::copy;
using std::vector;

template<typename T>
class MatrixMultiply : public ::testing::Test
{

};

typedef ::testing::Types<float, af::cfloat, double, af::cdouble> TestTypes;
TYPED_TEST_CASE(MatrixMultiply, TestTypes);

template<typename T, bool isBVector>
void MatMulCheck(string TestFile)
{
    if (noDoubleTests<T>()) return;

    using std::vector;
    vector<af::dim4> numDims;

    vector<vector<T> > hData;
    vector<vector<T> > tests;
    readTests<T,T,int>(TestFile, numDims, hData, tests);

    af_array a, aT, b, bT;
    ASSERT_EQ(AF_SUCCESS,
            af_create_array(&a, &hData[0].front(), numDims[0].ndims(), numDims[0].get(), (af_dtype) af::dtype_traits<T>::af_type));
    af::dim4 atdims = numDims[0];
    {
        dim_t f  =    atdims[0];
        atdims[0]   =    atdims[1];
        atdims[1]   =    f;
    }
    ASSERT_EQ(AF_SUCCESS,
            af_moddims(&aT, a, atdims.ndims(), atdims.get()));
    ASSERT_EQ(AF_SUCCESS,
            af_create_array(&b, &hData[1].front(), numDims[1].ndims(), numDims[1].get(), (af_dtype) af::dtype_traits<T>::af_type));
    af::dim4 btdims = numDims[1];
    {
        dim_t f = btdims[0];
        btdims[0] = btdims[1];
        btdims[1] = f;
    }
    ASSERT_EQ(AF_SUCCESS,
            af_moddims(&bT, b, btdims.ndims(), btdims.get()));

    vector<af_array> out(tests.size(), 0);
    if(isBVector) {
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[0] , aT, b,    AF_MAT_NONE,    AF_MAT_NONE));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[1] , bT, a,   AF_MAT_NONE,    AF_MAT_NONE));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[2] , b, a,    AF_MAT_TRANS,       AF_MAT_NONE));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[3] , bT, aT,   AF_MAT_NONE,    AF_MAT_TRANS));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[4] , b, aT,    AF_MAT_TRANS,       AF_MAT_TRANS));
    }
    else {
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[0] , a, b, AF_MAT_NONE,   AF_MAT_NONE));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[1] , a, bT, AF_MAT_NONE,   AF_MAT_TRANS));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[2] , a, bT, AF_MAT_TRANS,      AF_MAT_NONE));
        ASSERT_EQ(AF_SUCCESS, af_matmul( &out[3] , aT, bT, AF_MAT_TRANS,      AF_MAT_TRANS));
    }

    for(size_t i = 0; i < tests.size(); i++) {
        dim_t elems;
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&elems, out[i]));
        vector<T> h_out(elems);
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void *)&h_out.front(), out[i]));

        if( false == equal(h_out.begin(), h_out.end(), tests[i].begin()) ) {

            cout << "Failed test " << i << "\nCalculated: " << endl;
            copy(h_out.begin(), h_out.end(), ostream_iterator<T>(cout, ", "));
            cout << "Expected: " << endl;
            copy(tests[i].begin(), tests[i].end(), ostream_iterator<T>(cout, ", "));
            FAIL();
        }
    }

    ASSERT_EQ(AF_SUCCESS, af_release_array(a));
    ASSERT_EQ(AF_SUCCESS, af_release_array(aT));
    ASSERT_EQ(AF_SUCCESS, af_release_array(b));
    ASSERT_EQ(AF_SUCCESS, af_release_array(bT));

    for (size_t i = 0; i <  out.size(); i++) {
        ASSERT_EQ(AF_SUCCESS, af_release_array(out[i]));
    }
}

TYPED_TEST(MatrixMultiply, Square)
{
    MatMulCheck<TypeParam, false>(TEST_DIR"/blas/Basic.test");
}

TYPED_TEST(MatrixMultiply, NonSquare)
{
    MatMulCheck<TypeParam, false>(TEST_DIR"/blas/NonSquare.test");
}

TYPED_TEST(MatrixMultiply, SquareVector)
{
    MatMulCheck<TypeParam, true>(TEST_DIR"/blas/SquareVector.test");
}

TYPED_TEST(MatrixMultiply, RectangleVector)
{
    MatMulCheck<TypeParam, true>(TEST_DIR"/blas/RectangleVector.test");
}

template<typename T, bool isBVector>
void cppMatMulCheck(string TestFile)
{
    if (noDoubleTests<T>()) return;

    using std::vector;
    vector<af::dim4> numDims;

    vector<vector<T> > hData;
    vector<vector<T> > tests;
    readTests<T,T,int>(TestFile, numDims, hData, tests);

    af::array a(numDims[0], &hData[0].front());
    af::array b(numDims[1], &hData[1].front());

    af::dim4 atdims = numDims[0];
    {
        dim_t f  =    atdims[0];
        atdims[0]   =    atdims[1];
        atdims[1]   =    f;
    }
    af::dim4 btdims = numDims[1];
    {
        dim_t f = btdims[0];
        btdims[0] = btdims[1];
        btdims[1] = f;
    }

    af::array aT = moddims(a, atdims.ndims(), atdims.get());
    af::array bT = moddims(b, btdims.ndims(), btdims.get());

    vector<af::array> out(tests.size());
    if(isBVector) {
        out[0] = af::matmul(aT, b,    AF_MAT_NONE,    AF_MAT_NONE);
        out[1] = af::matmul(bT, a,   AF_MAT_NONE,    AF_MAT_NONE);
        out[2] = af::matmul(b, a,    AF_MAT_TRANS,       AF_MAT_NONE);
        out[3] = af::matmul(bT, aT,   AF_MAT_NONE,    AF_MAT_TRANS);
        out[4] = af::matmul(b, aT,    AF_MAT_TRANS,       AF_MAT_TRANS);
    }
    else {
        out[0] = af::matmul(a, b, AF_MAT_NONE,   AF_MAT_NONE);
        out[1] = af::matmul(a, bT, AF_MAT_NONE,   AF_MAT_TRANS);
        out[2] = af::matmul(a, bT, AF_MAT_TRANS,      AF_MAT_NONE);
        out[3] = af::matmul(aT, bT, AF_MAT_TRANS,      AF_MAT_TRANS);
    }

    for(size_t i = 0; i < tests.size(); i++) {
        dim_t elems = out[i].elements();
        vector<T> h_out(elems);
        out[i].host((void*)&h_out.front());

        if (false == equal(h_out.begin(), h_out.end(), tests[i].begin())) {

            cout << "Failed test " << i << "\nCalculated: " << endl;
            copy(h_out.begin(), h_out.end(), ostream_iterator<T>(cout, ", "));
            cout << "Expected: " << endl;
            copy(tests[i].begin(), tests[i].end(), ostream_iterator<T>(cout, ", "));
            FAIL();
        }
    }
}

TYPED_TEST(MatrixMultiply, Square_CPP)
{
    cppMatMulCheck<TypeParam, false>(TEST_DIR"/blas/Basic.test");
}

TYPED_TEST(MatrixMultiply, NonSquare_CPP)
{
    cppMatMulCheck<TypeParam, false>(TEST_DIR"/blas/NonSquare.test");
}

TYPED_TEST(MatrixMultiply, SquareVector_CPP)
{
    cppMatMulCheck<TypeParam, true>(TEST_DIR"/blas/SquareVector.test");
}

TYPED_TEST(MatrixMultiply, RectangleVector_CPP)
{
    cppMatMulCheck<TypeParam, true>(TEST_DIR"/blas/RectangleVector.test");
}

#define DEVICE_ITERATE(func) do {                                           \
    const char* ENV = getenv("AF_MULTI_GPU_TESTS");                         \
    if(ENV && ENV[0] == '0') {                                              \
        func;                                                               \
    } else {                                                                \
        int oldDevice = af::getDevice();                                    \
        for(int i = 0; i < af::getDeviceCount(); i++) {                     \
            af::setDevice(i);                                               \
            func;                                                           \
        }                                                                   \
        af::setDevice(oldDevice);                                           \
    }                                                                       \
} while(0);


TYPED_TEST(MatrixMultiply, MultiGPUSquare_CPP)
{
    DEVICE_ITERATE((cppMatMulCheck<TypeParam, false>(TEST_DIR"/blas/Basic.test")));
}

TYPED_TEST(MatrixMultiply, MultiGPUNonSquare_CPP)
{
    DEVICE_ITERATE((cppMatMulCheck<TypeParam, false>(TEST_DIR"/blas/NonSquare.test")));
}

TYPED_TEST(MatrixMultiply, MultiGPUSquareVector_CPP)
{
    DEVICE_ITERATE((cppMatMulCheck<TypeParam, true>(TEST_DIR"/blas/SquareVector.test")));
}

TYPED_TEST(MatrixMultiply, MultiGPURectangleVector_CPP)
{
    DEVICE_ITERATE((cppMatMulCheck<TypeParam, true>(TEST_DIR"/blas/RectangleVector.test")));
}

TEST(MatrixMultiply, Batched)
{
    const int M = 512;
    const int K = 1024;
    const int N = 32;
    const int D2 = 2;
    const int D3 = 3;

    for (int d3 = 1; d3 <= D3; d3 *= D3) {
        for (int d2 = 1; d2 <= D2; d2 *= D2) {
            af::array a = af::randu(M, K, d2, d3);
            af::array b = af::randu(K, N, d2, d3);
            af::array c = af::matmul(a, b);

            for (int j = 0; j < d3; j++) {
                for (int i = 0; i < d2; i++) {
                    af::array a_ij = a(af::span, af::span, i, j);
                    af::array b_ij = b(af::span, af::span, i, j);
                    af::array c_ij = c(af::span, af::span, i, j);
                    af::array res = af::matmul(a_ij, b_ij);
                    ASSERT_LT(af::max<float>(af::abs(c_ij - res)), 1E-5)
                        << " for d2 = " << d2 << " for d3 = " << d3;
                }
            }
        }
    }
}

#undef DEVICE_ITERATE

TEST(MatrixMultiply, ISSUE_1882)
{
    const int m = 2;
    const int n = 3;
    af::array A = af::randu(m, n);
    af::array BB = af::randu(n, m);
    af::array B = BB(0, af::span);

    af::array res1 = af::matmul(A.T(), B.T());
    af::array res2 = af::matmulTT(A, B);

    std::vector<float> hres1(res1.elements());
    std::vector<float> hres2(res2.elements());

    res1.host(&hres1.front());
    res2.host(&hres2.front());

    ASSERT_EQ(hres1.size(), hres2.size());

    for (size_t i = 0; i < hres1.size(); i++) {
        ASSERT_NEAR(hres1[i], hres2[i], 1E-5);
    }
}
