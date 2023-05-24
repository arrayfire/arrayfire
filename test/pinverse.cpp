/*******************************************************
 * Copyright (c) 2018, ArrayFire
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
#include <limits>

using af::array;
using af::cdouble;
using af::cfloat;
using af::constant;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::exception;
using af::identity;
using af::matmul;
using af::max;
using af::pinverse;
using af::randu;
using af::span;
using std::abs;
using std::string;
using std::vector;

template<typename T>
array makeComplex(dim4 dims, const vector<T>& real, const vector<T>& imag) {
    array realArr(dims, &real.front());
    array imagArr(dims, &imag.front());
    return af::complex(realArr, imagArr);
}

template<typename T>
array readTestInput(string testFilePath) {
    typedef typename dtype_traits<T>::base_type InBaseType;
    dtype outAfType = (dtype)dtype_traits<T>::af_type;

    vector<dim4> dimsVec;
    vector<vector<InBaseType>> inVec;
    vector<vector<InBaseType>> goldVec;
    readTestsFromFile<InBaseType, InBaseType>(testFilePath, dimsVec, inVec,
                                              goldVec);
    dim4 inDims = dimsVec[0];

    if (outAfType == c32 || outAfType == c64) {
        return makeComplex(inDims, inVec[1], inVec[2]);
    } else {
        return array(inDims, &inVec[0].front());
    }
}

template<typename T>
array readTestGold(string testFilePath) {
    typedef typename dtype_traits<T>::base_type InBaseType;
    dtype outAfType = (dtype)dtype_traits<T>::af_type;

    vector<dim4> dimsVec;
    vector<vector<InBaseType>> inVec;
    vector<vector<InBaseType>> goldVec;
    readTestsFromFile<InBaseType, InBaseType>(testFilePath, dimsVec, inVec,
                                              goldVec);
    dim4 goldDims(dimsVec[0][1], dimsVec[0][0]);

    if (outAfType == c32 || outAfType == c64) {
        return makeComplex(goldDims, goldVec[1], goldVec[2]);
    } else {
        return array(goldDims, &goldVec[0].front());
    }
}

template<typename T>
class Pinverse : public ::testing::Test {};

// Epsilons taken from test/inverse.cpp
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

template<typename T>
double relEps(array in) {
    typedef typename af::dtype_traits<T>::base_type InBaseType;
    double fixed_eps = eps<T>();
    double calc_eps  = std::numeric_limits<InBaseType>::epsilon() *
                      std::max(in.dims(0), in.dims(1)) * af::max<double>(in);
    // Use the fixed values above if calculated error tolerance is unnecessarily
    // too small
    return std::max(fixed_eps, calc_eps);
}

typedef ::testing::Types<float, cfloat, double, cdouble> TestTypes;
TYPED_TEST_SUITE(Pinverse, TestTypes);

// Test Moore-Penrose conditions in the following first 4 tests
// See https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Definition
TYPED_TEST(Pinverse, AApinvA_A) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    array in = readTestInput<TypeParam>(
        string(TEST_DIR "/pinverse/pinverse10x8.test"));
    array inpinv = pinverse(in);
    array out    = matmul(in, inpinv, in);
    ASSERT_ARRAYS_NEAR(in, out, eps<TypeParam>());
}

TYPED_TEST(Pinverse, ApinvAApinv_Apinv) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    array in = readTestInput<TypeParam>(
        string(TEST_DIR "/pinverse/pinverse10x8.test"));
    array inpinv = pinverse(in);
    array out    = matmul(inpinv, in, inpinv);
    ASSERT_ARRAYS_NEAR(inpinv, out, eps<TypeParam>());
}

TYPED_TEST(Pinverse, AApinv_IsHermitian) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    array in = readTestInput<TypeParam>(
        string(TEST_DIR "/pinverse/pinverse10x8.test"));
    array inpinv = pinverse(in);
    array aapinv = matmul(in, inpinv);
    array out    = matmul(in, inpinv).H();
    ASSERT_ARRAYS_NEAR(aapinv, out, eps<TypeParam>());
}

TYPED_TEST(Pinverse, ApinvA_IsHermitian) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    array in = readTestInput<TypeParam>(
        string(TEST_DIR "/pinverse/pinverse10x8.test"));
    array inpinv = pinverse(in);
    array apinva = af::matmul(inpinv, in);
    array out    = af::matmul(inpinv, in).H();
    ASSERT_ARRAYS_NEAR(apinva, out, eps<TypeParam>());
}

TYPED_TEST(Pinverse, Large) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    array in = readTestInput<TypeParam>(
        string(TEST_DIR "/pinverse/pinv_640x480_inputs.test"));
    array inpinv = pinverse(in);
    array out    = matmul(in, inpinv, in);
    ASSERT_ARRAYS_NEAR(in, out, relEps<TypeParam>(in));
}

TYPED_TEST(Pinverse, LargeTall) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    array in = readTestInput<TypeParam>(
                   string(TEST_DIR "/pinverse/pinv_640x480_inputs.test"))
                   .T();
    array inpinv = pinverse(in);
    array out    = matmul(in, inpinv, in);
    ASSERT_ARRAYS_NEAR(in, out, relEps<TypeParam>(in));
}

TEST(Pinverse, Square) {
    array in =
        readTestInput<float>(string(TEST_DIR "/pinverse/pinverse10x10.test"));
    array inpinv = pinverse(in);
    array out    = matmul(in, inpinv, in);
    ASSERT_ARRAYS_NEAR(in, out, eps<float>());
}

TEST(Pinverse, Dim1GtDim0) {
    array in =
        readTestInput<float>(string(TEST_DIR "/pinverse/pinverse8x10.test"));
    array inpinv = pinverse(in);
    array out    = matmul(in, inpinv, in);
    ASSERT_ARRAYS_NEAR(in, out, eps<float>());
}

TEST(Pinverse, CompareWithNumpy) {
    array in =
        readTestInput<float>(string(TEST_DIR "/pinverse/pinverse10x8.test"));
    array gold =
        readTestGold<float>(string(TEST_DIR "/pinverse/pinverse10x8.test"));
    array out = pinverse(in);
    ASSERT_ARRAYS_NEAR(gold, out, relEps<float>(gold));
}

TEST(Pinverse, SmallSigValExistsFloat) {
    array in =
        readTestInput<float>(string(TEST_DIR "/pinverse/pinverse10x8.test"));
    const dim_t dim0 = in.dims(0);
    const dim_t dim1 = in.dims(1);

    // Generate sigma with small non-zero value
    af::array u;
    af::array vT;
    af::array sVec;
    af::svd(u, sVec, vT, in);
    dim_t sSize = sVec.elements();

    sVec(2)         = 1e-12;
    af::array s     = af::diag(sVec, 0, false);
    af::array zeros = af::constant(0, dim0 > sSize ? dim0 - sSize : sSize,
                                   dim1 > sSize ? dim1 - sSize : sSize);
    s               = af::join(dim0 > dim1 ? 0 : 1, s, zeros);

    // Make new input array that has a small non-zero value in its SVD sigma
    in           = af::matmul(u, s, vT);
    array inpinv = pinverse(in);
    array out    = matmul(in, inpinv, in);

    ASSERT_ARRAYS_NEAR(in, out, eps<float>());
}

TEST(Pinverse, SmallSigValExistsDouble) {
    SUPPORTED_TYPE_CHECK(double);
    array in =
        readTestInput<double>(string(TEST_DIR "/pinverse/pinverse10x8.test"));
    const dim_t dim0 = in.dims(0);
    const dim_t dim1 = in.dims(1);

    // Generate sigma with small non-zero value
    array u;
    array vT;
    array sVec;
    svd(u, sVec, vT, in);
    dim_t sSize = sVec.elements();

    sVec(2)     = (double)1e-16;
    array s     = diag(sVec, 0, false);
    array zeros = constant(0, dim0 > sSize ? dim0 - sSize : sSize,
                           dim1 > sSize ? dim1 - sSize : sSize, f64);
    s           = join(dim0 > dim1 ? 0 : 1, s, zeros);

    // Make new input array that has a small non-zero value in its SVD sigma
    in           = matmul(u, s, vT);
    array inpinv = pinverse(in, 1e-15);
    array out    = matmul(in, inpinv, in);

    ASSERT_ARRAYS_NEAR(in, out, eps<double>());
}

TEST(Pinverse, Batching3D) {
    array in =
        readTestInput<float>(string(TEST_DIR "/pinverse/pinverse10x8x2.test"));
    array inpinv0 = pinverse(in(span, span, 0));
    array inpinv1 = pinverse(in(span, span, 1));

    array out  = pinverse(in);
    array out0 = out(span, span, 0);
    array out1 = out(span, span, 1);

    ASSERT_ARRAYS_NEAR(inpinv0, out0, relEps<float>(inpinv0));
    ASSERT_ARRAYS_NEAR(inpinv1, out1, relEps<float>(inpinv1));
}

TEST(Pinverse, Batching4D) {
    array in = readTestInput<float>(
        string(TEST_DIR "/pinverse/pinverse10x8x2x2.test"));
    array inpinv00 = pinverse(in(span, span, 0, 0));
    array inpinv01 = pinverse(in(span, span, 0, 1));
    array inpinv10 = pinverse(in(span, span, 1, 0));
    array inpinv11 = pinverse(in(span, span, 1, 1));

    array out   = pinverse(in);
    array out00 = out(span, span, 0, 0);
    array out01 = out(span, span, 0, 1);
    array out10 = out(span, span, 1, 0);
    array out11 = out(span, span, 1, 1);

    ASSERT_ARRAYS_NEAR(inpinv00, out00, relEps<float>(inpinv00));
    ASSERT_ARRAYS_NEAR(inpinv01, out01, relEps<float>(inpinv01));
    ASSERT_ARRAYS_NEAR(inpinv10, out10, relEps<float>(inpinv10));
    ASSERT_ARRAYS_NEAR(inpinv11, out11, relEps<float>(inpinv11));
}

TEST(Pinverse, CustomTol) {
    array in =
        readTestInput<float>(string(TEST_DIR "/pinverse/pinverse10x8.test"));
    array inpinv = pinverse(in, 1e-12);
    array out    = matmul(in, inpinv, in);
    ASSERT_ARRAYS_NEAR(in, out, eps<float>());
}

TEST(Pinverse, C) {
    array in =
        readTestInput<float>(string(TEST_DIR "/pinverse/pinverse10x8.test"));
    af_array inpinv = 0, identity = 0, out = 0;
    ASSERT_SUCCESS(af_pinverse(&inpinv, in.get(), 1e-6, AF_MAT_NONE));
    ASSERT_SUCCESS(
        af_matmul(&identity, in.get(), inpinv, AF_MAT_NONE, AF_MAT_NONE));
    ASSERT_SUCCESS(
        af_matmul(&out, identity, in.get(), AF_MAT_NONE, AF_MAT_NONE));

    ASSERT_ARRAYS_NEAR(in.get(), out, eps<float>());

    ASSERT_SUCCESS(af_release_array(out));
    ASSERT_SUCCESS(af_release_array(identity));
    ASSERT_SUCCESS(af_release_array(inpinv));
}

TEST(Pinverse, C_CustomTol) {
    array in =
        readTestInput<float>(string(TEST_DIR "/pinverse/pinverse10x8.test"));
    af_array inpinv = 0, identity = 0, out = 0;
    ASSERT_SUCCESS(af_pinverse(&inpinv, in.get(), 1e-12, AF_MAT_NONE));
    ASSERT_SUCCESS(
        af_matmul(&identity, in.get(), inpinv, AF_MAT_NONE, AF_MAT_NONE));
    ASSERT_SUCCESS(
        af_matmul(&out, identity, in.get(), AF_MAT_NONE, AF_MAT_NONE));

    ASSERT_ARRAYS_NEAR(in.get(), out, eps<float>());

    ASSERT_SUCCESS(af_release_array(out));
    ASSERT_SUCCESS(af_release_array(identity));
    ASSERT_SUCCESS(af_release_array(inpinv));
}

TEST(Pinverse, NegativeTol) {
    array in =
        readTestInput<float>(string(TEST_DIR "/pinverse/pinverse10x8.test"));
    array out;
    ASSERT_THROW(out = pinverse(in, -1.f), exception);
}

TEST(Pinverse, InvalidType) {
    array in = constant(0, 10, 8, u8);
    array out;
    ASSERT_THROW(out = pinverse(in, -1.f), exception);
}

TEST(Pinverse, InvalidMatProp) {
    array in = constant(0.f, 10, 8, f32);
    array out;
    ASSERT_THROW(out = pinverse(in, -1.f, AF_MAT_SYM), exception);
}

TEST(Pinverse, DocSnippet) {
    //! [ex_pinverse]
    float hA[] = {0, 1, 2, 3, 4, 5};
    array A(3, 2, hA);
    //  0.0000     3.0000
    //  1.0000     4.0000
    //  2.0000     5.0000

    array Apinv = pinverse(A);
    // -0.7778    -0.1111     0.5556
    //  0.2778     0.1111    -0.0556

    array MustBeA = matmul(A, Apinv, A);
    //  0.0000     3.0000
    //  1.0000     4.0000
    //  2.0000     5.0000
    //! [ex_pinverse]
    ASSERT_ARRAYS_NEAR(A, MustBeA, eps<float>());
}
