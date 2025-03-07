/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <half.hpp>
#include <testHelpers.hpp>

#include <cmath>
#include <vector>

using af::array;
using af::constant;
using af::deviceGC;
using af::diag;
using af::dim4;
using af::exception;
using af::max;
using af::seq;
using af::span;
using af::sum;
using std::abs;
using std::vector;

template<typename T>
class Diagonal : public ::testing::Test {};

typedef ::testing::Types<float, double, int, uint, char, unsigned char,
                         half_float::half>
    TestTypes;
TYPED_TEST_SUITE(Diagonal, TestTypes);

TYPED_TEST(Diagonal, Create) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    try {
        static const int size = 1000;
        vector<TypeParam> input(size * size);
        for (int i = 0; i < size; i++) { input[i] = i; }
        for (int jj = 10; jj < size; jj += 100) {
            array data(jj, &input.front(), afHost);
            array out = diag(data, 0, false);

            vector<TypeParam> h_out(out.elements());
            out.host(&h_out.front());

            for (int i = 0; i < (int)out.dims(0); i++) {
                for (int j = 0; j < (int)out.dims(1); j++) {
                    if (i == j)
                        ASSERT_EQ(input[i], h_out[i * out.dims(0) + j]);
                    else
                        ASSERT_EQ(TypeParam(0), h_out[i * out.dims(0) + j]);
                }
            }
        }
    } catch (const exception& ex) { FAIL() << ex.what(); }
}

TYPED_TEST(Diagonal, DISABLED_CreateLargeDim) {
    SUPPORTED_TYPE_CHECK(TypeParam);
    try {
        deviceGC();
        {
            static const size_t largeDim = 65535 + 1;
            array diagvals               = constant(1, largeDim);
            array out                    = diag(diagvals, 0, false);

            ASSERT_EQ(largeDim, sum<float>(out));
        }
    } catch (const exception& ex) { FAIL() << ex.what(); }
}

TYPED_TEST(Diagonal, Extract) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    try {
        static const int size = 1000;
        vector<TypeParam> input(size * size);
        for (int i = 0; i < size * size; i++) { input[i] = i; }
        for (int jj = 10; jj < size; jj += 100) {
            array data(jj, jj, &input.front(), afHost);
            array out = diag(data, 0);

            vector<TypeParam> h_out(out.elements());
            out.host(&h_out.front());

            for (int i = 0; i < (int)out.dims(0); i++) {
                ASSERT_EQ(input[i * data.dims(0) + i], h_out[i]);
            }
        }
    } catch (const exception& ex) { FAIL() << ex.what(); }
}

TYPED_TEST(Diagonal, ExtractLargeDim) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    try {
        static const size_t n        = 10;
        static const size_t largeDim = 65535 + 1;

        array largedata = constant(1, n, n, largeDim);
        array out       = diag(largedata, 0);

        ASSERT_EQ(n * largeDim, sum<float>(out));

        largedata  = constant(1, n, n, 1, largeDim);
        array out1 = diag(largedata, 0);

        ASSERT_EQ(n * largeDim, sum<float>(out1));

    } catch (const exception& ex) { FAIL() << ex.what(); }
}

TYPED_TEST(Diagonal, ExtractRect) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    try {
        static const int size0 = 1000, size1 = 900;
        vector<TypeParam> input(size0 * size1);
        for (int i = 0; i < size0 * size1; i++) { input[i] = i; }

        for (int jj = 10; jj < size0; jj += 100) {
            for (int kk = 10; kk < size1; kk += 90) {
                array data(jj, kk, &input.front(), afHost);
                array out = diag(data, 0);

                vector<TypeParam> h_out(out.elements());
                out.host(&h_out.front());

                ASSERT_EQ(out.dims(0), std::min(jj, kk));

                for (int i = 0; i < (int)out.dims(0); i++) {
                    ASSERT_EQ(input[i * data.dims(0) + i], h_out[i]);
                }
            }
        }
    } catch (const exception& ex) { FAIL() << ex.what(); }
}

TEST(Diagonal, ExtractGFOR) {
    dim4 dims = dim4(100, 100, 3);
    array A   = round(100 * randu(dims));
    array B   = constant(0, 100, 1, 3);

    gfor(seq ii, 3) { B(span, span, ii) = diag(A(span, span, ii)); }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = diag(A(span, span, ii));
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}
