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
#include <testHelpers.hpp>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::convolve1;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::exception;
using af::fir;
using af::iir;
using af::randu;
using std::string;
using std::vector;

template<typename T>
class filter : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;
TYPED_TEST_SUITE(filter, TestTypes);

template<typename T>
void firTest(const int xrows, const int xcols, const int brows,
             const int bcols) {
    SUPPORTED_TYPE_CHECK(T);
    try {
        dtype ty = (dtype)dtype_traits<T>::af_type;
        array x  = randu(xrows, xcols, ty);
        array b  = randu(brows, bcols, ty);

        array y = fir(b, x);
        array c = convolve1(x, b, AF_CONV_EXPAND);

        const int ycols = xcols * bcols;
        const int crows = xrows + brows - 1;
        const int yrows = xrows;

        vector<T> hy(yrows * ycols);
        vector<T> hc(crows * ycols);

        y.host(&hy[0]);
        c.host(&hc[0]);

        for (int j = 0; j < ycols; j++) {
            for (int i = 0; i < yrows; i++) {
                ASSERT_NEAR(real(hy[j * yrows + i]), real(hc[j * crows + i]),
                            0.01);
            }
        }
    } catch (exception &ex) { FAIL() << ex.what(); }
}

TYPED_TEST(filter, firVecVec) { firTest<TypeParam>(10000, 1, 1000, 1); }

TYPED_TEST(filter, firVecMat) { firTest<TypeParam>(10000, 1, 50, 10); }

TYPED_TEST(filter, firMatVec) { firTest<TypeParam>(5000, 10, 100, 1); }

TYPED_TEST(filter, firMatMat) { firTest<TypeParam>(5000, 10, 50, 10); }

template<typename T>
void iirA0Test(const int xrows, const int xcols, const int brows,
               const int bcols) {
    SUPPORTED_TYPE_CHECK(T);
    try {
        dtype ty    = (dtype)dtype_traits<T>::af_type;
        array x     = randu(xrows, xcols, ty);
        array b     = randu(brows, bcols, ty);
        array a     = randu(1, bcols, ty);
        array bNorm = b / tile(a, brows);

        array y = iir(b, a, x);
        array c = convolve1(x, bNorm, AF_CONV_EXPAND);

        const int ycols = xcols * bcols;
        const int crows = xrows + brows - 1;
        const int yrows = xrows;

        vector<T> hy(yrows * ycols);
        vector<T> hc(crows * ycols);

        y.host(&hy[0]);
        c.host(&hc[0]);

        for (int j = 0; j < ycols; j++) {
            for (int i = 0; i < yrows; i++) {
                ASSERT_NEAR(real(hy[j * yrows + i]), real(hc[j * crows + i]),
                            0.01);
            }
        }
    } catch (exception &ex) { FAIL() << ex.what(); }
}

TYPED_TEST(filter, iirA0VecVec) { iirA0Test<TypeParam>(10000, 1, 1000, 1); }

TYPED_TEST(filter, iirA0VecMat) { iirA0Test<TypeParam>(10000, 1, 50, 10); }

TYPED_TEST(filter, iirA0MatVec) { iirA0Test<TypeParam>(5000, 10, 100, 1); }

TYPED_TEST(filter, iirA0MatMat) { iirA0Test<TypeParam>(5000, 10, 50, 10); }

template<typename T>
void iirTest(const char *testFile) {
    SUPPORTED_TYPE_CHECK(T);
    vector<dim4> inDims;

    vector<vector<T>> inputs;
    vector<vector<T>> outputs;
    readTests<T, T, float>(testFile, inDims, inputs, outputs);

    try {
        array a = array(inDims[0], &inputs[0][0]);
        array b = array(inDims[1], &inputs[1][0]);
        array x = array(inDims[2], &inputs[2][0]);

        array y        = iir(b, a, x);
        vector<T> gold = outputs[0];
        ASSERT_EQ(gold.size(), (size_t)y.elements());

        vector<T> out(y.elements());
        y.host(&out[0]);

        for (size_t i = 0; i < gold.size(); i++) {
            ASSERT_NEAR(real(out[i]), real(gold[i]), 0.01) << "at: " << i;
        }

    } catch (exception &ex) { FAIL() << ex.what(); }
}

TYPED_TEST(filter, iirVecVec) {
    iirTest<TypeParam>(TEST_DIR "/iir/iir_vv.test");
}

TYPED_TEST(filter, iirVecMat) {
    iirTest<TypeParam>(TEST_DIR "/iir/iir_vm.test");
}

TYPED_TEST(filter, iirMatMat) {
    iirTest<TypeParam>(TEST_DIR "/iir/iir_mm.test");
}
