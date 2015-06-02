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
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using af::cfloat;
using af::cdouble;
using af::dim4;

template<typename T>
class filter : public ::testing::Test
{
public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;
TYPED_TEST_CASE(filter, TestTypes);


template<typename T>
void firTest(const int xrows, const int xcols, const int brows, const int bcols)
{
    if (noDoubleTests<T>()) return;
    try {
        af::dtype ty = (af::dtype)af::dtype_traits<T>::af_type;
        af::array x = af::randu(xrows, xcols, ty);
        af::array b = af::randu(brows, bcols, ty);

        af::array y = af::fir(b, x);
        af::array c = af::convolve1(x, b, AF_CONV_EXPAND);

        const int ycols = xcols * bcols;
        const int crows = xrows + brows - 1;
        const int yrows = xrows;

        vector<T> hy(yrows * ycols);
        vector<T> hc(crows * ycols);

        y.host(&hy[0]);
        c.host(&hc[0]);

        for (int j = 0; j < ycols; j++) {
            for (int i = 0; i < yrows; i++) {
                ASSERT_NEAR(real(hy[j * yrows + i]),
                            real(hc[j * crows + i]), 0.01);
            }
        }
    } catch (af::exception &ex) {
        FAIL() << ex.what();
    }
}

TYPED_TEST(filter, firVecVec)
{
    firTest<TypeParam>(10000, 1, 1000, 1);
}

TYPED_TEST(filter, firVecMat)
{
    firTest<TypeParam>(10000, 1, 50, 10);
}

TYPED_TEST(filter, firMatVec)
{
    firTest<TypeParam>(5000, 10, 100, 1);
}

TYPED_TEST(filter, firMatMat)
{
    firTest<TypeParam>(5000, 10, 50, 10);
}

template<typename T>
void iirA0Test(const int xrows, const int xcols, const int brows, const int bcols)
{
    if (noDoubleTests<T>()) return;
    try {
        af::dtype ty = (af::dtype)af::dtype_traits<T>::af_type;
        af::array x = af::randu(xrows, xcols, ty);
        af::array b = af::randu(brows, bcols, ty);
        af::array a = af::randu(    1, bcols, ty);
        af::array bNorm = b / tile(a, brows);

        af::array y = af::iir(b, a, x);
        af::array c = af::convolve1(x, bNorm, AF_CONV_EXPAND);

        const int ycols = xcols * bcols;
        const int crows = xrows + brows - 1;
        const int yrows = xrows;

        vector<T> hy(yrows * ycols);
        vector<T> hc(crows * ycols);

        y.host(&hy[0]);
        c.host(&hc[0]);

        for (int j = 0; j < ycols; j++) {
            for (int i = 0; i < yrows; i++) {
                ASSERT_NEAR(real(hy[j * yrows + i]),
                            real(hc[j * crows + i]), 0.01);
            }
        }
    } catch (af::exception &ex) {
        FAIL() << ex.what();
    }
}

TYPED_TEST(filter, iirA0VecVec)
{
    iirA0Test<TypeParam>(10000, 1, 1000, 1);
}

TYPED_TEST(filter, iirA0VecMat)
{
    iirA0Test<TypeParam>(10000, 1, 50, 10);
}

TYPED_TEST(filter, iirA0MatVec)
{
    iirA0Test<TypeParam>(5000, 10, 100, 1);
}

TYPED_TEST(filter, iirA0MatMat)
{
    iirA0Test<TypeParam>(5000, 10, 50, 10);
}

template<typename T>
void iirTest(const char *testFile)
{
    if (noDoubleTests<T>()) return;
    vector<af::dim4> inDims;

    vector<vector<T> > inputs;
    vector<vector<T> > outputs;
    readTests<T, T, float> (testFile, inDims, inputs, outputs);

    try {
        af::array a = af::array(inDims[0], &inputs[0][0]);
        af::array b = af::array(inDims[1], &inputs[1][0]);
        af::array x = af::array(inDims[2], &inputs[2][0]);

        af::array y = af::iir(b, a, x);
        std::vector<T> gold = outputs[0];
        ASSERT_EQ(gold.size(), (size_t)y.elements());

        std::vector<T> out(y.elements());
        y.host(&out[0]);

        for(size_t i = 0; i < gold.size(); i++) {
            ASSERT_NEAR(real(out[i]), real(gold[i]), 0.01) << "at: " << i;
        }

    } catch (af::exception &ex) {
        FAIL() << ex.what();
    }
}

TYPED_TEST(filter, iirVecVec)
{
    iirTest<TypeParam>(TEST_DIR"/iir/iir_vv.test");
}

TYPED_TEST(filter, iirVecMat)
{
    iirTest<TypeParam>(TEST_DIR"/iir/iir_vm.test");
}

TYPED_TEST(filter, iirMatMat)
{
    iirTest<TypeParam>(TEST_DIR"/iir/iir_mm.test");
}
