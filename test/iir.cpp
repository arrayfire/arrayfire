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
class fir : public ::testing::Test
{
public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;
TYPED_TEST_CASE(fir, TestTypes);

static double get_real(double val) { return val; }
static double get_real(cfloat val) { return std::real(val); }
static double get_real(cdouble val) { return std::real(val); }

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
                ASSERT_NEAR(get_real(hy[j * yrows + i]),
                            get_real(hc[j * crows + i]), 0.01);
            }
        }
    } catch (af::exception &ex) {
        FAIL() << ex.what();
    }
}

TYPED_TEST(fir, XVecBMat)
{
    firTest<TypeParam>(10000, 1, 50, 10);
}

TYPED_TEST(fir, XMatBVec)
{
    firTest<TypeParam>(5000, 10, 100, 1);
}

TYPED_TEST(fir, XMatBMat)
{
    firTest<TypeParam>(5000, 10, 50, 10);
}
