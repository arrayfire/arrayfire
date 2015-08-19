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
#include <algorithm>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;

template<typename T>
class Wrap : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned int, intl, uintl, char, unsigned char> TestTypes;

// register the type list
TYPED_TEST_CASE(Wrap, TestTypes);

template<typename T>
double get_val(T val)
{
    return val;
}

template<> double get_val<cfloat>(cfloat val)
{
    return abs(val);
}

template<> double get_val<cdouble>(cdouble val)
{
    return abs(val);
}

template<> double get_val<unsigned char>(unsigned char val)
{
    return ((int)(val)) % 256;
}

template<> double get_val<char>(char val)
{
    return (val != 0);
}

template<typename T>
void wrapTest(const dim_t ix, const dim_t iy,
              const dim_t wx, const dim_t wy,
              const dim_t sx, const dim_t sy,
              const dim_t px, const dim_t py,
              bool cond)
{
    if (noDoubleTests<T>()) return;

    const int nc = 1;

    int lim = std::max((dim_t)2, (dim_t)(250) / (wx * wy));

    af::dtype ty = (af::dtype)af::dtype_traits<T>::af_type;
    af::array in = af::round(lim * af::randu(ix, iy, nc, f32)).as(ty);

    std::vector<T> h_in(in.elements());
    in.host(&h_in[0]);

    std::vector<int> h_factor(ix * iy);

    dim_t ny = (iy + 2 * py - wy) / sy + 1;
    dim_t nx = (ix + 2 * px - wx) / sx + 1;

    for (int idy = 0; idy < ny; idy++) {
        int fy = idy * sy - py;
        if (fy + wy < 0 || fy >= iy) continue;

        for (int idx = 0; idx < nx; idx++) {
            int fx = idx * sx - px;
            if (fx + wx < 0 || fx >= ix) continue;

            for (int ly = 0; ly < wy; ly++) {
                if (fy + ly < 0 || fy + ly >= iy) continue;

                for (int lx = 0; lx < wx; lx++) {
                    if (fx + lx < 0 || fx + lx >= ix) continue;
                    h_factor[(fy + ly) * ix + (fx + lx)]++;
                }
            }
        }
    }

    af::array factor(ix, iy, &h_factor[0]);

    af::array in_dim = af::unwrap(in, wx, wy, sx, sy, px, py, cond);
    af::array res_dim = af::wrap(in_dim, ix, iy, wx, wy, sx, sy, px, py, cond);

    ASSERT_EQ(in.elements(), res_dim.elements());

    std::vector<T> h_res(ix * iy);
    res_dim.host(&h_res[0]);

    for (int n = 0; n < nc; n++) {
        T *iptr = &h_in[n * ix * iy];
        T *rptr = &h_res[n * ix * iy];

        for (int y = 0; y < iy; y++) {
            for (int x = 0; x < ix; x++) {

                // FIXME: Use a better test
                T ival = iptr[y * ix + x];
                T rval = rptr[y * ix + x];
                int factor = h_factor[y * ix + x];

                if (get_val(ival) == 0) continue;

                ASSERT_NEAR(get_val<T>(ival * factor), get_val<T>(rval), 1E-5)
                    << "at " << x << "," << y <<  " for cond  == " << cond << std::endl;
            }
        }

    }
}

#define WRAP_INIT(desc, ix, iy, wx, wy, sx, sy, px,py)              \
    TYPED_TEST(Wrap, Col##desc)                                     \
    {                                                               \
        wrapTest<TypeParam>(ix, iy, wx, wy, sx, sy, px, py, true ); \
    }                                                               \
    TYPED_TEST(Wrap, Row##desc)                                     \
    {                                                               \
        wrapTest<TypeParam>(ix, iy, wx, wy, sx, sy, px, py, false); \
    }

    WRAP_INIT(00, 300, 100,  3,  3,  1,  1,  0,  0);
    WRAP_INIT(01, 300, 100,  3,  3,  1,  1,  1,  1);
    WRAP_INIT(03, 300, 100,  3,  3,  2,  2,  0,  0);
    WRAP_INIT(04, 300, 100,  3,  3,  2,  2,  1,  1);
    WRAP_INIT(05, 300, 100,  3,  3,  2,  2,  2,  2);
    WRAP_INIT(06, 300, 100,  3,  3,  3,  3,  0,  0);
    WRAP_INIT(07, 300, 100,  3,  3,  3,  3,  1,  1);
    WRAP_INIT(08, 300, 100,  3,  3,  3,  3,  2,  2);
    WRAP_INIT(09, 300, 100,  4,  4,  1,  1,  0,  0);
    WRAP_INIT(13, 300, 100,  4,  4,  2,  2,  0,  0);
    WRAP_INIT(14, 300, 100,  4,  4,  2,  2,  1,  1);
    WRAP_INIT(15, 300, 100,  4,  4,  2,  2,  2,  2);
    WRAP_INIT(16, 300, 100,  4,  4,  2,  2,  3,  3);
    WRAP_INIT(17, 300, 100,  4,  4,  4,  4,  0,  0);
    WRAP_INIT(18, 300, 100,  4,  4,  4,  4,  1,  1);
    WRAP_INIT(19, 300, 100,  4,  4,  4,  4,  2,  2);
    WRAP_INIT(27, 300, 100,  8,  8,  8,  8,  0,  0);
    WRAP_INIT(28, 300, 100,  8,  8,  8,  8,  7,  7);
    WRAP_INIT(31, 300, 100, 12, 12, 12, 12,  0,  0);
    WRAP_INIT(32, 300, 100, 12, 12, 12, 12,  2,  2);
    WRAP_INIT(35, 300, 100, 16, 16, 16, 16, 15, 15);
    WRAP_INIT(36, 300, 100, 31, 31,  8,  8, 15, 15);
    WRAP_INIT(39, 300, 100,  8, 12,  8, 12,  0,  0);
    WRAP_INIT(40, 300, 100,  8, 12,  8, 12,  7, 11);
    WRAP_INIT(43, 300, 100, 15, 10, 15, 10,  0,  0);
    WRAP_INIT(44, 300, 100, 15, 10, 15, 10, 14,  9);
