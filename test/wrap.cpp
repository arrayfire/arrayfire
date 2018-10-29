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
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <algorithm>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using af::allTrue;
using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::randu;
using af::range;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Wrap : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned int,
                         intl, uintl, char, unsigned char, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_CASE(Wrap, TestTypes);

template<typename T>
inline double get_val(T val) {
    return val;
}

template<>
inline double get_val<cfloat>(cfloat val) {
    return abs(val);
}

template<>
inline double get_val<cdouble>(cdouble val) {
    return abs(val);
}

template<>
inline double get_val<unsigned char>(unsigned char val) {
    return ((int)(val)) % 256;
}

template<>
inline double get_val<char>(char val) {
    return (val != 0);
}

template<typename T>
void wrapTest(const dim_t ix, const dim_t iy, const dim_t wx, const dim_t wy,
              const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
              bool cond) {
    SUPPORTED_TYPE_CHECK(T);

    const int nc = 1;

    int lim = std::max((dim_t)2, (dim_t)(250) / (wx * wy));

    dtype ty = (dtype)dtype_traits<T>::af_type;
    array in = round(lim * randu(ix, iy, nc, f32)).as(ty);

    vector<T> h_in(in.elements());
    in.host(&h_in[0]);

    vector<int> h_factor(ix * iy);

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

    array factor(ix, iy, &h_factor[0]);

    array in_dim  = unwrap(in, wx, wy, sx, sy, px, py, cond);
    array res_dim = wrap(in_dim, ix, iy, wx, wy, sx, sy, px, py, cond);

    ASSERT_EQ(in.elements(), res_dim.elements());

    vector<T> h_res(ix * iy);
    res_dim.host(&h_res[0]);

    for (int n = 0; n < nc; n++) {
        T *iptr = &h_in[n * ix * iy];
        T *rptr = &h_res[n * ix * iy];

        for (int y = 0; y < iy; y++) {
            for (int x = 0; x < ix; x++) {
                // FIXME: Use a better test
                T ival     = iptr[y * ix + x];
                T rval     = rptr[y * ix + x];
                int factor = h_factor[y * ix + x];

                if (get_val(ival) == 0) continue;

                ASSERT_NEAR(get_val<T>(ival * factor), get_val<T>(rval), 1E-5)
                    << "at " << x << "," << y << " for cond  == " << cond
                    << endl;
            }
        }
    }
}

#define WRAP_INIT(desc, ix, iy, wx, wy, sx, sy, px, py)             \
    TYPED_TEST(Wrap, Col##desc) {                                   \
        wrapTest<TypeParam>(ix, iy, wx, wy, sx, sy, px, py, true);  \
    }                                                               \
    TYPED_TEST(Wrap, Row##desc) {                                   \
        wrapTest<TypeParam>(ix, iy, wx, wy, sx, sy, px, py, false); \
    }

    WRAP_INIT(00, 300, 100,  3,  3,  1,  1,  0,  0);
    // WRAP_INIT(01, 300, 100,  3,  3,  1,  1,  1,  1);
    // WRAP_INIT(03, 300, 100,  3,  3,  2,  2,  0,  0);
    // WRAP_INIT(04, 300, 100,  3,  3,  2,  2,  1,  1);
    // WRAP_INIT(05, 300, 100,  3,  3,  2,  2,  2,  2);
    // WRAP_INIT(06, 300, 100,  3,  3,  3,  3,  0,  0);
    // WRAP_INIT(07, 300, 100,  3,  3,  3,  3,  1,  1);
    // WRAP_INIT(08, 300, 100,  3,  3,  3,  3,  2,  2);
    // WRAP_INIT(09, 300, 100,  4,  4,  1,  1,  0,  0);
    // WRAP_INIT(13, 300, 100,  4,  4,  2,  2,  0,  0);
    // WRAP_INIT(14, 300, 100,  4,  4,  2,  2,  1,  1);
    // WRAP_INIT(15, 300, 100,  4,  4,  2,  2,  2,  2);
    // WRAP_INIT(16, 300, 100,  4,  4,  2,  2,  3,  3);
    // WRAP_INIT(17, 300, 100,  4,  4,  4,  4,  0,  0);
    // WRAP_INIT(18, 300, 100,  4,  4,  4,  4,  1,  1);
    // WRAP_INIT(19, 300, 100,  4,  4,  4,  4,  2,  2);
    // WRAP_INIT(27, 300, 100,  8,  8,  8,  8,  0,  0);
    // WRAP_INIT(28, 300, 100,  8,  8,  8,  8,  7,  7);
    // WRAP_INIT(31, 300, 100, 12, 12, 12, 12,  0,  0);
    // WRAP_INIT(32, 300, 100, 12, 12, 12, 12,  2,  2);
    // WRAP_INIT(35, 300, 100, 16, 16, 16, 16, 15, 15);
    // WRAP_INIT(36, 300, 100, 31, 31,  8,  8, 15, 15);
    // WRAP_INIT(39, 300, 100,  8, 12,  8, 12,  0,  0);
    // WRAP_INIT(40, 300, 100,  8, 12,  8, 12,  7, 11);
    // WRAP_INIT(43, 300, 100, 15, 10, 15, 10,  0,  0);
    // WRAP_INIT(44, 300, 100, 15, 10, 15, 10, 14,  9);

TEST(Wrap, MaxDim)
{
    const size_t largeDim = 65535 + 1;
    array input           = range(5, 5, 1, largeDim);

    const unsigned wx = 5;
    const unsigned wy = 5;
    const unsigned sx = 5;
    const unsigned sy = 5;
    const unsigned px = 0;
    const unsigned py = 0;

    array unwrapped = unwrap(input, wx, wy, sx, sy, px, py);
    array output    = wrap(unwrapped, 5, 5, wx, wy, sx, sy, px, py);

    ASSERT_ARRAYS_EQ(output, input);
}

TEST(Wrap, DocSnippet) {
    //! [ex_wrap_1]
    float hA[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    array A(dim4(3, 3), hA);
    //  1.     4.     7.
    //  2.     5.     8.
    //  3.     6.     9.

    array A_unwrapped = unwrap(A, 2, 2,  // window size
                               2, 2,     // stride (distinct)
                               1, 1);    // padding
    //  0.     0.     0.     5.
    //  0.     0.     4.     6.
    //  0.     2.     0.     8.
    //  1.     3.     7.     9.

    array A_wrapped = wrap(A_unwrapped, 3, 3,  // A's size
                           2, 2,               // window size
                           2, 2,               // stride (distinct)
                           1, 1);              // padding
    //  1.     4.     7.
    //  2.     5.     8.
    //  3.     6.     9.
    //! [ex_wrap_1]

    ASSERT_ARRAYS_EQ(A, A_wrapped);

    //! [ex_wrap_2]
    float hB[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    array B(dim4(3, 3), hB);
    //  1.     1.     1.
    //  1.     1.     1.
    //  1.     1.     1.
    array B_unwrapped = unwrap(B, 2, 2,  // window size
                               1, 1);    // stride (sliding)
    //  1.     1.     1.     1.
    //  1.     1.     1.     1.
    //  1.     1.     1.     1.
    //  1.     1.     1.     1.
    array B_wrapped = wrap(B_unwrapped, 3, 3,  // B's size
                           2, 2,               // window size
                           1, 1);              // stride (sliding)
    //  1.     2.     1.
    //  2.     4.     2.
    //  1.     2.     1.
    //! [ex_wrap_2]

    float gold_hB_wrapped[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    array gold_B_wrapped(dim4(3, 3), gold_hB_wrapped);
    ASSERT_ARRAYS_EQ(gold_B_wrapped, B_wrapped);
}

TEST(Wrap, NullInput)
{
    float h_in[] = { 10, 20, 30, 40,
                     20, 30, 40, 50,
                     30, 40, 50, 60,
                     40, 50, 60, 70 };
    af_array in = 0;
    dim_t in_dims[4] = { dim_t(4), dim_t(4), 1, 1};
    ASSERT_SUCCESS(af_create_array(&in, &h_in[0], 2, in_dims, f32));


    dim_t window_size = 2;
    dim_t stride_size = 2;
    dim_t padding_size = 0;
    af_array unwrapped = 0;
    ASSERT_SUCCESS(af_unwrap(&unwrapped, in,
                             window_size,  window_size,
                             stride_size,  stride_size,
                             padding_size, padding_size,
                             true));


    af_array out = 0;
    ASSERT_SUCCESS(af_wrap(&out, unwrapped,
                           in_dims[0], in_dims[1],
                           window_size,  window_size,
                           stride_size,  stride_size,
                           padding_size, padding_size,
                           true));
    ASSERT_FALSE(out == 0);

    if (in != 0) af_release_array(in);
    if (unwrapped != 0) af_release_array(unwrapped);
    if (out != 0) af_release_array(out);
}

TEST(Wrap, EmptyInput)
{
    float h_in[] = { 10, 20, 20, 30,
                     30, 40, 40, 50,
                     30, 40, 40, 50,
                     50, 60, 60, 70 };

    // NOTE: `h_in` was obtained calling `unwrap(image, wrap = 2, stride = 2, padding = 0)`
    // where image looks like the following:
    //
    // [4 4 1 1]
    //     10.0000    20.0000    30.0000    40.0000
    //     20.0000    30.0000    40.0000    50.0000
    //     30.0000    40.0000    50.0000    60.0000
    //     40.0000    50.0000    60.0000    70.0000
    // win = 2, stride = 2, padding = 0

    af_array in = 0;
    dim_t in_dims[4] = { dim_t(4), dim_t(4), 1, 1 };
    ASSERT_SUCCESS(af_create_array(&in, &h_in[0], 2, in_dims, f32));
    cout << "\nin:\n\n";
    af_print_array(in);

    af_array wrapped = 0;
    ASSERT_SUCCESS(af_create_handle(&wrapped, 2, in_dims, f32));
    {
        dim_t d0, d1, d2, d3;
        af_get_dims(&d0, &d1, &d2, &d3, wrapped);
        ASSERT_EQ(d0, in_dims[0]);
        ASSERT_EQ(d1, in_dims[1]);
        ASSERT_EQ(d2, 1);
        ASSERT_EQ(d3, 1);
    }

    // af_array wrapped_copy = wrapped;
    const dim_t window_size = 2;
    const dim_t stride_size = 2;
    const dim_t padding_size = 0;
    ASSERT_SUCCESS(af_wrap(&wrapped, in,
                           in_dims[0], in_dims[1],
                           window_size, window_size,
                           stride_size, stride_size,
                           padding_size, padding_size,
                           true));
    cout << "\nwrapped:\n\n";
    af_print_array(wrapped);


    af_array gold = 0;
    float h_gold[] = { 10, 20, 30, 40,
                       20, 30, 40, 50,
                       30, 40, 50, 60,
                       40, 50, 60, 70 };
    dim_t gold_dims[4] = { dim_t(4), dim_t(4), 1, 1 };
    ASSERT_SUCCESS(af_create_array(&gold, &h_gold[0], 2, gold_dims, f32));

    ASSERT_FALSE(wrapped == 0);

    // \TODO why are `wrapped` and `gold` mismatching? why does
    // `wrapped` have incorrect values greater than any of input
    // values?
    ASSERT_ARRAYS_EQ(wrapped, gold);

    // wrapped
    // [4 4 1 1]
    //    20.0000    40.0000    50.0000    70.0000
    //    50.0000    70.0000    80.0000   100.0000
    //    60.0000    80.0000    90.0000   110.0000
    //    90.0000   110.0000   120.0000   140.0000

    // gold
    // [4 4 1 1]
    //    10.0000    20.0000    30.0000    40.0000
    //    20.0000    30.0000    40.0000    50.0000
    //    30.0000    40.0000    50.0000    60.0000
    //    40.0000    50.0000    60.0000    70.0000




    // ASSERT_EQ(wrapped_copy, wrapped);
    // ASSERT_ARRAYS_EQ(wrapped_copy, wrapped);

    if (in != 0) af_release_array(in);
    if (wrapped != 0) af_release_array(wrapped);
    if (gold != 0) af_release_array(gold);
}

// TEST(Wrap, NonEmptyInput)
// {
//     float h_in[] = { 10, 20, 30, 40,
//                      20, 30, 40, 50,
//                      30, 40, 50, 60,
//                      40, 50, 60, 70 };
//     af_array in = 0;
//     dim_t in_dims[4] = { dim_t(4), dim_t(4), 1, 1 };
//     ASSERT_SUCCESS(af_create_array(&in, &h_in[0], 2, in_dims, f32));

//     const dim_t window_size = 2;
//     const dim_t stride_size = 2;
//     const dim_t padding_size = 0;
//     af_array unwrapped = 0;
//     ASSERT_SUCCESS(af_unwrap(&unwrapped, in,
//                              window_size,  window_size,
//                              stride_size,  stride_size,
//                              padding_size, padding_size,
//                              true));
//     af_print_array(unwrapped);

//     af_array out = 0;
//     dim_t h_out_dims[2] = {4, 4};
//     ASSERT_SUCCESS(af_randu(&out, 2, h_out_dims, f32));

//     // \TODO indexing

//     ASSERT_SUCCESS(af_wrap(&out, unwrapped,
//                            in_dims[0], in_dims[1],
//                            window_size,  window_size,
//                            stride_size,  stride_size,
//                            padding_size, padding_size,
//                            true));

//     ASSERT_FALSE(out == 0);
//     ASSERT_ARRAYS_EQ(out, in);

//     if (in != 0) af_release_array(in);
//     if (unwrapped != 0) af_release_array(unwrapped);
//     if (out != 0) af_release_array(out);
// }
