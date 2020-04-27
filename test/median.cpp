/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/algorithm.h>
#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>
#include <af/device.h>
#include <af/random.h>
#include <af/statistics.h>

using af::array;
using af::dtype;
using af::dtype_traits;
using af::median;
using af::randu;
using af::seq;
using af::span;
using af::sum;
using std::vector;

template<typename Ti>
array generateArray(int nx, int ny, int nz, int nw) {
    array a = randu(nx, ny, nz, nw, (dtype)dtype_traits<Ti>::af_type);
    return a;
}

template<>
array generateArray<int>(int nx, int ny, int nz, int nw) {
    array a = (randu(nx, ny, nz, nw, (dtype)dtype_traits<float>::af_type) * 1e6)
                  .as(s32);
    return a;
}

template<>
array generateArray<unsigned int>(int nx, int ny, int nz, int nw) {
    array a = (randu(nx, ny, nz, nw, (dtype)dtype_traits<float>::af_type) * 1e6)
                  .as(u32);
    return a;
}

template<typename To, typename Ti>
void median_flat(int nx, int ny = 1, int nz = 1, int nw = 1) {
    SUPPORTED_TYPE_CHECK(Ti);
    array a = generateArray<Ti>(nx, ny, nz, nw);

    // Verification
    array sa  = sort(flat(a));
    dim_t mid = (sa.dims(0) + 1) / 2;

    To verify;

    To *h_sa = sa.as((af_dtype)dtype_traits<To>::af_type).host<To>();
    if (sa.dims(0) % 2 == 1) {
        verify = h_sa[mid - 1];
    } else {
        verify = (h_sa[mid - 1] + h_sa[mid]) / (To)2;
    }

    // Test Part
    To val = median<To>(a);

    ASSERT_EQ(verify, val);

    af_free_host(h_sa);
}

template<typename To, typename Ti, int dim>
void median_test(int nx, int ny = 1, int nz = 1, int nw = 1) {
    SUPPORTED_TYPE_CHECK(Ti);

    array a = generateArray<Ti>(nx, ny, nz, nw);

    // If selected dim is higher than input ndims, then return
    if (dim >= a.dims().ndims()) return;

    array verify;

    // Verification
    array sa = sort(a, dim);

    double mid  = (a.dims(dim) + 1) / 2;
    seq mSeq[4] = {span, span, span, span};
    mSeq[dim]   = seq(mid, mid, 1.0);

    if (sa.dims(dim) % 2 == 1) {
        mSeq[dim] = mSeq[dim] - 1.0;
        verify    = sa(mSeq[0], mSeq[1], mSeq[2], mSeq[3]);
    } else {
        dim_t sdim[4] = {0};
        sdim[dim]     = 1;
        sa            = sa.as((af_dtype)dtype_traits<To>::af_type);
        array sas     = shift(sa, sdim[0], sdim[1], sdim[2], sdim[3]);
        verify        = ((sa + sas) / 2)(mSeq[0], mSeq[1], mSeq[2], mSeq[3]);
    }

    // Test Part
    array out = median(a, dim);

    ASSERT_EQ(out.dims() == verify.dims(), true);
    ASSERT_NEAR(0, sum<double>(abs(out - verify)), 1e-5);
}

#define MEDIAN_FLAT(To, Ti)                                                    \
    TEST(MedianFlat, Ti##_flat_even) { median_flat<To, Ti>(1000); }            \
    TEST(MedianFlat, Ti##_flat_odd) { median_flat<To, Ti>(783); }              \
    TEST(MedianFlat, Ti##_flat_multi_even) { median_flat<To, Ti>(24, 11, 3); } \
    TEST(MedianFlat, Ti##_flat_multi_odd) { median_flat<To, Ti>(15, 21, 7); }

MEDIAN_FLAT(float, float)
MEDIAN_FLAT(float, int)
MEDIAN_FLAT(float, uint)
MEDIAN_FLAT(float, uchar)
MEDIAN_FLAT(float, short)
MEDIAN_FLAT(float, ushort)
MEDIAN_FLAT(double, double)

#define MEDIAN_TEST(To, Ti, dim)                                               \
    TEST(Median, Ti##_1D_##dim##_even) { median_test<To, Ti, dim>(1000); }     \
    TEST(Median, Ti##_2D_##dim##_even) { median_test<To, Ti, dim>(1000, 25); } \
    TEST(Median, Ti##_3D_##dim##_even) {                                       \
        median_test<To, Ti, dim>(100, 25, 4);                                  \
    }                                                                          \
    TEST(Median, Ti##_4D_##dim##_even) {                                       \
        median_test<To, Ti, dim>(100, 25, 2, 2);                               \
    }                                                                          \
    TEST(Median, Ti##_1D_##dim##_odd) { median_test<To, Ti, dim>(783); }       \
    TEST(Median, Ti##_2D_##dim##_odd) { median_test<To, Ti, dim>(783, 25); }   \
    TEST(Median, Ti##_3D_##dim##_odd) {                                        \
        median_test<To, Ti, dim>(123, 25, 3);                                  \
    }                                                                          \
    TEST(Median, Ti##_4D_##dim##_odd) {                                        \
        median_test<To, Ti, dim>(123, 25, 3, 3);                               \
    }

#define MEDIAN(To, Ti)     \
    MEDIAN_TEST(To, Ti, 0) \
    MEDIAN_TEST(To, Ti, 1) \
    MEDIAN_TEST(To, Ti, 2) \
    MEDIAN_TEST(To, Ti, 3)

MEDIAN(float, float)
MEDIAN(float, int)
MEDIAN(float, uint)
MEDIAN(float, uchar)
MEDIAN(float, short)
MEDIAN(float, ushort)
MEDIAN(double, double)

TEST(Median, OneElement) {
    af::array in = randu(1, f32);

    af::array out = median(in);
    ASSERT_ARRAYS_EQ(in, out);
}

TEST(Median, TwoElements) {
    af::array in = randu(2, f32);

    af::array out  = median(in);
    af::array gold = mean(in);
    ASSERT_ARRAYS_EQ(gold, out);
}
