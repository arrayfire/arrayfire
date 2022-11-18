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

#include <algorithm>

using af::allTrue;
using af::array;
using af::constant;
using af::dtype;
using af::dtype_traits;
using af::max;
using af::min;
using af::randu;
using af::seq;
using af::span;
using std::complex;
using std::vector;

#define MINMAXOP(fn, ty)                                         \
    TEST(IndexedReduce, fn##_##ty##_0) {                         \
        SUPPORTED_TYPE_CHECK(ty);                                \
        dtype dty    = (dtype)dtype_traits<ty>::af_type;         \
        const int nx = 10;                                       \
        const int ny = 100;                                      \
        array in     = randu(nx, ny, dty);                       \
        array val, idx;                                          \
        fn(val, idx, in, 0);                                     \
                                                                 \
        ty *h_in    = in.host<ty>();                             \
        ty *h_in_st = h_in;                                      \
        uint *h_idx = idx.host<uint>();                          \
        vector<ty> gold;                                         \
        vector<ty> igold;                                        \
        gold.reserve(ny);                                        \
        igold.reserve(ny);                                       \
        for (int i = 0; i < ny; i++) {                           \
            gold.push_back(*std::fn##_element(h_in, h_in + nx)); \
            igold.push_back(h_in[h_idx[i]]);                     \
            h_in += nx;                                          \
        }                                                        \
        ASSERT_VEC_ARRAY_EQ(gold, af::dim4(1, ny), val);         \
        ASSERT_VEC_ARRAY_EQ(igold, af::dim4(1, ny), val);        \
        af_free_host(h_in_st);                                   \
        af_free_host(h_idx);                                     \
    }                                                            \
    TEST(IndexedReduce, fn##_##ty##_1) {                         \
        SUPPORTED_TYPE_CHECK(ty);                                \
        dtype dty    = (dtype)dtype_traits<ty>::af_type;         \
        const int nx = 100;                                      \
        const int ny = 100;                                      \
        array in     = randu(nx, ny, dty);                       \
        array val, idx;                                          \
        fn(val, idx, in, 1);                                     \
                                                                 \
        ty *h_in    = in.host<ty>();                             \
        ty *h_val   = val.host<ty>();                            \
        uint *h_idx = idx.host<uint>();                          \
        for (int i = 0; i < nx; i++) {                           \
            ty val = h_val[i];                                   \
            for (int j = 0; j < ny; j++) {                       \
                ty tmp = std::fn(val, h_in[j * nx + i]);         \
                ASSERT_EQ(tmp, val);                             \
            }                                                    \
            ASSERT_EQ(val, h_in[h_idx[i] * nx + i]);             \
        }                                                        \
        af_free_host(h_in);                                      \
        af_free_host(h_val);                                     \
        af_free_host(h_idx);                                     \
    }                                                            \
    TEST(IndexedReduce, fn##_##ty##_all) {                       \
        SUPPORTED_TYPE_CHECK(ty);                                \
        dtype dty     = (dtype)dtype_traits<ty>::af_type;        \
        const int num = 100000;                                  \
        array in      = randu(num, dty);                         \
        ty val;                                                  \
        uint idx;                                                \
        fn<ty>(&val, &idx, in);                                  \
        ty *h_in = in.host<ty>();                                \
        ty tmp   = *std::fn##_element(h_in, h_in + num);         \
        ASSERT_EQ(tmp, val);                                     \
        ASSERT_EQ(tmp, h_in[idx]);                               \
        af_free_host(h_in);                                      \
    }

MINMAXOP(min, float)
MINMAXOP(min, double)
MINMAXOP(min, int)
MINMAXOP(min, uint)
MINMAXOP(min, char)
MINMAXOP(min, uchar)

MINMAXOP(max, float)
MINMAXOP(max, double)
MINMAXOP(max, int)
MINMAXOP(max, uint)
MINMAXOP(max, char)
MINMAXOP(max, uchar)

TEST(IndexedReduce, MaxIndexedSmall) {
    const int num = 1000;
    const int st  = 10;
    const int en  = num - 100;
    array a       = randu(num);

    float b;
    unsigned idx;
    max<float>(&b, &idx, a(seq(st, en)));

    vector<float> ha(num);
    a.host(&ha[0]);

    float res = ha[st];
    for (int i = st; i <= en; i++) { res = std::max(res, ha[i]); }

    ASSERT_EQ(b, res);
}

TEST(IndexedReduce, MaxIndexedBig) {
    const int num = 100000;
    const int st  = 1000;
    const int en  = num - 1000;
    array a       = randu(num);

    float b;
    unsigned idx;
    max<float>(&b, &idx, a(seq(st, en)));

    vector<float> ha(num);
    a.host(&ha[0]);

    float res = ha[st];
    for (int i = st; i <= en; i++) { res = std::max(res, ha[i]); }

    ASSERT_EQ(b, res);
}

TEST(IndexedReduce, BUG_FIX_1005) {
    const int m = 64;
    const int n = 100;
    const int b = 5;

    array in = constant(0, m, n, b);
    for (int i = 0; i < b; i++) {
        array tmp         = randu(m, n);
        in(span, span, i) = tmp;

        float val0, val1;
        unsigned idx0, idx1;

        min<float>(&val0, &idx0, in(span, span, i));
        min<float>(&val1, &idx1, tmp);

        ASSERT_EQ(val0, val1);
        ASSERT_EQ(idx0, idx1);
    }
}

TEST(IndexedReduce, MinReduceDimensionHasSingleValue) {
    array data = randu(10, 10, 1);

    array mm, indx;
    min(mm, indx, data, 2);

    ASSERT_ARRAYS_EQ(data, mm);
    ASSERT_TRUE(allTrue<bool>(indx == 0));
}

TEST(IndexedReduce, MaxReduceDimensionHasSingleValue) {
    array data = randu(10, 10, 1);

    array mm, indx;
    max(mm, indx, data, 2);

    ASSERT_ARRAYS_EQ(data, mm);
    ASSERT_TRUE(allTrue<bool>(indx == 0));
}

TEST(IndexedReduce, MinNaN) {
    float test_data[] = {1.f, NAN, 5.f, 0.1f, NAN, -0.5f, NAN, 0.f};
    int rows          = 4;
    int cols          = 2;
    array a(rows, cols, test_data);

    float gold_min_val[] = {0.1f, -0.5f};
    int gold_min_idx[]   = {3, 1};

    array min_val;
    array min_idx;
    min(min_val, min_idx, a);

    vector<float> h_min_val(cols);
    min_val.host(&h_min_val[0]);

    vector<int> h_min_idx(cols);
    min_idx.host(&h_min_idx[0]);

    for (int i = 0; i < cols; i++) {
        ASSERT_FLOAT_EQ(h_min_val[i], gold_min_val[i]);
    }

    for (int i = 0; i < cols; i++) { ASSERT_EQ(h_min_idx[i], gold_min_idx[i]); }
}

TEST(IndexedReduce, MaxNaN) {
    float test_data[] = {1.f, NAN, 5.f, 0.1f, NAN, -0.5f, NAN, 0.f};
    int rows          = 4;
    int cols          = 2;
    array a(rows, cols, test_data);

    float gold_max_val[] = {5.0f, 0.f};
    int gold_max_idx[]   = {2, 3};

    array max_val;
    array max_idx;
    max(max_val, max_idx, a);

    vector<float> h_max_val(cols);
    max_val.host(&h_max_val[0]);

    vector<int> h_max_idx(cols);
    max_idx.host(&h_max_idx[0]);

    for (int i = 0; i < cols; i++) {
        ASSERT_FLOAT_EQ(h_max_val[i], gold_max_val[i]);
    }

    for (int i = 0; i < cols; i++) { ASSERT_EQ(h_max_idx[i], gold_max_idx[i]); }
}

TEST(IndexedReduce, MinCplxNaN) {
    float real_wnan_data[] = {0.005f, NAN, -6.3f, NAN,      -0.5f,
                              NAN,    NAN, 0.2f,  -1205.4f, 8.9f};

    float imag_wnan_data[] = {NAN,    NAN, -9.0f, -0.005f, -0.3f,
                              0.007f, NAN, 0.1f,  NAN,     4.5f};

    int rows = 5;
    int cols = 2;
    array real_wnan(rows, cols, real_wnan_data);
    array imag_wnan(rows, cols, imag_wnan_data);
    array a = af::complex(real_wnan, imag_wnan);

    float gold_min_real[] = {-0.5f, 0.2f};
    float gold_min_imag[] = {-0.3f, 0.1f};
    int gold_min_idx[]    = {4, 2};

    array min_val;
    array min_idx;
    af::min(min_val, min_idx, a);

    vector<complex<float>> h_min_val(cols);
    min_val.host(&h_min_val[0]);

    vector<int> h_min_idx(cols);
    min_idx.host(&h_min_idx[0]);

    for (int i = 0; i < cols; i++) {
        ASSERT_FLOAT_EQ(h_min_val[i].real(), gold_min_real[i]);
        ASSERT_FLOAT_EQ(h_min_val[i].imag(), gold_min_imag[i]);
    }

    for (int i = 0; i < cols; i++) { ASSERT_EQ(h_min_idx[i], gold_min_idx[i]); }
}

TEST(IndexedReduce, MaxCplxNaN) {
    float real_wnan_data[] = {0.005f, NAN, -6.3f, NAN,      -0.5f,
                              NAN,    NAN, 0.2f,  -1205.4f, 8.9f};

    float imag_wnan_data[] = {NAN,    NAN, -9.0f, -0.005f, -0.3f,
                              0.007f, NAN, 0.1f,  NAN,     4.5f};

    int rows = 5;
    int cols = 2;
    array real_wnan(rows, cols, real_wnan_data);
    array imag_wnan(rows, cols, imag_wnan_data);
    array a = af::complex(real_wnan, imag_wnan);

    float gold_max_real[] = {-6.3f, 8.9f};
    float gold_max_imag[] = {-9.0f, 4.5f};
    int gold_max_idx[]    = {2, 4};

    array max_val;
    array max_idx;
    af::max(max_val, max_idx, a);

    vector<complex<float>> h_max_val(cols);
    max_val.host(&h_max_val[0]);

    vector<int> h_max_idx(cols);
    max_idx.host(&h_max_idx[0]);

    for (int i = 0; i < cols; i++) {
        ASSERT_FLOAT_EQ(h_max_val[i].real(), gold_max_real[i]);
        ASSERT_FLOAT_EQ(h_max_val[i].imag(), gold_max_imag[i]);
    }

    for (int i = 0; i < cols; i++) { ASSERT_EQ(h_max_idx[i], gold_max_idx[i]); }
}

TEST(IndexedReduce, MinPreferLargerIdxIfEqual) {
    float test_data[] = {0.f, 50.f, 50.f, 0.f};
    int len           = 4;
    array a(len, test_data);

    float gold_min_val = 0.f;
    int gold_min_idx   = 3;

    array min_val;
    array min_idx;
    min(min_val, min_idx, a);

    vector<float> h_min_val(1);
    min_val.host(&h_min_val[0]);

    vector<int> h_min_idx(1);
    min_idx.host(&h_min_idx[0]);

    ASSERT_FLOAT_EQ(h_min_val[0], gold_min_val);
    ASSERT_EQ(h_min_idx[0], gold_min_idx);
}

TEST(IndexedReduce, MaxPreferSmallerIdxIfEqual) {
    float test_data[] = {0.f, 50.f, 50.f, 0.f};
    int len           = 4;
    array a(len, test_data);

    float gold_max_val = 50.f;
    int gold_max_idx   = 1;

    array max_val;
    array max_idx;
    max(max_val, max_idx, a);

    vector<float> h_max_val(1);
    max_val.host(&h_max_val[0]);

    vector<int> h_max_idx(1);
    max_idx.host(&h_max_idx[0]);

    ASSERT_FLOAT_EQ(h_max_val[0], gold_max_val);
    ASSERT_EQ(h_max_idx[0], gold_max_idx);
}

TEST(IndexedReduce, MinCplxPreferLargerIdxIfEqual) {
    float real_wnan_data[] = {0.f, 50.f, 50.f, 0.f};
    float imag_wnan_data[] = {0.f, 50.f, 50.f, 0.f};

    int len = 4;
    array real_wnan(len, real_wnan_data);
    array imag_wnan(len, imag_wnan_data);
    array a = af::complex(real_wnan, imag_wnan);

    float gold_min_real = 0.f;
    float gold_min_imag = 0.f;
    int gold_min_idx    = 3;

    array min_val;
    array min_idx;
    min(min_val, min_idx, a);

    vector<complex<float>> h_min_val(1);
    min_val.host(&h_min_val[0]);

    vector<int> h_min_idx(1);
    min_idx.host(&h_min_idx[0]);

    ASSERT_FLOAT_EQ(h_min_val[0].real(), gold_min_real);
    ASSERT_FLOAT_EQ(h_min_val[0].imag(), gold_min_imag);

    ASSERT_EQ(h_min_idx[0], gold_min_idx);
}

TEST(IndexedReduce, MaxCplxPreferSmallerIdxIfEqual) {
    float real_wnan_data[] = {0.f, 50.f, 50.f, 0.f};
    float imag_wnan_data[] = {0.f, 50.f, 50.f, 0.f};

    int len = 4;
    array real_wnan(len, real_wnan_data);
    array imag_wnan(len, imag_wnan_data);
    array a = af::complex(real_wnan, imag_wnan);

    float gold_max_real = 50.f;
    float gold_max_imag = 50.f;
    int gold_max_idx    = 1;

    array max_val;
    array max_idx;
    max(max_val, max_idx, a);

    vector<complex<float>> h_max_val(1);
    max_val.host(&h_max_val[0]);

    vector<int> h_max_idx(1);
    max_idx.host(&h_max_idx[0]);

    ASSERT_FLOAT_EQ(h_max_val[0].real(), gold_max_real);
    ASSERT_FLOAT_EQ(h_max_val[0].imag(), gold_max_imag);

    ASSERT_EQ(h_max_idx[0], gold_max_idx);
}
