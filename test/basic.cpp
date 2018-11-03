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
#include <af/data.h>
#include <testHelpers.hpp>

#include <vector>

using std::vector;
using af::array;
using af::constant;

TEST(BasicTests, constant1000x1000)
{
    if (noDoubleTests<float>()) return;

    static const int ndims = 2;
    static const int dim_size = 1000;
    dim_t d[ndims] = {dim_size, dim_size};

    double valA = 3.9;
    af_array a;
    ASSERT_SUCCESS(af_constant(&a, valA, ndims, d, f32));

    vector<float> h_a(dim_size * dim_size, 100);
    ASSERT_SUCCESS(af_get_data_ptr((void **)&h_a[0], a));

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        ASSERT_FLOAT_EQ(valA, h_a[i]);
    }

    ASSERT_SUCCESS(af_release_array(a));
}

TEST(BasicTests, constant10x10)
{
    if (noDoubleTests<float>()) return;

    static const int ndims = 2;
    static const int dim_size = 10;
    dim_t d[2] = {dim_size, dim_size};

    double valA = 3.9;
    af_array a;
    ASSERT_SUCCESS(af_constant(&a, valA, ndims, d, f32));

    vector<float> h_a(dim_size * dim_size, 0);
    ASSERT_SUCCESS(af_get_data_ptr((void **)&h_a[0], a));

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        ASSERT_FLOAT_EQ(valA, h_a[i]);
    }

    ASSERT_SUCCESS(af_release_array(a));
}

TEST(BasicTests, constant100x100)
{
    if (noDoubleTests<float>()) return;

    static const int ndims = 2;
    static const int dim_size = 100;
    dim_t d[2] = {dim_size, dim_size};

    double valA = 4.9;
    af_array a;
    ASSERT_SUCCESS(af_constant(&a, valA, ndims, d, f32));

    vector<float> h_a(dim_size * dim_size, 0);
    ASSERT_SUCCESS(af_get_data_ptr((void **)&h_a[0], a));

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        ASSERT_FLOAT_EQ(valA, h_a[i]);
    }

    ASSERT_SUCCESS(af_release_array(a));
}

//TODO: Test All The Types \o/
TEST(BasicTests, AdditionSameType)
{
    if (noDoubleTests<float>()) return;
    if (noDoubleTests<double>()) return;

    static const int ndims = 2;
    static const int dim_size = 100;
    dim_t d[ndims] = {dim_size, dim_size};

    double valA = 3.9;
    double valB = 5.7;
    double  valCf = valA + valB;

    af_array af32, bf32, cf32;
    af_array af64, bf64, cf64;

    ASSERT_SUCCESS(af_constant(&af32, valA, ndims, d, f32));
    ASSERT_SUCCESS(af_constant(&af64, valA, ndims, d, f64));

    ASSERT_SUCCESS(af_constant(&bf32, valB, ndims, d, f32));
    ASSERT_SUCCESS(af_constant(&bf64, valB, ndims, d, f64));

    ASSERT_SUCCESS(af_add(&cf32, af32, bf32, false));
    ASSERT_SUCCESS(af_add(&cf64, af64, bf64, false));

    vector<float>  h_cf32 (dim_size * dim_size);
    vector<double> h_cf64 (dim_size * dim_size);
    ASSERT_SUCCESS(af_get_data_ptr((void **)&h_cf32[0], cf32));
    ASSERT_SUCCESS(af_get_data_ptr((void **)&h_cf64[0], cf64));

    double err = 0;

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        float df = h_cf32[i] - (valCf);
        ASSERT_FLOAT_EQ(valCf,  h_cf32[i]);
        ASSERT_FLOAT_EQ(valCf,  h_cf64[i]);
        err = err + df * df;
    }
    ASSERT_NEAR(0.0f, err, 1e-8);

    ASSERT_SUCCESS(af_release_array(af32));
    ASSERT_SUCCESS(af_release_array(af64));
    ASSERT_SUCCESS(af_release_array(bf32));
    ASSERT_SUCCESS(af_release_array(bf64));
    ASSERT_SUCCESS(af_release_array(cf32));
    ASSERT_SUCCESS(af_release_array(cf64));
}

TEST(BasicTests, Additionf64f64)
{
    if (noDoubleTests<double>()) return;

    static const int ndims = 2;
    static const int dim_size = 100;
    dim_t d[ndims] = {dim_size, dim_size};

    double valA = 3.9;
    double valB = 5.7;
    double valC = valA + valB;

    af_array a, b, c;

    ASSERT_SUCCESS(af_constant(&a, valA, ndims, d, f64));
    ASSERT_SUCCESS(af_constant(&b, valB, ndims, d, f64));
    ASSERT_SUCCESS(af_add(&c, a, b, false));

    vector<double> h_c(dim_size * dim_size, 0);
    ASSERT_SUCCESS(af_get_data_ptr((void **)&h_c[0], c));

    double err = 0;

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        double df = h_c[i] - (valC);
        ASSERT_FLOAT_EQ(valA + valB, h_c[i]);
        err = err + df * df;
    }
    ASSERT_NEAR(0.0f, err, 1e-8);

    ASSERT_SUCCESS(af_release_array(a));
    ASSERT_SUCCESS(af_release_array(b));
    ASSERT_SUCCESS(af_release_array(c));

}

TEST(BasicTests, Additionf32f64)
{
    if (noDoubleTests<float>()) return;
    if (noDoubleTests<double>()) return;

    static const int ndims = 2;
    static const int dim_size = 100;
    dim_t d[ndims] = {dim_size, dim_size};

    double valA = 3.9;
    double valB = 5.7;
    double valC = valA + valB;

    af_array a, b, c;

    ASSERT_SUCCESS(af_constant(&a, valA, ndims, d, f32));
    ASSERT_SUCCESS(af_constant(&b, valB, ndims, d, f64));
    ASSERT_SUCCESS(af_add(&c, a, b, false));

    vector<double> h_c(dim_size * dim_size);
    ASSERT_SUCCESS(af_get_data_ptr((void **)&h_c[0], c));

    double err = 0;

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        double df = h_c[i] - (valC);
        ASSERT_FLOAT_EQ(valA + valB, h_c[i]);
        err = err + df * df;
    }
    ASSERT_NEAR(0.0f, err, 1e-8);

    ASSERT_SUCCESS(af_release_array(a));
    ASSERT_SUCCESS(af_release_array(b));
    ASSERT_SUCCESS(af_release_array(c));
}

TEST(BasicArrayTests, constant10x10)
{
    if (noDoubleTests<float>()) return;

    dim_t dim_size = 10;
    double valA = 3.14;
    array a = constant(valA, dim_size, dim_size, f32);

    vector<float> h_a(dim_size * dim_size, 0);
    a.host(&h_a.front());

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        ASSERT_FLOAT_EQ(valA, h_a[i]);
    }
}

////////////////////////////////////// CPP Tests //////////////////////////////////
using af::dim4;

TEST(BasicTests, constant100x100_CPP)
{
    if (noDoubleTests<float>()) return;

    static const int dim_size = 100;
    dim_t d[2] = {dim_size, dim_size};

    double valA = 4.9;
    dim4 dims(d[0], d[1]);
    array a = constant(valA, dims);

    vector<float> h_a(dim_size * dim_size, 0);
    a.host((void**)&h_a[0]);

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        ASSERT_FLOAT_EQ(valA, h_a[i]);
    }
}

//TODO: Test All The Types \o/
TEST(BasicTests, AdditionSameType_CPP)
{
    if (noDoubleTests<float>()) return;
    if (noDoubleTests<double>()) return;

    static const int dim_size = 100;
    dim_t d[2] = {dim_size, dim_size};
    dim4 dims(d[0], d[1]);

    double valA = 3.9;
    double valB = 5.7;
    double  valCf = valA + valB;

    array a32 = constant(valA, dims, f32);
    array b32 = constant(valB, dims, f32);
    array c32 = a32 + b32;

    array a64 = constant(valA, dims, f64);
    array b64 = constant(valB, dims, f64);
    array c64 = a64 + b64;

    vector<float>  h_cf32 (dim_size * dim_size);
    vector<double> h_cf64 (dim_size * dim_size);

    c32.host((void**)&h_cf32[0]);
    c64.host((void**)&h_cf64[0]);

    double err = 0;

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        float df = h_cf32[i] - (valCf);
        ASSERT_FLOAT_EQ(valCf,  h_cf32[i]);
        ASSERT_FLOAT_EQ(valCf,  h_cf64[i]);
        err = err + df * df;
    }
    ASSERT_NEAR(0.0f, err, 1e-8);
}

TEST(BasicTests, Additionf32f64_CPP)
{
    if (noDoubleTests<float>()) return;
    if (noDoubleTests<double>()) return;

    static const int dim_size = 100;
    dim_t d[2] = {dim_size, dim_size};
    dim4 dims(d[0], d[1]);

    double valA = 3.9;
    double valB = 5.7;
    double valC = valA + valB;

    array a = constant(valA, dims);
    array b = constant(valB, dims, f64);
    array c = a + b;

    vector<double> h_c(dim_size * dim_size);
    c.host((void**)&h_c[0]);

    double err = 0;

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        double df = h_c[i] - (valC);
        ASSERT_FLOAT_EQ(valA + valB, h_c[i]);
        err = err + df * df;
    }
    ASSERT_NEAR(0.0f, err, 1e-8);
}

TEST(Assert, TestEqualsCpp) {
    array gold = constant(1, 10, 10);
    array out = constant(1, 10, 10);

    // Testing this macro
    // ASSERT_ARRAYS_EQ(gold, out);
    ASSERT_TRUE(assertArrayEq("gold", "out", gold, out));
}

TEST(Assert, TestEqualsC) {
    af_array gold = 0;
    af_array out = 0;
    dim_t dims[] = {10, 10, 1, 1};
    af_constant(&gold, 1.0, 4, dims, f32);
    af_constant(&out, 1.0, 4, dims, f32);

    // Testing this macro
    // ASSERT_ARRAYS_EQ(gold, out);
    ASSERT_TRUE(assertArrayEq("gold", "out", gold, out));

    ASSERT_SUCCESS(af_release_array(out));
    ASSERT_SUCCESS(af_release_array(gold));
}

TEST(Assert, TestEqualsDiffTypes) {
    array gold = constant(1, 10, 10, f64);
    array out = constant(1, 10, 10);

    // Testing this macro
    // ASSERT_ARRAYS_EQ(gold, out);
    ASSERT_FALSE(assertArrayEq("gold", "out", gold, out));
}

TEST(Assert, TestEqualsDiffSizes) {
    array gold = constant(1, 10, 9);
    array out = constant(1, 10, 10);

    // Testing this macro
    // ASSERT_ARRAYS_EQ(gold, out);
    ASSERT_FALSE(assertArrayEq("gold", "out", gold, out));
}

TEST(Assert, TestEqualsDiffValue) {
    array gold = constant(1, 3, 3);
    array out = gold;
    out(2, 2) = 2;

    // Testing this macro
    // ASSERT_ARRAYS_EQ(gold, out);
    ASSERT_FALSE(assertArrayEq("gold", "out", gold, out));
}

TEST(Assert, TestEqualsDiffComplexValue) {
    array gold = constant(af::cfloat(3.1f, 3.1f), 3, 3, c32);
    array out = gold;
    out(2, 2) = 2.2;

    // Testing this macro
    // ASSERT_ARRAYS_EQ(gold, out);
    ASSERT_FALSE(assertArrayEq("gold", "out", gold, out));
}

TEST(Assert, TestVectorEquals) {
    array out = constant(3.1f, 3, 3);

    vector<float> gold(out.elements());
    dim4 goldDims(3, 3);
    fill(gold.begin(), gold.end(), 3.1f);

    // Testing this macro
    // ASSERT_VEC_ARRAY_EQ(gold, goldDims, out);
    ASSERT_TRUE(assertArrayEq("gold", "goldDims", "out",
                              gold, goldDims, out));
}

TEST(Assert, TestVectorDiffVecType) {
    array out = constant(3.1f, 3, 3);

    vector<int> gold(out.elements());
    dim4 goldDims(3, 3);
    fill(gold.begin(), gold.end(), 3.1f);

    // Testing this macro
    // ASSERT_VEC_ARRAY_EQ(gold, goldDims, out);
    ASSERT_FALSE(assertArrayEq("gold", "goldDims", "out",
                              gold, goldDims, out));
}

TEST(Assert, TestVectorDiffGoldSizeDims) {
    array out = constant(3.1f, 3, 3);

    vector<float> gold(3 * 3);
    dim4 goldDims(3, 2);
    fill(gold.begin(), gold.end(), 3.1f);

    // Testing this macro
    // ASSERT_VEC_ARRAY_EQ(gold, goldDims, out);
    ASSERT_FALSE(assertArrayEq("gold", "goldDims", "out",
                               gold, goldDims, out));
}

TEST(Assert, TestVectorDiffOutSizeGoldSize) {
    array out = constant(3.1f, 3, 3);

    vector<float> gold(3 * 2);
    dim4 goldDims(3, 2);
    fill(gold.begin(), gold.end(), 3.1f);

    // Testing this macro
    // ASSERT_VEC_ARRAY_EQ(gold, goldDims, out);
    ASSERT_FALSE(assertArrayEq("gold", "goldDims", "out",
                               gold, goldDims, out));
}

TEST(Assert, TestVectorDiffDim4) {
    array A = constant(3.1f, 3, 3);
    vector<float> hA(A.elements());
    dim4 adims(3, 2);
    fill(hA.begin(), hA.end(), 3.1f);

    // Testing this macro
    // ASSERT_ARRAYS_EQ(A, B);
    ASSERT_FALSE(assertArrayEq("hA", "adims", "A", hA, adims, A));
}

TEST(Assert, TestVectorDiffVecSize) {
    array A = constant(3.1f, 3, 3);
    vector<float> hA(A.elements()-1);
    dim4 adims(3, 3);
    fill(hA.begin(), hA.end(), 3.1f);

    // Testing this macro
    // ASSERT_ARRAYS_EQ(A, B);
    ASSERT_FALSE(assertArrayEq("hA", "adims", "A", hA, adims, A));
}

TEST(Assert, TestArraysNear) {
    array gold = constant(1, 3, 3);
    array out = constant(1, 3, 3);
    gold(2, 2) = 2.2345;
    out(2, 2) = 2.2445;
    float maxDiff = 0.001;

    // Testing this macro
    // ASSERT_ARRAYS_NEAR(gold, out, maxDiff);
    ASSERT_FALSE(assertArrayEq("gold", "out", gold, out, maxDiff));
}

TEST(Assert, TestVecArrayNear) {
    ASSERT_TRUE(true);
}
