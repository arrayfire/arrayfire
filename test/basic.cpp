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
#include <vector>

using namespace std;

TEST(BasicTests, constant1000x1000)
{
    static const int ndims = 2;
    static const int dim_size = 1000;
    dim_type d[ndims] = {dim_size, dim_size};

    double valA = 3.9;
    af_array a;
    ASSERT_EQ(AF_SUCCESS, af_constant(&a, valA, ndims, d, f32));

    vector<float> h_a(dim_size * dim_size, 100);
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void **)&h_a[0], a));

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        ASSERT_FLOAT_EQ(valA, h_a[i]);
    }
}

TEST(BasicTests, constant10x10)
{
    static const int ndims = 2;
    static const int dim_size = 10;
    dim_type d[2] = {dim_size, dim_size};

    double valA = 3.9;
    af_array a;
    ASSERT_EQ(AF_SUCCESS, af_constant(&a, valA, ndims, d, f32));

    vector<float> h_a(dim_size * dim_size, 0);
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void **)&h_a[0], a));

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        ASSERT_FLOAT_EQ(valA, h_a[i]);
    }
}

TEST(BasicTests, constant100x100)
{
    static const int ndims = 2;
    static const int dim_size = 100;
    dim_type d[2] = {dim_size, dim_size};

    double valA = 4.9;
    af_array a;
    ASSERT_EQ(AF_SUCCESS, af_constant(&a, valA, ndims, d, f32));

    vector<float> h_a(dim_size * dim_size, 0);
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void **)&h_a[0], a));

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        ASSERT_FLOAT_EQ(valA, h_a[i]);
    }
}

//TODO: Test All The Types \o/
TEST(BasicTests, AdditionSameType)
{
    static const int ndims = 2;
    static const int dim_size = 100;
    dim_type d[ndims] = {dim_size, dim_size};

    double valA = 3.9;
    double valB = 5.7;
    double  valCf = valA + valB;

    af_array af32, bf32, cf32;
    af_array af64, bf64, cf64;

    ASSERT_EQ(AF_SUCCESS, af_constant(&af32, valA, ndims, d, f32));
    ASSERT_EQ(AF_SUCCESS, af_constant(&af64, valA, ndims, d, f64));

    ASSERT_EQ(AF_SUCCESS, af_constant(&bf32, valB, ndims, d, f32));
    ASSERT_EQ(AF_SUCCESS, af_constant(&bf64, valB, ndims, d, f64));

    ASSERT_EQ(AF_SUCCESS, af_add(&cf32, af32, bf32));
    ASSERT_EQ(AF_SUCCESS, af_add(&cf64, af64, bf64));

    vector<float>  h_cf32 (dim_size * dim_size);
    vector<double> h_cf64 (dim_size * dim_size);
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void **)&h_cf32[0], cf32));
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void **)&h_cf64[0], cf64));

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

TEST(BasicTests, Additionf64f64)
{
    static const int ndims = 2;
    static const int dim_size = 100;
    dim_type d[ndims] = {dim_size, dim_size};

    double valA = 3.9;
    double valB = 5.7;
    double valC = valA + valB;

    af_array a, b, c;

    ASSERT_EQ(AF_SUCCESS, af_constant(&a, valA, ndims, d, f64));
    ASSERT_EQ(AF_SUCCESS, af_constant(&b, valB, ndims, d, f64));
    ASSERT_EQ(AF_SUCCESS, af_add(&c, a, b));

    vector<double> h_c(dim_size * dim_size, 0);
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void **)&h_c[0], c));

    double err = 0;

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        double df = h_c[i] - (valC);
        ASSERT_FLOAT_EQ(valA + valB, h_c[i]);
        err = err + df * df;
    }
    ASSERT_NEAR(0.0f, err, 1e-8);
}

TEST(BasicTests, Additionf32f64)
{
    static const int ndims = 2;
    static const int dim_size = 100;
    dim_type d[ndims] = {dim_size, dim_size};

    double valA = 3.9;
    double valB = 5.7;
    double valC = valA + valB;

    af_array a, b, c;

    ASSERT_EQ(AF_SUCCESS, af_constant(&a, valA, ndims, d, f32));
    ASSERT_EQ(AF_SUCCESS, af_constant(&b, valB, ndims, d, f64));
    ASSERT_EQ(AF_SUCCESS, af_add(&c, a, b));

    vector<double> h_c(dim_size * dim_size);
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void **)&h_c[0], c));

    double err = 0;

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        double df = h_c[i] - (valC);
        ASSERT_FLOAT_EQ(valA + valB, h_c[i]);
        err = err + df * df;
    }
    ASSERT_NEAR(0.0f, err, 1e-8);
}

TEST(BasicArrayTests, constant10x10)
{
    dim_type dim_size = 10;
    double valA = 3.14;
    af::array a = af::constant(valA, dim_size, dim_size, f32);

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
    static const int dim_size = 100;
    dim_type d[2] = {dim_size, dim_size};

    double valA = 4.9;
    dim4 dims(d[0], d[1]);
    af::array a = constant(valA, dims);

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
    static const int dim_size = 100;
    dim_type d[2] = {dim_size, dim_size};
    dim4 dims(d[0], d[1]);

    double valA = 3.9;
    double valB = 5.7;
    double  valCf = valA + valB;

    af::array a32 = constant(valA, dims, f32);
    af::array b32 = constant(valB, dims, f32);
    af::array c32 = a32 + b32;

    af::array a64 = constant(valA, dims, f64);
    af::array b64 = constant(valB, dims, f64);
    af::array c64 = a64 + b64;

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
    static const int dim_size = 100;
    dim_type d[2] = {dim_size, dim_size};
    dim4 dims(d[0], d[1]);

    double valA = 3.9;
    double valB = 5.7;
    double valC = valA + valB;

    af::array a = constant(valA, dims);
    af::array b = constant(valB, dims, f64);
    af::array c = a + b;

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
