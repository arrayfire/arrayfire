#include <gtest/gtest.h>
#include <arrayfire.h>
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
