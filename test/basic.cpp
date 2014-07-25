

#include "gtest/gtest.h"
#include "arrayfire.h"

TEST(BasicTests, constant1000x1000)
{
    static const int ndims = 2;
    static const int dim_size = 1000;
    long d[ndims] = {dim_size, dim_size};

    double valA = 3.9;
    af_array a;
    float *h_a;
    EXPECT_EQ(AF_SUCCESS, af_constant(&a, valA, ndims, d, f32));
    EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **)&h_a, a));
    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        EXPECT_FLOAT_EQ(valA, h_a[i]);
    }
}

TEST(BasicTests, constant10x10)
{
    static const int ndims = 2;
    static const int dim_size = 10;
    long d[2] = {dim_size, dim_size};

    double valA = 3.9;
    af_array a;
    float *h_a;
    EXPECT_EQ(AF_SUCCESS, af_constant(&a, valA, ndims, d, f32));
    EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **)&h_a, a));
    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        EXPECT_FLOAT_EQ(valA, h_a[i]);
    }
}

TEST(BasicTests, constant100x100)
{
    static const int ndims = 2;
    static const int dim_size = 100;
    long d[2] = {dim_size, dim_size};

    double valA = 4.9;
    af_array a;
    float *h_a;
    EXPECT_EQ(AF_SUCCESS, af_constant(&a, valA, ndims, d, f32));
    EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **)&h_a, a));
    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        EXPECT_FLOAT_EQ(valA, h_a[i]);
    }
}

//TODO: Test All The Types \o/
TEST(BasicTests, AdditionSameType)
{
    static const int ndims = 2;
    static const int dim_size = 100;
    long d[ndims] = {dim_size, dim_size};

    double valA = 3.9;
    double valB = 5.7;
    double  valCf = valA + valB;
    //int     valC = (int)valA + (int)valB;

    af_array af32, bf32, cf32;
    af_array af64, bf64, cf64;
    //af_array as32, bs32, cs32;
    //af_array au32, bu32, cu32;
    //af_array au8, bu8, cu8;

    EXPECT_EQ(AF_SUCCESS, af_constant(&af32, valA, ndims, d, f32));
    EXPECT_EQ(AF_SUCCESS, af_constant(&af64, valA, ndims, d, f64));
    //EXPECT_EQ(AF_SUCCESS, af_constant(&as32, valA, ndims, d, s32));
    //EXPECT_EQ(AF_SUCCESS, af_constant(&au32, valA, ndims, d, u32));
    //EXPECT_EQ(AF_SUCCESS, af_constant(&au8, valA, ndims, d, u8));

    EXPECT_EQ(AF_SUCCESS, af_constant(&bf32, valB, ndims, d, f32));
    EXPECT_EQ(AF_SUCCESS, af_constant(&bf64, valB, ndims, d, f64));
    //EXPECT_EQ(AF_SUCCESS, af_constant(&bs32, valB, ndims, d, s32));
    //EXPECT_EQ(AF_SUCCESS, af_constant(&bu32, valB, ndims, d, u32));
    //EXPECT_EQ(AF_SUCCESS, af_constant(&bu8, valB, ndims, d, u8));

    EXPECT_EQ(AF_SUCCESS, af_add(&cf32, af32, bf32));
    EXPECT_EQ(AF_SUCCESS, af_add(&cf64, af64, bf64));
    //EXPECT_EQ(AF_SUCCESS, af_add(&cs32, as32, bs32));
    //EXPECT_EQ(AF_SUCCESS, af_add(&cu32, au32, bu32));
    //EXPECT_EQ(AF_SUCCESS, af_add(&cu8, au8, bu8)); //TODO: not working

    float*          h_cf32 = NULL;
    double*         h_cf64 = NULL;
    //int*            h_cs32 = NULL;
    //unsigned*       h_cu32 = NULL;
    //unsigned char*  h_cu8 = NULL;
    EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **)&h_cf32, cf32));
    EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **)&h_cf64, cf64));
    //EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **)&h_cs32, cs32));
    //EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **)&h_cu32, cu32));
    //EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **)&h_cu8,  cu8));

    double err = 0;

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        float df = h_cf32[i] - (valCf);
        EXPECT_FLOAT_EQ(valCf,  h_cf32[i]);
        EXPECT_FLOAT_EQ(valCf,  h_cf64[i]);
        //EXPECT_EQ(valC,         h_cs32[i]);
        //EXPECT_EQ(valC,         h_cu32[i]);
        //EXPECT_EQ(valC,         (int)h_cu8[i]);
        err = err + df * df;
    }
    EXPECT_NEAR(0.0f, err, 1e-8);
}

//TEST(BasicTests, Subtractionf32f32)
//{
//    static const int ndims = 2;
//    static const int dim_size = 100;
//    unsigned d[ndims] = {dim_size, dim_size};
//
//    double valA = 3.9;
//    double valB = 5.7;
//    double valC = valA - valB;
//
//    af_array af32, bf32, cf32;
//    //af_array as32, bs32, cs32;
//    //af_array au32, bu32, cu32;
//    //af_array au8, bu8, cu32;
//    //af_array af64, bf64, cf64;
//
//    EXPECT_EQ(AF_SUCCESS, af_constant(&af32, valA, ndims, d, f32));
//    EXPECT_EQ(AF_SUCCESS, af_constant(&bf32, valB, ndims, d, f32));
//    EXPECT_EQ(AF_SUCCESS, af_sub(&cf32, af32, bf32));
//
//    float* h_c = NULL;
//    EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **)&h_c, cf32));
//
//    double err = 0;
//
//    size_t elements = dim_size * dim_size;
//    for(int i = 0; i < elements; i++) {
//        float df = h_c[i] - (valC);
//        EXPECT_FLOAT_EQ(valA - valB, h_c[i]);
//        err = err + df * df;
//    }
//    EXPECT_NEAR(0.0f, err, 1e-8);
//}

TEST(BasicTests, Additionf32f64)
{
    static const int ndims = 2;
    static const int dim_size = 100;
    long d[ndims] = {dim_size, dim_size};

    double valA = 3.9;
    double valB = 5.7;
    double valC = valA + valB;

    af_array a, b, c;

    EXPECT_EQ(AF_SUCCESS, af_constant(&a, valA, ndims, d, f32));
    EXPECT_EQ(AF_SUCCESS, af_constant(&b, valB, ndims, d, f64));
    EXPECT_EQ(AF_SUCCESS, af_add(&c, a, b));

    double* h_c = NULL;
    EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **)&h_c, c));

    double err = 0;

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        double df = h_c[i] - (valC);
        EXPECT_FLOAT_EQ(valA + valB, h_c[i]);
        err = err + df * df;
    }
    EXPECT_NEAR(0.0f, err, 1e-8);
}

TEST(BasicTests, Additionf64f64)
{
    static const int ndims = 2;
    static const int dim_size = 100;
    long d[ndims] = {dim_size, dim_size};

    double valA = 3.9;
    double valB = 5.7;
    double valC = valA + valB;

    af_array a, b, c;

    EXPECT_EQ(AF_SUCCESS, af_constant(&a, valA, ndims, d, f64));
    EXPECT_EQ(AF_SUCCESS, af_constant(&b, valB, ndims, d, f64));
    EXPECT_EQ(AF_SUCCESS, af_add(&c, a, b));

    double* h_c = NULL;
    EXPECT_EQ(AF_SUCCESS, af_host_ptr((void **)&h_c, c));

    double err = 0;

    size_t elements = dim_size * dim_size;
    for(size_t i = 0; i < elements; i++) {
        double df = h_c[i] - (valC);
        EXPECT_FLOAT_EQ(valA + valB, h_c[i]);
        err = err + df * df;
    }
    EXPECT_NEAR(0.0f, err, 1e-8);
}
