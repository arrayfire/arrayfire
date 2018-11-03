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
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype;
using af::dtype_traits;

template<typename T>
class Random : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble, int, unsigned, intl, uintl, unsigned char> TestTypes;

// register the type list
TYPED_TEST_CASE(Random, TestTypes);

template<typename T>
class Random_norm : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

template<typename T>
class RandomEngine : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

template<typename T>
class RandomEngineSeed : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

template<typename T>
class RandomSeed : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble> TestTypesNorm;
// register the type list
TYPED_TEST_CASE(Random_norm, TestTypesNorm);

// create a list of types to be tested
typedef ::testing::Types<float, double> TestTypesEngine;
// register the type list
TYPED_TEST_CASE(RandomEngine, TestTypesEngine);

typedef ::testing::Types<unsigned> TestTypesEngineSeed;
// register the type list
TYPED_TEST_CASE(RandomEngineSeed, TestTypesEngineSeed);

// create a list of types to be tested
typedef ::testing::Types<unsigned> TestTypesSeed;
// register the type list
TYPED_TEST_CASE(RandomSeed, TestTypesSeed);

template<typename T>
void randuTest(dim4 & dims)
{
    if (noDoubleTests<T>()) return;

    af_array outArray = 0;
    ASSERT_SUCCESS(af_randu(&outArray, dims.ndims(), dims.get(), (af_dtype) dtype_traits<T>::af_type));
    ASSERT_EQ(af_sync(-1), AF_SUCCESS);
    if(outArray != 0) af_release_array(outArray);
}

template<typename T>
void randnTest(dim4 &dims)
{
    if (noDoubleTests<T>()) return;

    af_array outArray = 0;
    ASSERT_SUCCESS(af_randn(&outArray, dims.ndims(), dims.get(), (af_dtype) dtype_traits<T>::af_type));
    ASSERT_EQ(af_sync(-1), AF_SUCCESS);
    if(outArray != 0) af_release_array(outArray);
}

#define RAND(d0, d1, d2, d3)                                \
    TYPED_TEST(Random,randu_##d0##_##d1##_##d2##_##d3)      \
    {                                                       \
        dim4 dims(d0, d1, d2, d3);                          \
        randuTest<TypeParam>(dims);                         \
    }                                                       \
    TYPED_TEST(Random_norm,randn_##d0##_##d1##_##d2##_##d3) \
    {                                                       \
        dim4 dims(d0, d1, d2, d3);                          \
        randnTest<TypeParam>(dims);                         \
    }                                                       \

RAND(1024, 1024,    1,    1);
RAND( 512,  512,    1,    1);
RAND( 256,  256,    1,    1);
RAND( 128,  128,    1,    1);
RAND(  64,   64,    1,    1);
RAND(  32,   32,    1,    1);
RAND(  16,   16,    1,    1);
RAND(   8,    8,    1,    1);
RAND(   4,    4,    1,    1);
RAND(   2,    2,    2,    2);
RAND(   1,    1,    1,    1);
RAND( 256,   16,    4,    2);
RAND(  32,   16,    8,    4);
RAND(   2,    4,   16,  256);
RAND(   4,    8,   16,   32);

RAND(  10,   10,   10,   10);

RAND(1920, 1080,    1,    1);
RAND(1280,  720,    1,    1);
RAND( 640,  480,    1,    1);

RAND( 215,   24,    6,    5);
RAND( 132,   64,   23,    2);
RAND(  15,   35,   50,    3);
RAND(  77,   43,    8,    1);
RAND( 123,   45,    6,    7);
RAND( 345,   28,    9,    1);
RAND(  79,   68,   12,    6);
RAND(  45,    1,    1,    1);

template<typename T>
void randuArgsTest()
{
    if (noDoubleTests<T>()) return;

    dim_t ndims = 4;
    dim_t dims[] = {1, 2, 3, 0};
    af_array outArray = 0;
    ASSERT_EQ(AF_ERR_SIZE, af_randu(&outArray, ndims, dims, (af_dtype) dtype_traits<char>::af_type));
    ASSERT_EQ(af_sync(-1), AF_SUCCESS);
    if(outArray != 0) af_release_array(outArray);
}

TYPED_TEST(Random,InvalidArgs)
{
    randuArgsTest<TypeParam>();
}

template<typename T>
void randuDimsTest()
{
    if (noDoubleTests<T>()) return;

    dim4 dims(1, 65535*32, 1, 1);
    array large_rand = randu(dims, (af_dtype) dtype_traits<T>::af_type);
    ASSERT_EQ(large_rand.dims()[1], 65535*32);

    dims = dim4(1, 1, 65535*32, 1);
    large_rand = randu(dims, (af_dtype) dtype_traits<T>::af_type);
    ASSERT_EQ(large_rand.dims()[2], 65535*32);

    dims = dim4(1, 1, 1, 65535*32);
    large_rand = randu(dims, (af_dtype) dtype_traits<T>::af_type);
    ASSERT_EQ(large_rand.dims()[3], 65535*32);
}

TYPED_TEST(Random,InvalidDims)
{
    randuDimsTest<TypeParam>();
}

////////////////////////////////////// CPP /////////////////////////////////////
//

using af::allTrue;
using af::constant;
using af::getDefaultRandomEngine;
using af::getSeed;
using af::mean;
using af::randomEngine;
using af::randomEngineType;
using af::randu;
using af::setDefaultRandomEngineType;
using af::setSeed;
using af::stdev;
using af::sum;

TEST(RandomEngine, Default)
{
    // Using default Random engine will cause segfaults
    // without setting one. This test should be before
    // setting it to test if default engine setup is working
    // as expected, otherwise the test will fail.
    randomEngine engine = getDefaultRandomEngine();
}

TEST(Random, CPP)
{
    if (noDoubleTests<float>()) return;

    // TEST will fail if exception is thrown, which are thrown
    // when only wrong inputs are thrown on bad access happens
    dim4 dims(1, 2, 3, 1);
    array out1 = randu(dims);
    array out2 = randn(dims);
    setDefaultRandomEngineType(AF_RANDOM_ENGINE_PHILOX);
    array out3 = randu(dims);
    array out4 = randn(dims);
    setDefaultRandomEngineType(AF_RANDOM_ENGINE_THREEFRY);
    array out5 = randu(dims);
    array out6 = randn(dims);
    setDefaultRandomEngineType(AF_RANDOM_ENGINE_MERSENNE);
    array out7 = randu(dims);
    array out8 = randn(dims);
    af::sync();
}

template<typename T>
void testSetSeed(const uintl seed0, const uintl seed1)
{

    if (noDoubleTests<T>()) return;

    uintl orig_seed = getSeed();

    const int num = 1024 * 1024;
    dtype ty = (dtype)dtype_traits<T>::af_type;

    setSeed(seed0);
    array in0 = randu(num, ty);

    setSeed(seed1);
    array in1 = randu(num, ty);

    setSeed(seed0);
    array in2 = randu(num, ty);
    array in3 = randu(num, ty);

    vector<T> h_in0(num);
    vector<T> h_in1(num);
    vector<T> h_in2(num);
    vector<T> h_in3(num);

    in0.host((void *)&h_in0[0]);
    in1.host((void *)&h_in1[0]);
    in2.host((void *)&h_in2[0]);
    in3.host((void *)&h_in3[0]);

    for (int i = 0; i < num; i++) {
        // Verify if same seed produces same arrays
        ASSERT_EQ(h_in0[i], h_in2[i]) << "at : " << i;

        // Verify different arrays created with different seeds differ
        // b8 and u9 can clash because they generate a small set of values
        if (ty != b8 && ty != u8) {
            ASSERT_NE(h_in0[i], h_in1[i]) << "at : " << i;
        }

        // Verify different arrays created one after the other with same seed differ
        // b8 and u9 can clash because they generate a small set of values
        if (ty != b8 && ty != u8) {
            ASSERT_NE(h_in2[i], h_in3[i]) << "at : " << i;
        }
    }

    setSeed(orig_seed); // Reset the seed
}

TYPED_TEST(RandomSeed, setSeed)
{
    testSetSeed<TypeParam>(10101, 23232);
}

template<typename T>
void testGetSeed(const uintl seed0, const uintl seed1)
{
    if (noDoubleTests<T>()) return;

    uintl orig_seed = getSeed();

    const int num = 1024;
    dtype ty = (dtype)dtype_traits<T>::af_type;

    setSeed(seed0);
    array in0 = randu(num, ty);
    ASSERT_EQ(getSeed(), seed0);

    setSeed(seed1);
    array in1 = randu(num, ty);
    ASSERT_EQ(getSeed(), seed1);

    setSeed(seed0);
    array in2 = randu(num, ty);
    ASSERT_EQ(getSeed(), seed0);

    setSeed(orig_seed); // Reset the seed
}

TYPED_TEST(Random, getSeed)
{
    testGetSeed<TypeParam>(1234, 9876);
}

template <typename T>
void testRandomEngineUniform(randomEngineType type)
{
    if (noDoubleTests<T>()) return;
    dtype ty = (dtype)dtype_traits<T>::af_type;

    int elem = 16*1024*1024;
    randomEngine r(type, 0);
    array A = randu(elem, ty, r);
    T m = mean<T>(A);
    T s = stdev<T>(A);
    ASSERT_NEAR(m, 0.5, 1e-3);
    ASSERT_NEAR(s, 0.2887, 1e-2);
}

template <typename T>
void testRandomEngineNormal(randomEngineType type)
{
    if (noDoubleTests<T>()) return;
    dtype ty = (dtype)dtype_traits<T>::af_type;

    int elem = 16*1024*1024;
    randomEngine r(type, 0);
    array A = randn(elem, ty, r);
    T m = mean<T>(A);
    T s = stdev<T>(A);
    ASSERT_NEAR(m, 0, 1e-1);
    ASSERT_NEAR(s, 1, 1e-1);
}

TYPED_TEST(RandomEngine, philoxRandomEngineUniform)
{
    testRandomEngineUniform<TypeParam>(AF_RANDOM_ENGINE_PHILOX_4X32_10);
}

TYPED_TEST(RandomEngine, philoxRandomEngineNormal)
{
    testRandomEngineNormal<TypeParam>(AF_RANDOM_ENGINE_PHILOX_4X32_10);
}

TYPED_TEST(RandomEngine, threefryRandomEngineUniform)
{
    testRandomEngineUniform<TypeParam>(AF_RANDOM_ENGINE_THREEFRY_2X32_16);
}

TYPED_TEST(RandomEngine, threefryRandomEngineNormal)
{
    testRandomEngineNormal<TypeParam>(AF_RANDOM_ENGINE_THREEFRY_2X32_16);
}

TYPED_TEST(RandomEngine, mersenneRandomEngineUniform)
{
    testRandomEngineUniform<TypeParam>(AF_RANDOM_ENGINE_MERSENNE_GP11213);
}

TYPED_TEST(RandomEngine, mersenneRandomEngineNormal)
{
    testRandomEngineNormal<TypeParam>(AF_RANDOM_ENGINE_MERSENNE_GP11213);
}

template <typename T>
void testRandomEngineSeed(randomEngineType type)
{
    int elem = 4*32*1024;
    uintl orig_seed = 0;
    uintl new_seed = 1;
    randomEngine e(type, orig_seed);

    dtype ty = (dtype)dtype_traits<T>::af_type;
    array d1 = randu(elem, ty, e);
    e.setSeed(new_seed);
    array d2 = randu(elem, ty, e);
    e.setSeed(orig_seed);
    array d3 = randu(elem, ty, e);
    array d4 = randu(elem, ty, e);

    vector<T> h1(elem);
    vector<T> h2(elem);
    vector<T> h3(elem);
    vector<T> h4(elem);

    d1.host((void*)h1.data());
    d2.host((void*)h2.data());
    d3.host((void*)h3.data());
    d4.host((void*)h4.data());

    for (int i = 0; i < elem; i++) {
        ASSERT_EQ(h1[i], h3[i]) << "at : " << i;
        if (ty != b8 && ty != u8) {
            ASSERT_NE(h1[i], h2[i]) << "at : " << i;
            ASSERT_NE(h3[i], h4[i]) << "at : " << i;
        }
    }
}

TYPED_TEST(RandomEngineSeed, philoxSeedUniform)
{
    testRandomEngineSeed<TypeParam>(AF_RANDOM_ENGINE_PHILOX_4X32_10);
}

TYPED_TEST(RandomEngineSeed, threefrySeedUniform)
{
    testRandomEngineSeed<TypeParam>(AF_RANDOM_ENGINE_THREEFRY_2X32_16);
}

TYPED_TEST(RandomEngineSeed, mersenneSeedUniform)
{
    testRandomEngineSeed<TypeParam>(AF_RANDOM_ENGINE_MERSENNE_GP11213);
}

template <typename T>
void testRandomEnginePeriod(randomEngineType type)
{
    if (noDoubleTests<T>()) return;
    dtype ty = (dtype)dtype_traits<T>::af_type;

    uint elem = 1024*1024;
    uint steps = 4*1024;
    randomEngine r(type, 0);

    array first = randu(elem, ty, r);

    for (int i = 0; i < steps; ++i) {
        array step = randu(elem, ty, r);
        bool different = !allTrue<bool>(first == step);
        ASSERT_TRUE(different);
    }
}

TYPED_TEST(RandomEngine, DISABLED_philoxRandomEnginePeriod)
{
    testRandomEnginePeriod<TypeParam>(AF_RANDOM_ENGINE_PHILOX_4X32_10);
}

TYPED_TEST(RandomEngine, DISABLED_threefryRandomEnginePeriod)
{
    testRandomEnginePeriod<TypeParam>(AF_RANDOM_ENGINE_THREEFRY_2X32_16);
}

TYPED_TEST(RandomEngine, DISABLED_mersenneRandomEnginePeriod)
{
    testRandomEnginePeriod<TypeParam>(AF_RANDOM_ENGINE_MERSENNE_GP11213);
}

template <typename T>
T chi2_statistic(array input, array expected) {
    expected *= sum<T>(input) / sum<T>(expected);
    array diff = input - expected;
    return sum<T>((diff * diff) / expected);
}

template <typename T>
void testRandomEngineUniformChi2(randomEngineType type)
{
    if (noDoubleTests<T>()) return;
    dtype ty = (dtype)dtype_traits<T>::af_type;

    int elem = 256*1024*1024;
    int steps = 32;
    int bins = 100;

    array total_hist = constant(0.0, bins, ty);
    array expected = constant(1.0/bins, bins, ty);

    randomEngine r(type, 0);

    // R> qchisq(c(5e-6, 1 - 5e-6), 99)
    // [1]  48.68125 173.87456
    T lower = 48.68125;
    T upper = 173.87456;

    bool prev_step = true;
    bool prev_total = true;
    for (int i = 0; i < steps; ++i) {
        array step_hist = histogram(randu(elem, ty, r), bins, 0.0, 1.0);
        T step_chi2 = chi2_statistic<T>(step_hist, expected);
        if (!prev_step) {
          EXPECT_GT(step_chi2, lower) << "at step: " << i;
          EXPECT_LT(step_chi2, upper) << "at step: " << i;
        }
        prev_step = step_chi2 > lower && step_chi2 < upper;

        total_hist += step_hist;
        T total_chi2 = chi2_statistic<T>(total_hist, expected);
        if (!prev_total) {
          EXPECT_GT(total_chi2, lower) << "at step: " << i;
          EXPECT_LT(total_chi2, upper) << "at step: " << i;
        }
        prev_total = total_chi2 > lower && total_chi2 < upper;
    }
}

TYPED_TEST(RandomEngine, DISABLED_philoxRandomEngineUniformChi2)
{
    testRandomEngineUniformChi2<TypeParam>(AF_RANDOM_ENGINE_PHILOX_4X32_10);
}

TYPED_TEST(RandomEngine, DISABLED_threefryRandomEngineUniformChi2)
{
    testRandomEngineUniformChi2<TypeParam>(AF_RANDOM_ENGINE_THREEFRY_2X32_16);
}

TYPED_TEST(RandomEngine, DISABLED_mersenneRandomEngineUniformChi2)
{
    testRandomEngineUniformChi2<TypeParam>(AF_RANDOM_ENGINE_MERSENNE_GP11213);
}
