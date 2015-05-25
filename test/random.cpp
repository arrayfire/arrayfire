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
using af::cfloat;
using af::cdouble;

template<typename T>
class Random : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble, int, unsigned, unsigned char> TestTypes;

// register the type list
TYPED_TEST_CASE(Random, TestTypes);

template<typename T>
class Random_norm : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble> TestTypesNorm;

// register the type list
TYPED_TEST_CASE(Random_norm, TestTypesNorm);

template<typename T>
void randuTest(af::dim4 & dims)
{
    if (noDoubleTests<T>()) return;

    af_array outArray = 0;
    ASSERT_EQ(AF_SUCCESS, af_randu(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    if(outArray != 0) af_release_array(outArray);
}

template<typename T>
void randnTest(af::dim4 &dims)
{
    if (noDoubleTests<T>()) return;

    af_array outArray = 0;
    ASSERT_EQ(AF_SUCCESS, af_randn(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    if(outArray != 0) af_release_array(outArray);
}

// INT, UNIT, CHAR, UCHAR Not Supported by RANDN
template<>
void randnTest<int>(af::dim4 &dims)
{
    if (noDoubleTests<int>()) return;

    af_array outArray = 0;
    ASSERT_EQ(AF_ERR_TYPE, af_randn(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<int>::af_type));
    if(outArray != 0) af_release_array(outArray);
}

template<>
void randnTest<unsigned>(af::dim4 &dims)
{
    if (noDoubleTests<unsigned>()) return;

    af_array outArray = 0;
    ASSERT_EQ(AF_ERR_TYPE, af_randn(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<unsigned>::af_type));
    if(outArray != 0) af_release_array(outArray);
}

template<>
void randnTest<char>(af::dim4 &dims)
{
    if (noDoubleTests<char>()) return;

    af_array outArray = 0;
    ASSERT_EQ(AF_ERR_TYPE, af_randn(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<char>::af_type));
    if(outArray != 0) af_release_array(outArray);
}

template<>
void randnTest<unsigned char>(af::dim4 &dims)
{
    if (noDoubleTests<unsigned char>()) return;

    af_array outArray = 0;
    ASSERT_EQ(AF_ERR_TYPE, af_randn(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<unsigned char>::af_type));
    if(outArray != 0) af_release_array(outArray);
}

#define RAND(d0, d1, d2, d3)                            \
    TYPED_TEST(Random,randu_##d0##_##d1##_##d2##_##d3)  \
    {                                                   \
        af::dim4 dims(d0, d1, d2, d3);                  \
        randuTest<TypeParam>(dims);                     \
    }                                                   \
    TYPED_TEST(Random,randn_##d0##_##d1##_##d2##_##d3)  \
    {                                                   \
        af::dim4 dims(d0, d1, d2, d3);                  \
        randnTest<TypeParam>(dims);                     \
    }                                                   \

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
    ASSERT_EQ(AF_ERR_SIZE, af_randu(&outArray, ndims, dims, (af_dtype) af::dtype_traits<char>::af_type));
    if(outArray != 0) af_release_array(outArray);
}

TYPED_TEST(Random,InvalidArgs)
{
    randuArgsTest<TypeParam>();
}

////////////////////////////////////// CPP /////////////////////////////////////
//
TEST(Random, CPP)
{
    if (noDoubleTests<float>()) return;

    // TEST will fail if exception is thrown, which are thrown
    // when only wrong inputs are thrown on bad access happens
    af::dim4 dims(1, 2, 3, 1);
    af::array out1 = af::randu(dims);
    af::array out2 = af::randn(dims);
}

template<typename T>
void testSetSeed(const uintl seed0, const uintl seed1, bool is_norm = false)
{

    if (noDoubleTests<T>()) return;

    const int num = 1024 * 1024;
    af::dtype ty = (af::dtype)af::dtype_traits<T>::af_type;

    af::setSeed(seed0);
    af::array in0 = is_norm ? af::randn(num, ty) : af::randu(num, ty);

    af::setSeed(seed1);
    af::array in1 = is_norm ? af::randn(num, ty) : af::randu(num, ty);

    af::setSeed(seed0);
    af::array in2 = is_norm ? af::randn(num, ty) : af::randu(num, ty);

    std::vector<T> h_in0(num);
    std::vector<T> h_in1(num);
    std::vector<T> h_in2(num);

    in0.host((void *)&h_in0[0]);
    in1.host((void *)&h_in1[0]);
    in2.host((void *)&h_in2[0]);

    for (int i = 0; i < num; i++) {
        // Verify if same seed produces same arrays
        ASSERT_EQ(h_in0[i], h_in2[i]);

        // Verify different arrays don't clash at same location
        // b8 and u9 can clash because they generate a small set of values
        if (ty != b8 && ty != u8) ASSERT_NE(h_in0[i], h_in1[i]);
    }
}

TYPED_TEST(Random, setSeed)
{
    testSetSeed<TypeParam>(10101, 23232, false);
}

TYPED_TEST(Random_norm, setSeed)
{
    testSetSeed<TypeParam>(456, 789, false);
}

template<typename T>
void testGetSeed(const uintl seed0, const uintl seed1)
{
    if (noDoubleTests<T>()) return;

    const int num = 1024;
    af::dtype ty = (af::dtype)af::dtype_traits<T>::af_type;

    af::setSeed(seed0);
    af::array in0 = af::randu(num, ty);
    ASSERT_EQ(af::getSeed(), seed0);

    af::setSeed(seed1);
    af::array in1 = af::randu(num, ty);
    ASSERT_EQ(af::getSeed(), seed1);

    af::setSeed(seed0);
    af::array in2 = af::randu(num, ty);
    ASSERT_EQ(af::getSeed(), seed0);
}

TYPED_TEST(Random, getSeed)
{
    testGetSeed<TypeParam>(1234, 9876);
}
