#include <gtest/gtest.h>
#include <arrayfire.h>
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
using af::af_cfloat;
using af::af_cdouble;

template<typename T>
class Random : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

// create a list of types to be tested
typedef ::testing::Types<float, af_cfloat, double, af_cdouble, int, unsigned, unsigned char> TestTypes;

// register the type list
TYPED_TEST_CASE(Random, TestTypes);

template<typename T>
void randuTest()
{
    af::dim4 dims(rand() % 100 + 1, rand() % 100 + 1, rand() % 25 + 1, rand() % 10 + 1);
    af_array outArray = 0;
    ASSERT_EQ(AF_SUCCESS, af_randu(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    if(outArray != 0) af_destroy_array(outArray);
}

template<typename T>
void randnTest()
{
    af::dim4 dims(rand() % 100 + 1, rand() % 100 + 1, rand() % 25 + 1, rand() % 10 + 1);
    af_array outArray = 0;
    ASSERT_EQ(AF_SUCCESS, af_randn(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    if(outArray != 0) af_destroy_array(outArray);
}

// INT, UNIT, CHAR, UCHAR Not Supported by RANDN
template<>
void randnTest<int>()
{
    af::dim4 dims(rand() % 100 + 1, rand() % 100 + 1, rand() % 25 + 1, rand() % 10 + 1);
    af_array outArray = 0;
    ASSERT_EQ(AF_ERR_NOT_SUPPORTED, af_randn(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<int>::af_type));
    if(outArray != 0) af_destroy_array(outArray);
}

template<>
void randnTest<unsigned>()
{
    af::dim4 dims(rand() % 100 + 1, rand() % 100 + 1, rand() % 25 + 1, rand() % 10 + 1);
    af_array outArray = 0;
    ASSERT_EQ(AF_ERR_NOT_SUPPORTED, af_randn(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<unsigned>::af_type));
    if(outArray != 0) af_destroy_array(outArray);
}

template<>
void randnTest<char>()
{
    af::dim4 dims(rand() % 100 + 1, rand() % 100 + 1, rand() % 25 + 1, rand() % 10 + 1);
    af_array outArray = 0;
    ASSERT_EQ(AF_ERR_NOT_SUPPORTED, af_randn(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<char>::af_type));
    if(outArray != 0) af_destroy_array(outArray);
}

template<>
void randnTest<unsigned char>()
{
    af::dim4 dims(rand() % 100 + 1, rand() % 100 + 1, rand() % 25 + 1, rand() % 10 + 1);
    af_array outArray = 0;
    ASSERT_EQ(AF_ERR_NOT_SUPPORTED, af_randn(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<unsigned char>::af_type));
    if(outArray != 0) af_destroy_array(outArray);
}

// Diff on 0 dimension
TYPED_TEST(Random,randu)
{
    randuTest<TypeParam>();
}

TYPED_TEST(Random,randn)
{
    randnTest<TypeParam>();
}

template<typename T>
void randuArgsTest()
{
    af::dim4 dims(1, 2, 3, 0);
    af_array outArray = 0;
    ASSERT_EQ(AF_ERR_ARG, af_randu(&outArray, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<char>::af_type));
    if(outArray != 0) af_destroy_array(outArray);
}

TYPED_TEST(Random,InvalidArgs)
{
    randuArgsTest<TypeParam>();
}
