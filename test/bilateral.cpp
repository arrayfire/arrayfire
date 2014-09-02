#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <testHelpers.hpp>
#include <cmath>

using std::string;
using std::vector;
using af::dim4;

template<typename T>
class Bilateral : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
// FIXME: since af_load_image returns only f32 type arrays
//       only float, double data types test are enabled & passing
//       Note: compareArraysRMSD is handling upcasting while working
//       with two different type of types
//
//typedef ::testing::Types<float, double, int, uint, char, uchar> TestTypes;
typedef ::testing::Types<float, double> TestTypes;

// register the type list
TYPED_TEST_CASE(Bilateral, TestTypes);

TYPED_TEST(Bilateral, InvalidArgs)
{
    vector<TypeParam>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    // check for gray scale bilateral
    af::dim4 dims(5,5,2,2);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<TypeParam>::af_type));
    ASSERT_EQ(AF_ERR_ARG, af_bilateral(&outArray, inArray, 0.12f, 0.34f, false));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));

    // check for color image bilateral
    dims = af::dim4(100,1,1,1);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(),
                dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<TypeParam>::af_type));
    ASSERT_EQ(AF_ERR_ARG, af_bilateral(&outArray, inArray, 0.12f, 0.34f, true));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
}

template<typename T, bool isColor>
void bilateralTest(string pTestFile)
{
    vector<dim4>       inDims;
    vector<string>    inFiles;
    vector<dim_type> outSizes;
    vector<string>   outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {

        af_array inArray  = 0;
        af_array outArray = 0;
        af_array goldArray= 0;
        dim_type nElems   = 0;

        inFiles[testId].insert(0,string(TEST_DIR"/bilateral/"));
        outFiles[testId].insert(0,string(TEST_DIR"/bilateral/"));

        ASSERT_EQ(AF_SUCCESS, af_load_image(&inArray, inFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, af_load_image(&goldArray, outFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&nElems, goldArray));

        ASSERT_EQ(AF_SUCCESS, af_bilateral(&outArray, inArray, 2.25f, 25.56f, isColor));

        T * outData = new T[nElems];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        T * goldData= new T[nElems];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)goldData, goldArray));

        ASSERT_EQ(true, compareArraysRMSD(nElems, goldData, outData, 0.02f));

        ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
        ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
        ASSERT_EQ(AF_SUCCESS, af_destroy_array(goldArray));
    }
}

TYPED_TEST(Bilateral, Grayscale)
{
    bilateralTest<TypeParam, false>(string(TEST_DIR"/bilateral/gray.test"));
}

TYPED_TEST(Bilateral, Color)
{
    bilateralTest<TypeParam, true>(string(TEST_DIR"/bilateral/color.test"));
}
