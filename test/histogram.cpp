#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <testHelpers.hpp>

using std::string;
using std::vector;

template<typename T>
class Histogram : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(Histogram, TestTypes);

TYPED_TEST(Histogram,InvalidArgs)
{
    af::dim4            dims(1);
    vector<TypeParam>   in(100,1);

    af_array inArray   = 0;
    af_array outArray  = 0;

    // square test file is 100x100 originally
    // use new dimensions for this argument
    // unit test
    af::dim4 newDims(5,5,2,2);
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), newDims.ndims(), newDims.get(), (af_dtype) af::dtype_traits<TypeParam>::af_type));

    ASSERT_EQ(AF_ERR_SIZE, af_histogram(&outArray,inArray,256,0,255));
}

template<typename inType, typename outType>
void histTest(string pTestFile, unsigned nbins, double minval, double maxval)
{
    vector<af::dim4> numDims;

    vector<vector<inType>>  in;
    vector<vector<outType>> tests;
    readTests<inType,uint,int>(pTestFile,numDims,in,tests);
    af::dim4 dims       = numDims[0];

    af_array outArray   = 0;
    af_array inArray    = 0;
    outType *outData;
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<inType>::af_type));

    ASSERT_EQ(AF_SUCCESS,af_histogram(&outArray,inArray,nbins,minval,maxval));

    outData = new outType[dims.elements()];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t testIter=0; testIter<tests.size(); ++testIter) {
        vector<outType> currGoldBar = tests[testIter];
        size_t nElems        = currGoldBar.size();
        for (size_t elIter=0; elIter<nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter],outData[elIter])<< "at: " << elIter<< std::endl;
        }
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_destroy_array(outArray));
}

TYPED_TEST(Histogram,256Bins0min255max_ones)
{
    histTest<TypeParam,uint>(string(TEST_DIR"/histogram/256bin1min1max.test"),256,0,255);
}

TYPED_TEST(Histogram,100Bins0min99max)
{
    histTest<TypeParam,uint>(string(TEST_DIR"/histogram/100bin0min99max.test"),100,0,99);
}

TYPED_TEST(Histogram,40Bins0min100max)
{
    histTest<TypeParam,uint>(string(TEST_DIR"/histogram/40bin0min100max.test"),40,0,100);
}

TYPED_TEST(Histogram,40Bins0min100max_Batch)
{
    histTest<TypeParam,uint>(string(TEST_DIR"/histogram/40bin0min100max_batch.test"),40,0,100);
}

TYPED_TEST(Histogram,256Bins0min255max_zeros)
{
    histTest<TypeParam,uint>(string(TEST_DIR"/histogram/256bin0min0max.test"),256,0,255);
}
