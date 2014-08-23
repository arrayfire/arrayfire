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
class Diff1 : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat0.push_back({1, 4, 1});
            subMat0.push_back({0, 2, 1});
            subMat0.push_back({0, 1, 1});

            subMat1.push_back({0, 4, 1});
            subMat1.push_back({1, 3, 1});
            subMat1.push_back({1, 3, 1});

            subMat2.push_back({1, 5, 1});
            subMat2.push_back({0, 3, 1});
            subMat2.push_back({0, 2, 1});
        }
        vector<af_seq> subMat0;
        vector<af_seq> subMat1;
        vector<af_seq> subMat2;
};

// create a list of types to be tested
typedef ::testing::Types<float, af_cfloat, double, af_cdouble, int, unsigned, char, unsigned char> TestTypes;

// register the type list
TYPED_TEST_CASE(Diff1, TestTypes);

template<typename T, unsigned dim>
void diff1Test(string pTestFile, bool isSubRef=false, const vector<af_seq> *seqv=nullptr)
{
    af::dim4            dims(1);
    vector<T>           in;
    vector<vector<T>>   tests;
    ReadTests<int, T>(pTestFile,dims,in,tests);

    T *outData;

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;
    // Get input array
    if (isSubRef) {

        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    // Run diff1
    ASSERT_EQ(AF_SUCCESS, af_diff1(&outArray, inArray, dim));

    // Get result
    outData = new T[dims.elements()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    // Compare result
    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<T> currGoldBar = tests[testIter];
        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter << std::endl;
        }
    }

    // Delete
    delete[] outData;

    if(inArray   != 0) af_destroy_array(inArray);
    if(outArray  != 0) af_destroy_array(outArray);
    if(tempArray != 0) af_destroy_array(tempArray);
}

TYPED_TEST(Diff1,Vector0)
{
    diff1Test<TypeParam, 0>(string(TEST_DIR"/diff1/vector0.test"));
}

TYPED_TEST(Diff1,Matrix0)
{
    diff1Test<TypeParam, 0>(string(TEST_DIR"/diff1/matrix0.test"));
}

TYPED_TEST(Diff1,Matrix1)
{
    diff1Test<TypeParam, 1>(string(TEST_DIR"/diff1/matrix1.test"));
}

// Diff on 0 dimension
TYPED_TEST(Diff1,Basic0)
{
    diff1Test<TypeParam, 0>(string(TEST_DIR"/diff1/basic0.test"));
}

// Diff on 1 dimension
TYPED_TEST(Diff1,Basic1)
{
    diff1Test<TypeParam, 1>(string(TEST_DIR"/diff1/basic1.test"));
}

// Diff on 2 dimension
TYPED_TEST(Diff1,Basic2)
{
    diff1Test<TypeParam, 2>(string(TEST_DIR"/diff1/basic2.test"));
}

#if defined(AF_CPU)
// Diff on 0 dimension subref
TYPED_TEST(Diff1,Subref0)
{
    diff1Test<TypeParam, 0>(string(TEST_DIR"/diff1/subref0.test"),true,&(this->subMat0));
}

// Diff on 1 dimension subref
TYPED_TEST(Diff1,Subref1)
{
    diff1Test<TypeParam, 1>(string(TEST_DIR"/diff1/subref1.test"),true,&(this->subMat1));
}

// Diff on 2 dimension subref
TYPED_TEST(Diff1,Subref2)
{
    diff1Test<TypeParam, 2>(string(TEST_DIR"/diff1/subref2.test"),true,&(this->subMat2));
}
#endif

template<typename T>
void diff1ArgsTest(string pTestFile)
{
    af::dim4          dims(1);
    vector<T>         in;
    vector<vector<T>> tests;
    ReadTests<int, T>(pTestFile,dims,in,tests);

    af_array inArray  = 0;
    af_array outArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<T>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_diff1(&outArray, inArray, -1));
    ASSERT_EQ(AF_ERR_ARG, af_diff1(&outArray, inArray,  5));

    if(inArray  != 0) af_destroy_array(inArray);
    if(outArray != 0) af_destroy_array(outArray);
}

TYPED_TEST(Diff1,InvalidArgs)
{
    diff1ArgsTest<TypeParam>(string(TEST_DIR"/diff1/basic0.test"));
}
