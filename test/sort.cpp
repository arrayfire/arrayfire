#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::af_cfloat;
using af::af_cdouble;

template<typename T>
class Sort : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat0.push_back({0, 4, 1});
            subMat0.push_back({2, 6, 1});
            subMat0.push_back({0, 2, 1});
        }
        vector<af_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, uint, int, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(Sort, TestTypes);

template<typename T>
void sortTest(string pTestFile, const bool dir, const unsigned resultIdx0, const unsigned resultIdx1, bool isSubRef = false, const vector<af_seq> * seqv = nullptr)
{
    vector<af::dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, float>(pTestFile,numDims,in,tests);

    af::dim4 idims = numDims[0];

    af_array inArray = 0;
    af_array tempArray = 0;
    af_array sxArray = 0;
    af_array ixArray = 0;

    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_sort(&sxArray, &ixArray, inArray, dir, 0));

    size_t nElems = tests[resultIdx0].size();

    // Get result
    T* sxData = new T[tests[resultIdx0].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)sxData, sxArray));

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx0][elIter], sxData[elIter]) << "at: " << elIter << std::endl;
    }

    // Get result
    unsigned* ixData = new unsigned[tests[resultIdx1].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)ixData, ixArray));

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx1][elIter], ixData[elIter]) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] sxData;
    delete[] ixData;

    if(inArray   != 0) af_destroy_array(inArray);
    if(sxArray   != 0) af_destroy_array(sxArray);
    if(ixArray   != 0) af_destroy_array(ixArray);
    if(tempArray != 0) af_destroy_array(tempArray);
}

#define SORT_INIT(desc, file, dir, resultIdx0, resultIdx1)                                       \
    TYPED_TEST(Sort, desc)                                                                       \
    {                                                                                            \
        sortTest<TypeParam>(string(TEST_DIR"/sort/"#file".test"), dir, resultIdx0, resultIdx1);  \
    }

    SORT_INIT(Sort0True,  sort, true, 0, 1);
    SORT_INIT(Sort0False, sort,false, 2, 3);
