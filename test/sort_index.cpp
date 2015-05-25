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
using af::cfloat;
using af::cdouble;

template<typename T>
class Sort : public ::testing::Test
{
    public:
        virtual void SetUp() {
            subMat0.push_back(af_make_seq(0, 4, 1));
            subMat0.push_back(af_make_seq(2, 6, 1));
            subMat0.push_back(af_make_seq(0, 2, 1));
        }
        vector<af_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, uint, int, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(Sort, TestTypes);

template<typename T>
void sortTest(string pTestFile, const bool dir, const unsigned resultIdx0, const unsigned resultIdx1, bool isSubRef = false, const vector<af_seq> * seqv = NULL)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;
    vector<vector<T> > in;
    vector<vector<float> > tests;
    readTests<T, float, int>(pTestFile,numDims,in,tests);

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

    ASSERT_EQ(AF_SUCCESS, af_sort_index(&sxArray, &ixArray, inArray, 0, dir));

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

#ifndef AF_OPENCL
    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx1][elIter], ixData[elIter]) << "at: " << elIter << std::endl;
    }
#endif

    // Delete
    delete[] sxData;
    delete[] ixData;

    if(inArray   != 0) af_release_array(inArray);
    if(sxArray   != 0) af_release_array(sxArray);
    if(ixArray   != 0) af_release_array(ixArray);
    if(tempArray != 0) af_release_array(tempArray);
}

#define SORT_INIT(desc, file, dir, resultIdx0, resultIdx1)                                       \
    TYPED_TEST(Sort, desc)                                                                       \
    {                                                                                            \
        sortTest<TypeParam>(string(TEST_DIR"/sort/"#file".test"), dir, resultIdx0, resultIdx1);  \
    }

    SORT_INIT(Sort0True,  sort, true, 0, 1);
    SORT_INIT(Sort0False, sort,false, 2, 3);

    SORT_INIT(Sort2d0False, basic_2d, true, 0, 1);

    SORT_INIT(Sort10x10True,  sort_10x10, true,  0, 1);
    SORT_INIT(Sort10x10False, sort_10x10, false, 2, 3);
    SORT_INIT(Sort1000True,   sort_1000,  true,  0, 1);
    SORT_INIT(Sort1000False,  sort_1000,  false, 2, 3);
    SORT_INIT(SortMedTrue,    sort_med1,  true,  0, 1);
    SORT_INIT(SortMedFalse,   sort_med1,  false, 2, 3);
    // Takes too much time in current implementation. Enable when everything is parallel
    //SORT_INIT(SortMed5True,   sort_med,   true,  0, 1);
    //SORT_INIT(SortMed5False,  sort_med,   false, 2, 3);
    //SORT_INIT(SortLargeTrue,  sort_large, true,  0, 1);
    //SORT_INIT(SortLargeFalse, sort_large, false, 2, 3);
;


//////////////////////////////////// CPP /////////////////////////////////
//
TEST(SortIndex, CPP)
{
    if (noDoubleTests<float>()) return;

    const bool dir = true;
    const unsigned resultIdx0 = 0;
    const unsigned resultIdx1 = 1;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, int>(string(TEST_DIR"/sort/sort_10x10.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array input(idims, &(in[0].front()));
    af::array outValues, outIndices;
    af::sort(outValues, outIndices, input, 0, dir);

    size_t nElems = tests[resultIdx0].size();

    // Get result
    float* sxData = new float[tests[resultIdx0].size()];
    outValues.host((void*)sxData);

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx0][elIter], sxData[elIter]) << "at: " << elIter << std::endl;
    }

    // Get result
    unsigned* ixData = new unsigned[tests[resultIdx1].size()];
    outIndices.host((void*)ixData);

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx1][elIter], ixData[elIter]) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] sxData;
    delete[] ixData;
}
