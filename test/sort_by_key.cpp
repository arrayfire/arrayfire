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

    af_array ikeyArray = 0;
    af_array ivalArray = 0;
    af_array tempArray = 0;
    af_array okeyArray = 0;
    af_array ovalArray = 0;

    if (isSubRef) {
        //ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));

        //ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv->size(), &seqv->front()));
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&ikeyArray, &(in[0].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));
        ASSERT_EQ(AF_SUCCESS, af_create_array(&ivalArray, &(in[1].front()), idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type));
    }

    ASSERT_EQ(AF_SUCCESS, af_sort_by_key(&okeyArray, &ovalArray, ikeyArray, ivalArray, 0, dir));

    size_t nElems = tests[resultIdx0].size();

    // Get result
    T* keyData = new T[tests[resultIdx0].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)keyData, okeyArray));

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx0][elIter], keyData[elIter]) << "at: " << elIter << std::endl;
    }

    T* valData = new T[tests[resultIdx1].size()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)valData, ovalArray));

#ifndef AF_OPENCL
    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx1][elIter], valData[elIter]) << "at: " << elIter << std::endl;
    }
#endif

    // Delete
    delete[] keyData;
    delete[] valData;

    if(ikeyArray != 0) af_release_array(ikeyArray);
    if(ivalArray != 0) af_release_array(ivalArray);
    if(okeyArray != 0) af_release_array(okeyArray);
    if(ovalArray != 0) af_release_array(ovalArray);
    if(tempArray != 0) af_release_array(tempArray);
}

#define SORT_INIT(desc, file, dir, resultIdx0, resultIdx1)                                       \
    TYPED_TEST(Sort, desc)                                                                       \
    {                                                                                            \
        sortTest<TypeParam>(string(TEST_DIR"/sort/"#file".test"), dir, resultIdx0, resultIdx1);  \
    }

    SORT_INIT(Sort0True,      sort_by_key_tiny,  true,  0, 1);
    SORT_INIT(Sort0False,     sort_by_key_tiny,  false, 2, 3);
    SORT_INIT(Sort10x10True,  sort_by_key_2D,    true,  0, 1);
    SORT_INIT(Sort10x10False, sort_by_key_2D,    false, 2, 3);
    SORT_INIT(Sort1000True,   sort_by_key_1000,  true,  0, 1);
    SORT_INIT(Sort1000False,  sort_by_key_1000,  false, 2, 3);
    SORT_INIT(SortMedTrue,    sort_by_key_med,   true,  0, 1);
    SORT_INIT(SortMedFalse,   sort_by_key_med,   false, 2, 3);
    // Takes too much time in current implementation. Enable when everything is parallel
    //SORT_INIT(SortLargeTrue,  sort_by_key_large, true,  0, 1);
    //SORT_INIT(SortLargeFalse, sort_by_key_large, false, 2, 3);




////////////////////////////////////// CPP ///////////////////////////////
//
TEST(SortByKey, CPP)
{
    if (noDoubleTests<float>()) return;

    const bool dir = true;
    const unsigned resultIdx0 = 0;
    const unsigned resultIdx1 = 1;

    vector<af::dim4> numDims;
    vector<vector<float> > in;
    vector<vector<float> > tests;
    readTests<float, float, int>(string(TEST_DIR"/sort/sort_by_key_tiny.test"),numDims,in,tests);

    af::dim4 idims = numDims[0];
    af::array keys(idims, &(in[0].front()));
    af::array vals(idims, &(in[1].front()));
    af::array out_keys, out_vals;
    af::sort(out_keys, out_vals, keys, vals, 0, dir);

    size_t nElems = tests[resultIdx0].size();
    // Get result
    float* keyData = new float[tests[resultIdx0].size()];
    out_keys.host((void*)keyData);

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx0][elIter], keyData[elIter]) << "at: " << elIter << std::endl;
    }

    float* valData = new float[tests[resultIdx1].size()];
    out_vals.host((void*)valData);

    // Compare result
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(tests[resultIdx1][elIter], valData[elIter]) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] keyData;
    delete[] valData;
}

