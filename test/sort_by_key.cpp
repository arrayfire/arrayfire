/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class SortByKey : public ::testing::Test {
   public:
    virtual void SetUp() {
        subMat0.push_back(af_make_seq(0, 4, 1));
        subMat0.push_back(af_make_seq(2, 6, 1));
        subMat0.push_back(af_make_seq(0, 2, 1));
    }
    vector<af_seq> subMat0;
};

// create a list of types to be tested
typedef ::testing::Types<float, double, uint, int, uchar, short, ushort, intl,
                         uintl>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(SortByKey, TestTypes);

template<typename T>
void sortTest(string pTestFile, const bool dir, const unsigned resultIdx0,
              const unsigned resultIdx1, bool isSubRef = false) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;
    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 idims = numDims[0];

    af_array ikeyArray = 0;
    af_array ivalArray = 0;
    af_array tempArray = 0;
    af_array okeyArray = 0;
    af_array ovalArray = 0;

    if (isSubRef) {
        // ASSERT_SUCCESS(af_create_array(&tempArray, &(in[0].front()),
        // idims.ndims(), idims.get(), (af_dtype) dtype_traits<T>::af_type));

        // ASSERT_SUCCESS(af_index(&inArray, tempArray, seqv->size(),
        // &seqv->front()));
    } else {
        ASSERT_SUCCESS(af_create_array(&ikeyArray, &(in[0].front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
        ASSERT_SUCCESS(af_create_array(&ivalArray, &(in[1].front()),
                                       idims.ndims(), idims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
    }

    ASSERT_SUCCESS(
        af_sort_by_key(&okeyArray, &ovalArray, ikeyArray, ivalArray, 0, dir));

    // Compare result
    ASSERT_VEC_ARRAY_EQ(tests[resultIdx0], idims, okeyArray);

#ifdef AF_OPENCL
    UNUSED(resultIdx1);
#else
    // Compare result
    ASSERT_VEC_ARRAY_EQ(tests[resultIdx1], idims, ovalArray);
#endif

    if (ikeyArray != 0) af_release_array(ikeyArray);
    if (ivalArray != 0) af_release_array(ivalArray);
    if (okeyArray != 0) af_release_array(okeyArray);
    if (ovalArray != 0) af_release_array(ovalArray);
    if (tempArray != 0) af_release_array(tempArray);
}

#define SORT_INIT(desc, file, dir, resultIdx0, resultIdx1)                \
    TYPED_TEST(SortByKey, desc) {                                         \
        sortTest<TypeParam>(string(TEST_DIR "/sort/" #file ".test"), dir, \
                            resultIdx0, resultIdx1);                      \
    }

SORT_INIT(Sort0True, sort_by_key_tiny, true, 0, 1);
SORT_INIT(Sort0False, sort_by_key_tiny, false, 2, 3);
SORT_INIT(Sort10x10True, sort_by_key_2D, true, 0, 1);
SORT_INIT(Sort10x10False, sort_by_key_2D, false, 2, 3);
SORT_INIT(Sort1000True, sort_by_key_1000, true, 0, 1);
SORT_INIT(SortMedTrue, sort_by_key_med, true, 0, 1);
SORT_INIT(Sort1000False, sort_by_key_1000, false, 2, 3);
SORT_INIT(SortMedFalse, sort_by_key_med, false, 2, 3);

SORT_INIT(SortLargeTrue, sort_by_key_large, true, 0, 1);
SORT_INIT(SortLargeFalse, sort_by_key_large, false, 2, 3);

////////////////////////////////////// CPP ///////////////////////////////
//
TEST(SortByKey, CPPDim0) {
    const bool dir            = true;
    const unsigned resultIdx0 = 0;
    const unsigned resultIdx1 = 1;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, int>(string(TEST_DIR "/sort/sort_by_key_tiny.test"),
                                 numDims, in, tests);

    dim4 idims = numDims[0];
    array keys(idims, &(in[0].front()));
    array vals(idims, &(in[1].front()));
    array out_keys, out_vals;
    sort(out_keys, out_vals, keys, vals, 0, dir);

    ASSERT_VEC_ARRAY_EQ(tests[resultIdx0], idims, out_keys);
    ASSERT_VEC_ARRAY_EQ(tests[resultIdx1], idims, out_vals);
}

TEST(SortByKey, CPPDim1) {
    const bool dir            = true;
    const unsigned resultIdx0 = 0;
    const unsigned resultIdx1 = 1;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, int>(
        string(TEST_DIR "/sort/sort_by_key_large.test"), numDims, in, tests);

    dim4 idims = numDims[0];
    array keys(idims, &(in[0].front()));
    array vals(idims, &(in[1].front()));

    array keys_ = reorder(keys, 1, 0, 2, 3);
    array vals_ = reorder(vals, 1, 0, 2, 3);

    array out_keys, out_vals;
    sort(out_keys, out_vals, keys_, vals_, 1, dir);

    out_keys = reorder(out_keys, 1, 0, 2, 3);
    out_vals = reorder(out_vals, 1, 0, 2, 3);

    ASSERT_VEC_ARRAY_EQ(tests[resultIdx0], idims, out_keys);
    ASSERT_VEC_ARRAY_EQ(tests[resultIdx1], idims, out_vals);
}

TEST(SortByKey, CPPDim2) {
    const bool dir            = false;
    const unsigned resultIdx0 = 2;
    const unsigned resultIdx1 = 3;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;
    readTests<float, float, int>(
        string(TEST_DIR "/sort/sort_by_key_large.test"), numDims, in, tests);

    dim4 idims = numDims[0];
    array keys(idims, &(in[0].front()));
    array vals(idims, &(in[1].front()));

    array keys_ = reorder(keys, 1, 2, 0, 3);
    array vals_ = reorder(vals, 1, 2, 0, 3);

    array out_keys, out_vals;
    sort(out_keys, out_vals, keys_, vals_, 2, dir);

    out_keys = reorder(out_keys, 2, 0, 1, 3);
    out_vals = reorder(out_vals, 2, 0, 1, 3);

    ASSERT_VEC_ARRAY_EQ(tests[resultIdx0], idims, out_keys);
    ASSERT_VEC_ARRAY_EQ(tests[resultIdx1], idims, out_vals);
}
