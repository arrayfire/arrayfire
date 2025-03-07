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
#include <af/algorithm.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
void uniqueTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int, int, int>(pTestFile, numDims, data, tests);

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {
        dim4 dims = numDims[d];
        vector<T> in(data[d].begin(), data[d].end());

        af_array inArray  = 0;
        af_array outArray = 0;

        // Get input array
        ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                       dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));

        vector<T> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        ASSERT_SUCCESS(
            af_set_unique(&outArray, inArray, d == 0 ? false : true));

        // Get result
        vector<T> outData(currGoldBar.size());
        ASSERT_SUCCESS(af_get_data_ptr((void *)&outData.front(), outArray));

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << " for test: " << d << endl;
        }

        if (inArray != 0) af_release_array(inArray);
        if (outArray != 0) af_release_array(outArray);
    }
}

#define UNIQUE_TESTS(T) \
    TEST(Set, Test_Unique_##T) { uniqueTest<T>(TEST_DIR "/set/unique.test"); }

UNIQUE_TESTS(float)
UNIQUE_TESTS(double)
UNIQUE_TESTS(int)
UNIQUE_TESTS(uint)
UNIQUE_TESTS(uchar)
UNIQUE_TESTS(short)
UNIQUE_TESTS(ushort)
UNIQUE_TESTS(intl)
UNIQUE_TESTS(uintl)

typedef af_err (*setFunc)(af_array *, const af_array, const af_array,
                          const bool);

template<typename T, setFunc af_set_func>
void setTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int, int, int>(pTestFile, numDims, data, tests);

    // Compare result
    for (int d = 0; d < (int)tests.size(); d += 2) {
        dim4 dims0 = numDims[d + 0];
        vector<T> in0(data[d + 0].begin(), data[d + 0].end());

        dim4 dims1 = numDims[d + 1];
        vector<T> in1(data[d + 1].begin(), data[d + 1].end());

        af_array inArray0 = 0;
        af_array inArray1 = 0;
        af_array outArray = 0;

        ASSERT_SUCCESS(af_create_array(&inArray0, &in0.front(), dims0.ndims(),
                                       dims0.get(),
                                       (af_dtype)dtype_traits<T>::af_type));

        ASSERT_SUCCESS(af_create_array(&inArray1, &in1.front(), dims1.ndims(),
                                       dims1.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
        vector<T> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        ASSERT_SUCCESS(
            af_set_func(&outArray, inArray0, inArray1, d == 0 ? false : true));

        // Get result
        vector<T> outData(currGoldBar.size());
        ASSERT_SUCCESS(af_get_data_ptr((void *)&outData.front(), outArray));

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter])
                << "at: " << elIter << " for test: " << d << endl;
        }

        if (inArray0 != 0) af_release_array(inArray0);
        if (inArray1 != 0) af_release_array(inArray1);
        if (outArray != 0) af_release_array(outArray);
    }
}

#define SET_TESTS(T)                                                  \
    TEST(Set, Test_Union_##T) {                                       \
        setTest<T, af_set_union>(TEST_DIR "/set/union.test");         \
    }                                                                 \
    TEST(Set, Test_Intersect_##T) {                                   \
        setTest<T, af_set_intersect>(TEST_DIR "/set/intersect.test"); \
    }

SET_TESTS(float)
SET_TESTS(double)
SET_TESTS(int)
SET_TESTS(uint)
SET_TESTS(uchar)
SET_TESTS(short)
SET_TESTS(ushort)
SET_TESTS(intl)
SET_TESTS(uintl)

// Documentation examples for setUnique
TEST(Set, SNIPPET_setUniqueSorted) {
    //! [ex_set_unique_sorted]

    // input data
    int h_set[6] = {1, 2, 2, 3, 3, 3};
    af::array set(6, h_set);

    // is_sorted flag specifies if input is sorted,
    // allows algorithm to skip internal sorting step
    const bool is_sorted = true;
    af::array unique     = setUnique(set, is_sorted);
    // unique == { 1, 2, 3 };

    //! [ex_set_unique_sorted]

    vector<int> unique_gold = {1, 2, 3};
    dim4 gold_dim(3, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(unique_gold, gold_dim, unique);
}

TEST(Set, SNIPPET_setUniqueSortedDesc) {
    //! [ex_set_unique_desc]

    // input data
    int h_set[6] = {3, 3, 3, 2, 2, 1};
    af::array set(6, h_set);

    // is_sorted flag specifies if input is sorted,
    // allows algorithm to skip internal sorting step
    // input can be sorted in ascending or descending order
    const bool is_sorted = true;
    af::array unique     = setUnique(set, is_sorted);
    // unique == { 3, 2, 1 };

    //! [ex_set_unique_desc]

    vector<int> unique_gold = {3, 2, 1};
    dim4 gold_dim(3, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(unique_gold, gold_dim, unique);
}

TEST(Set, SNIPPET_setUniqueSimple) {
    //! [ex_set_unique_simple]

    // input data
    int h_set[6] = {3, 2, 3, 3, 2, 1};
    af::array set(6, h_set);

    af::array unique = setUnique(set);
    // unique == { 1, 2, 3 };

    //! [ex_set_unique_simple]

    vector<int> unique_gold = {1, 2, 3};
    dim4 gold_dim(3, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(unique_gold, gold_dim, unique);
}

// Documentation examples for setUnion
TEST(Set, SNIPPET_setUnion) {
    //! [ex_set_union]

    // input data
    int h_setA[4] = {1, 2, 3, 4};
    int h_setB[4] = {2, 3, 4, 5};
    af::array setA(4, h_setA);
    af::array setB(4, h_setB);

    const bool is_unique = true;
    // is_unique flag specifies if inputs are unique,
    // allows algorithm to skip internal calls to setUnique
    // inputs must be unique and sorted in increasing order
    af::array setAB = setUnion(setA, setB, is_unique);
    // setAB == { 1, 2, 3, 4, 5 };

    //! [ex_set_union]

    vector<int> union_gold = {1, 2, 3, 4, 5};
    dim4 gold_dim(5, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(union_gold, gold_dim, setAB);
}

TEST(Set, SNIPPET_setUnionSimple) {
    //! [ex_set_union_simple]

    // input data
    int h_setA[4] = {1, 2, 3, 3};
    int h_setB[4] = {3, 4, 5, 5};
    af::array setA(4, h_setA);
    af::array setB(4, h_setB);

    af::array setAB = setUnion(setA, setB);
    // setAB == { 1, 2, 3, 4, 5 };

    //! [ex_set_union_simple]

    vector<int> union_gold = {1, 2, 3, 4, 5};
    dim4 gold_dim(5, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(union_gold, gold_dim, setAB);
}

// Documentation examples for setIntersect()
TEST(Set, SNIPPET_setIntersect) {
    //! [ex_set_intersect]

    // input data
    int h_setA[4] = {1, 2, 3, 4};
    int h_setB[4] = {2, 3, 4, 5};
    af::array setA(4, h_setA);
    af::array setB(4, h_setB);

    const bool is_unique = true;
    // is_unique flag specifies if inputs are unique,
    // allows algorithm to skip internal calls to setUnique
    // inputs must be unique and sorted in increasing order
    af::array setA_B = setIntersect(setA, setB, is_unique);
    // setA_B == { 2, 3, 4 };

    //! [ex_set_intersect]

    vector<int> intersect_gold = {2, 3, 4};
    dim4 gold_dim(3, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(intersect_gold, gold_dim, setA_B);
}

TEST(Set, SNIPPET_setIntersectSimple) {
    //! [ex_set_intersect_simple]

    // input data
    int h_setA[4] = {1, 2, 3, 3};
    int h_setB[4] = {3, 3, 4, 5};
    af::array setA(4, h_setA);
    af::array setB(4, h_setB);

    af::array setA_B = setIntersect(setA, setB);
    // setA_B == { 3 };

    //! [ex_set_intersect_simple]

    vector<int> intersect_gold = {3};
    dim4 gold_dim(1, 1, 1, 1);
    ASSERT_VEC_ARRAY_EQ(intersect_gold, gold_dim, setA_B);
}
