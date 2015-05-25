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
#include <af/traits.hpp>
#include <af/algorithm.h>
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
void uniqueTest(string pTestFile)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (pTestFile,numDims,data,tests);


    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {

        af::dim4 dims       = numDims[d];
        vector<T> in(data[d].begin(), data[d].end());

        af_array inArray   = 0;
        af_array outArray  = 0;

        // Get input array
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), dims.ndims(),
                                              dims.get(), (af_dtype) af::dtype_traits<T>::af_type));


        vector<T> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        ASSERT_EQ(AF_SUCCESS, af_set_unique(&outArray, inArray, d == 0 ? false : true));

        // Get result
        T *outData;
        outData = new T[currGoldBar.size()];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                                                            << " for test: " << d << std::endl;
        }

        // Delete
        delete[] outData;

        if(inArray   != 0) af_release_array(inArray);
        if(outArray  != 0) af_release_array(outArray);
    }
}

#define UNIQUE_TESTS(T)                             \
    TEST(Set, Test_Unique_##T)                      \
    {                                               \
        uniqueTest<T>(TEST_DIR"/set/unique.test");  \
    }                                               \

UNIQUE_TESTS(float)
UNIQUE_TESTS(double)
UNIQUE_TESTS(int)
UNIQUE_TESTS(uint)
UNIQUE_TESTS(uchar)

typedef af_err (*setFunc)(af_array *, const af_array, const af_array, const bool);

template<typename T, setFunc af_set_func>
void setTest(string pTestFile)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (pTestFile,numDims,data,tests);


    // Compare result
    for (int d = 0; d < (int)tests.size(); d += 2) {

        af::dim4 dims0       = numDims[d + 0];
        vector<T> in0(data[d + 0].begin(), data[d + 0].end());

        af::dim4 dims1       = numDims[d + 1];
        vector<T> in1(data[d + 1].begin(), data[d + 1].end());

        af_array inArray0   = 0;
        af_array inArray1   = 0;
        af_array outArray  = 0;

        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray0, &in0.front(), dims0.ndims(),
                                              dims0.get(), (af_dtype) af::dtype_traits<T>::af_type));


        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray1, &in1.front(), dims1.ndims(),
                                              dims1.get(), (af_dtype) af::dtype_traits<T>::af_type));


        vector<T> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        ASSERT_EQ(AF_SUCCESS, af_set_func(&outArray, inArray0, inArray1, d == 0 ? false : true));

        // Get result
        T *outData;
        outData = new T[currGoldBar.size()];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                                                            << " for test: " << d << std::endl;
        }

        // Delete
        delete[] outData;

        if(inArray0   != 0) af_release_array(inArray0);
        if(inArray1   != 0) af_release_array(inArray1);
        if(outArray  != 0) af_release_array(outArray);
    }
}

#define SET_TESTS(T)                                                    \
    TEST(Set, Test_Union_##T)                                           \
    {                                                                   \
        setTest<T, af_set_union>(TEST_DIR"/set/union.test");            \
    }                                                                   \
    TEST(Set, Test_Intersect_##T)                                       \
    {                                                                   \
        setTest<T, af_set_intersect>(TEST_DIR"/set/intersect.test");    \
    }                                                                   \

SET_TESTS(float)
SET_TESTS(double)
SET_TESTS(int)
SET_TESTS(uint)
SET_TESTS(uchar)
