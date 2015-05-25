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
#include <af/array.h>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>
#include <af/device.h>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;

typedef af_err (*scanFunc)(af_array *, const af_array, const int);

template<typename Ti, typename To, scanFunc af_scan>
void scanTest(string pTestFile, int off = 0, bool isSubRef=false, const vector<af_seq> seqv=vector<af_seq>())
{
    if (noDoubleTests<Ti>()) return;

    vector<af::dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (pTestFile,numDims,data,tests);
    af::dim4 dims       = numDims[0];

    vector<Ti> in(data[0].begin(), data[0].end());

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    // Get input array
    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<Ti>::af_type));
        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv.size(), &seqv.front()));
        ASSERT_EQ(AF_SUCCESS, af_release_array(tempArray));
    } else {

        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<Ti>::af_type));
    }

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {
        vector<To> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        ASSERT_EQ(AF_SUCCESS, af_scan(&outArray, inArray, d + off));

        // Get result
        To *outData;
        outData = new To[dims.elements()];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                << " for dim " << d +off
                << std::endl;
        }

        // Delete
        delete[] outData;
        ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
    }

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
}

vector<af_seq> init_subs()
{
    vector<af_seq> subs;
    subs.push_back(af_make_seq(2, 6, 1));
    subs.push_back(af_make_seq(1, 5, 1));
    subs.push_back(af_make_seq(1, 3, 1));
    subs.push_back(af_make_seq(1, 2, 1));
    return subs;
}

#define SCAN_TESTS(FN, TAG, Ti, To)             \
    TEST(Scan,Test_##FN##_##TAG)                \
    {                                           \
        scanTest<Ti, To, af_##FN>(              \
            string(TEST_DIR"/scan/"#FN".test")  \
            );                                  \
    }                                           \

SCAN_TESTS(accum, float   , float     , float     );
SCAN_TESTS(accum, double  , double    , double    );
SCAN_TESTS(accum, int     , int       , int       );
SCAN_TESTS(accum, cfloat  , cfloat , cfloat );
SCAN_TESTS(accum, cdouble , cdouble, cdouble);
SCAN_TESTS(accum, unsigned, unsigned  , unsigned  );
SCAN_TESTS(accum, uchar   , unsigned char, unsigned);

TEST(Scan,Test_Scan_Big0)
{
    scanTest<int, int, af_accum>(
        string(TEST_DIR"/scan/big0.test"),
        0
        );
}

TEST(Scan,Test_Scan_Big1)
{
    scanTest<int, int, af_accum>(
        string(TEST_DIR"/scan/big1.test"),
        1
        );
}

///////////////////////////////// CPP ////////////////////////////////////
//
TEST(Scan, CPP)
{
    vector<af::dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (string(TEST_DIR"/scan/accum.test"),numDims,data,tests);
    af::dim4 dims       = numDims[0];

    vector<float> in(data[0].begin(), data[0].end());

    if (noDoubleTests<float>()) return;

    af::array input(dims, &(in.front()));

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {
        vector<float> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        af::array output = af::accum(input, d);

        // Get result
        float *outData;
        outData = new float[dims.elements()];
        output.host((void*)outData);

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                << " for dim " << d
                << std::endl;
        }

        // Delete
        delete[] outData;
    }
}
