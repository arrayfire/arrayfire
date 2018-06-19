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
#include <utility>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::allTrue;
using af::array;
using af::cfloat;
using af::cdouble;
using af::constant;
using af::dim4;
using af::dtype_traits;
using af::range;
using af::span;
using af::seq;

typedef af_err (*scanFunc)(af_array *, const af_array, const int);

template<typename Ti, typename To, scanFunc af_scan>
void scanTest(string pTestFile, int off = 0, bool isSubRef=false, const vector<af_seq> seqv=vector<af_seq>())
{
    if (noDoubleTests<Ti>()) return;

    vector<dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (pTestFile,numDims,data,tests);
    dim4 dims       = numDims[0];

    vector<Ti> in(data[0].begin(), data[0].end());

    af_array inArray   = 0;
    af_array outArray  = 0;
    af_array tempArray = 0;

    // Get input array
    if (isSubRef) {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&tempArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) dtype_traits<Ti>::af_type));
        ASSERT_EQ(AF_SUCCESS, af_index(&inArray, tempArray, seqv.size(), &seqv.front()));
        ASSERT_EQ(AF_SUCCESS, af_release_array(tempArray));
    } else {

        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) dtype_traits<Ti>::af_type));
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
                << endl;
        }

        // Delete
        delete[] outData;
        ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
    }

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
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
SCAN_TESTS(accum, cfloat  , cfloat    , cfloat    );
SCAN_TESTS(accum, cdouble , cdouble   , cdouble   );
SCAN_TESTS(accum, unsigned, unsigned  , unsigned  );
SCAN_TESTS(accum, intl    , intl      , intl      );
SCAN_TESTS(accum, uintl   , uintl     , uintl     );
SCAN_TESTS(accum, uchar   , uchar     , unsigned  );
SCAN_TESTS(accum, short   , short     , int       );
SCAN_TESTS(accum, ushort  , ushort    , uint      );

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
TEST(Accum, CPP)
{
    vector<dim4> numDims;

    vector<vector<int> > data;
    vector<vector<int> > tests;
    readTests<int,int,int> (string(TEST_DIR"/scan/accum.test"),numDims,data,tests);
    dim4 dims       = numDims[0];

    vector<float> in(data[0].begin(), data[0].end());

    if (noDoubleTests<float>()) return;

    array input(dims, &(in.front()));

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {
        vector<float> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        array output = accum(input, d);

        // Get result
        float *outData;
        outData = new float[dims.elements()];
        output.host((void*)outData);

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                << " for dim " << d
                << endl;
        }

        // Delete
        delete[] outData;
    }
}

TEST(Accum, MaxDim)
{
    const size_t largeDim = 65535 * 32 + 1;

    //first dimension kernel tests
    array input = constant(0, 2, largeDim, 2, 2);
    input(span, seq(0, 9999), span, span) = 1;

    array gold_first = constant(0, 2, largeDim, 2, 2);
    gold_first(span, seq(0, 9999), span, span) = range(2, 10000, 2, 2) + 1;

    array output_first = accum(input, 0);
    ASSERT_TRUE(allTrue<bool>(output_first == gold_first));


    input = constant(0, 2, 2, 2, largeDim);
    input(span, span, span, seq(0, 9999)) = 1;

    gold_first = constant(0, 2, 2, 2, largeDim);
    gold_first(span, span, span, seq(0, 9999)) = range(2, 2, 2, 10000) + 1;

    output_first = accum(input, 0);
    ASSERT_TRUE(allTrue<bool>(output_first == gold_first));


    //other dimension kernel tests
    input = constant(0, 2, largeDim, 2, 2);
    input(span, seq(0, 9999), span, span) = 1;

    array gold_dim = constant(10000, 2, largeDim, 2, 2);
    gold_dim(span, seq(0, 9999), span, span) = range(dim4(2, 10000, 2, 2), 1) + 1;

    array output_dim = accum(input, 1);
    ASSERT_TRUE(allTrue<bool>(output_dim == gold_dim));


    input = constant(0, 2, 2, 2, largeDim);
    input(span, span, span, seq(0, 9999)) = 1;

    gold_dim = constant(0, 2, 2, 2, largeDim);
    gold_dim(span, span, span, seq(0, 9999)) = range(dim4(2, 2, 2, 10000), 1) + 1;

    output_dim = accum(input, 1);
    ASSERT_TRUE(allTrue<bool>(output_dim == gold_dim));

}
