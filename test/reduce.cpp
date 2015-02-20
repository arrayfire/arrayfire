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
#include <af/reduce.h>
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

typedef af_err (*reduceFunc)(af_array *, const af_array, const int);

template<typename Ti, typename To, reduceFunc af_reduce>
void reduceTest(string pTestFile, int off = 0, bool isSubRef=false, const vector<af_seq> seqv=vector<af_seq>())
{
    if (noDoubleTests<Ti>()) return;
    if (noDoubleTests<To>()) return;

    vector<af::dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
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
    } else {
        ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &in.front(), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<Ti>::af_type));
    }

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {

        vector<To> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        ASSERT_EQ(AF_SUCCESS, af_reduce(&outArray, inArray, d + off));

        // Get result
        To *outData;
        outData = new To[dims.elements()];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                                                            << " for dim " << d + off << std::endl;
        }

        // Delete
        delete[] outData;
    }


    if(inArray   != 0) af_destroy_array(inArray);
    if(outArray  != 0) af_destroy_array(outArray);
    if(tempArray != 0) af_destroy_array(tempArray);
}

vector<af_seq> init_subs()
{
    vector<af_seq> subs;
    subs.push_back({2, 6, 1});
    subs.push_back({1, 5, 1});
    subs.push_back({1, 3, 1});
    subs.push_back({1, 2, 1});
    return subs;
}

#define REDUCE_TESTS(FN, TAG, Ti, To)                   \
    TEST(Reduce,Test_##FN##_##TAG)                      \
    {                                                   \
        reduceTest<Ti, To, af_##FN>(                    \
            string(TEST_DIR"/reduce/"#FN".test")        \
            );                                          \
    }                                                   \

REDUCE_TESTS(sum, float   , float     , float     );
REDUCE_TESTS(sum, double  , double    , double    );
REDUCE_TESTS(sum, int     , int       , int       );
REDUCE_TESTS(sum, cfloat  , cfloat , cfloat );
REDUCE_TESTS(sum, cdouble , cdouble, cdouble);
REDUCE_TESTS(sum, unsigned, unsigned  , unsigned  );
REDUCE_TESTS(sum, uchar   , unsigned char, unsigned);

REDUCE_TESTS(min, float   , float     , float     );
REDUCE_TESTS(min, double  , double    , double    );
REDUCE_TESTS(min, int     , int       , int       );
REDUCE_TESTS(min, cfloat  , cfloat , cfloat );
REDUCE_TESTS(min, cdouble , cdouble, cdouble);
REDUCE_TESTS(min, unsigned, unsigned  , unsigned  );
REDUCE_TESTS(min, uchar   , unsigned char, unsigned char);

REDUCE_TESTS(max, float   , float     , float     );
REDUCE_TESTS(max, double  , double    , double    );
REDUCE_TESTS(max, int     , int       , int       );
REDUCE_TESTS(max, cfloat  , cfloat , cfloat );
REDUCE_TESTS(max, cdouble , cdouble, cdouble);
REDUCE_TESTS(max, unsigned, unsigned  , unsigned  );
REDUCE_TESTS(max, uchar   , unsigned char, unsigned char);

REDUCE_TESTS(anytrue, float   , float     , unsigned char);
REDUCE_TESTS(anytrue, double  , double    , unsigned char);
REDUCE_TESTS(anytrue, int     , int       , unsigned char);
REDUCE_TESTS(anytrue, cfloat  , cfloat , unsigned char);
REDUCE_TESTS(anytrue, cdouble , cdouble, unsigned char);
REDUCE_TESTS(anytrue, unsigned, unsigned  , unsigned char);
REDUCE_TESTS(anytrue, uchar   , unsigned char, unsigned char);

REDUCE_TESTS(alltrue, float   , float     , unsigned char);
REDUCE_TESTS(alltrue, double  , double    , unsigned char);
REDUCE_TESTS(alltrue, int     , int       , unsigned char);
REDUCE_TESTS(alltrue, cfloat  , cfloat , unsigned char);
REDUCE_TESTS(alltrue, cdouble , cdouble, unsigned char);
REDUCE_TESTS(alltrue, unsigned, unsigned  , unsigned char);
REDUCE_TESTS(alltrue, uchar   , unsigned char, unsigned char);

REDUCE_TESTS(count, float   , float     , unsigned);
REDUCE_TESTS(count, double  , double    , unsigned);
REDUCE_TESTS(count, int     , int       , unsigned);
REDUCE_TESTS(count, cfloat  , cfloat , unsigned);
REDUCE_TESTS(count, cdouble , cdouble, unsigned);
REDUCE_TESTS(count, unsigned, unsigned  , unsigned);
REDUCE_TESTS(count, uchar   , unsigned char, unsigned);

TEST(Reduce,Test_Reduce_Big0)
{
    if (noDoubleTests<int>()) return;

    reduceTest<int, int, af_sum>(
        string(TEST_DIR"/reduce/big0.test"),
        0
        );
}

TEST(Reduce,Test_Reduce_Big1)
{
    if (noDoubleTests<int>()) return;

    reduceTest<int, int, af_sum>(
        string(TEST_DIR"/reduce/big1.test"),
        1
        );
}

/////////////////////////////////// CPP //////////////////////////////////
//
typedef af::array (*ReductionOp)(const af::array&, const int);

using af::sum;
using af::min;
using af::max;
using af::alltrue;
using af::anytrue;
using af::count;

template<typename Ti, typename To, ReductionOp reduce>
void cppReduceTest(string pTestFile)
{
    if (noDoubleTests<Ti>()) return;
    if (noDoubleTests<To>()) return;

    vector<af::dim4> numDims;

    vector<vector<int>> data;
    vector<vector<int>> tests;
    readTests<int,int,int> (pTestFile,numDims,data,tests);
    af::dim4 dims       = numDims[0];

    vector<Ti> in(data[0].begin(), data[0].end());

    af::array input(dims, &in.front());

    // Compare result
    for (int d = 0; d < (int)tests.size(); ++d) {

        vector<To> currGoldBar(tests[d].begin(), tests[d].end());

        // Run sum
        af::array output = reduce(input, d);

        // Get result
        To *outData = new To[dims.elements()];
        output.host((void*)outData);

        size_t nElems = currGoldBar.size();
        for (size_t elIter = 0; elIter < nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter], outData[elIter]) << "at: " << elIter
                                                            << " for dim " << d << std::endl;
        }

        // Delete
        delete[] outData;
    }
}

#define CPP_REDUCE_TESTS(FN, Ti, To)               \
    TEST(Reduce, Test_##FN##_CPP)                  \
    {                                              \
        cppReduceTest<Ti, To, FN>(                 \
            string(TEST_DIR"/reduce/"#FN".test")   \
            );                                     \
    }

CPP_REDUCE_TESTS(sum, float, float);
CPP_REDUCE_TESTS(min, float, float);
CPP_REDUCE_TESTS(max, float, float);
CPP_REDUCE_TESTS(anytrue, float, unsigned char);
CPP_REDUCE_TESTS(alltrue, float, unsigned char);
CPP_REDUCE_TESTS(count, float, unsigned);

TEST(Reduce, Test_Sum_Global)
{
    int num = 10000;
    af::array a = af::round(2 * af::randu(num, 1));

    float res = af::sum<float>(a);
    float *h_a = a.host<float>();
    float gold = 0;

    for (int i = 0; i < num; i++) {
        gold += h_a[i];
    }

    ASSERT_EQ(gold, res);
    delete[] h_a;
}

TEST(Reduce, Test_Count_Global)
{
    int num = 10000;
    af::array a = af::round(2 * af::randu(num, 1));
    af::array b = a.as(b8);

    int res = af::count<int>(b);
    char *h_b = b.host<char>();
    int gold = 0;

    for (int i = 0; i < num; i++) {
        gold += h_b[i];
    }

    ASSERT_EQ(gold, res);
    delete[] h_b;
}

TEST(Reduce, Test_min_Global)
{
    if (noDoubleTests<double>()) return;

    int num = 10000;
    af::array a = af::randu(num, 1, f64);
    double res = af::min<double>(a);
    double *h_a = a.host<double>();
    double gold = std::numeric_limits<double>::max();

    if (noDoubleTests<double>()) return;

    for (int i = 0; i < num; i++) {
        gold = std::min(gold, h_a[i]);
    }

    ASSERT_EQ(gold, res);
    delete[] h_a;
}

TEST(Reduce, Test_max_Global)
{
    int num = 10000;
    af::array a = af::randu(num, 1);
    float res = af::max<float>(a);
    float *h_a = a.host<float>();
    float gold = -std::numeric_limits<float>::max();

    for (int i = 0; i < num; i++) {
        gold = std::max(gold, h_a[i]);
    }

    ASSERT_EQ(gold, res);
    delete[] h_a;
}

TEST(Reduce, Test_All_Global)
{
    int num = 10000;
    af::array a = af::round(2 * af::randu(num, 1));

    char res = af::alltrue<char>(a);
    float *h_a = a.host<float>();
    char gold = 1;

    for (int i = 0; i < num; i++) {
        gold = gold && h_a[i];
    }

    ASSERT_EQ(gold, res);
    delete[] h_a;
}

TEST(Reduce, Test_Any_Global)
{
    int num = 10000;
    af::array a = af::round(2 * af::randu(num, 1));

    char res = af::anytrue<char>(a);
    float *h_a = a.host<float>();
    char gold = 0;

    for (int i = 0; i < num; i++) {
        gold = gold || h_a[i];
    }

    ASSERT_EQ(gold, res);
    delete[] h_a;
}
