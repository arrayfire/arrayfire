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
#include <string>
#include <vector>
#include <iostream>
#include <testHelpers.hpp>

using std::string;
using std::vector;

template<typename T>
class Histogram : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(Histogram, TestTypes);

template<typename inType, typename outType>
void histTest(string pTestFile, unsigned nbins, double minval, double maxval)
{
    if (noDoubleTests<inType>()) return;
    if (noDoubleTests<outType>()) return;

    vector<af::dim4> numDims;

    vector<vector<inType> >  in;
    vector<vector<outType> > tests;
    readTests<inType,uint,int>(pTestFile,numDims,in,tests);
    af::dim4 dims       = numDims[0];

    af_array outArray   = 0;
    af_array inArray    = 0;
    outType *outData;
    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()), dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<inType>::af_type));

    ASSERT_EQ(AF_SUCCESS,af_histogram(&outArray,inArray,nbins,minval,maxval));

    outData = new outType[dims.elements()];

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    for (size_t testIter=0; testIter<tests.size(); ++testIter) {
        vector<outType> currGoldBar = tests[testIter];
        size_t nElems        = currGoldBar.size();
        for (size_t elIter=0; elIter<nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter],outData[elIter])<< "at: " << elIter<< std::endl;
        }
    }

    // cleanup
    delete[] outData;
    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
}

TYPED_TEST(Histogram,256Bins0min255max_ones)
{
    histTest<TypeParam,uint>(string(TEST_DIR"/histogram/256bin1min1max.test"),256,0,255);
}

TYPED_TEST(Histogram,100Bins0min99max)
{
    histTest<TypeParam,uint>(string(TEST_DIR"/histogram/100bin0min99max.test"),100,0,99);
}

TYPED_TEST(Histogram,40Bins0min100max)
{
    histTest<TypeParam,uint>(string(TEST_DIR"/histogram/40bin0min100max.test"),40,0,100);
}

TYPED_TEST(Histogram,40Bins0min100max_Batch)
{
    histTest<TypeParam,uint>(string(TEST_DIR"/histogram/40bin0min100max_batch.test"),40,0,100);
}

TYPED_TEST(Histogram,256Bins0min255max_zeros)
{
    histTest<TypeParam,uint>(string(TEST_DIR"/histogram/256bin0min0max.test"),256,0,255);
}

/////////////////////////////////// CPP //////////////////////////////////
//
TEST(Histogram, CPP)
{
    if (noDoubleTests<float>()) return;
    if (noDoubleTests<int>()) return;

    const unsigned nbins = 100;
    const double minval = 0.0;
    const double maxval = 99.0;

    vector<af::dim4> numDims;

    vector<vector<float> >  in;
    vector<vector<uint> > tests;
    readTests<float,uint,int>(string(TEST_DIR"/histogram/100bin0min99max.test"),numDims,in,tests);

//! [hist_nominmax]
    af::array input(numDims[0], &(in[0].front()));
    af::array output = histogram(input, nbins, minval, maxval);
//! [hist_nominmax]

    uint *outData = new uint[output.elements()];
    output.host((void*)outData);

    for (size_t testIter=0; testIter<tests.size(); ++testIter) {
        vector<uint> currGoldBar = tests[testIter];
        size_t nElems        = currGoldBar.size();
        for (size_t elIter=0; elIter<nElems; ++elIter) {
            ASSERT_EQ(currGoldBar[elIter],outData[elIter])<< "at: " << elIter<< std::endl;
        }
    }

    // cleanup
    delete[] outData;
}

/////////////////////////////////// Documentation Snippets //////////////////////////////////
//
TEST(Histogram, SNIPPET_hist_nominmax)
{
    using af::array;
    using af::histogram;
    using std::ostream_iterator;
    using std::cout;
    using std::endl;

    unsigned output[] = {3, 1, 2, 0, 0, 0, 0, 1, 1, 1};

    //! [ex_image_hist_nominmax]
    float input[]  = {1, 2, 1, 1, 3, 6, 7, 8, 3};
    int nbins = 10;

    size_t nElems = sizeof(input)/sizeof(float);
    array hist_in(nElems, input);

    array hist_out = histogram(hist_in, nbins);
    // hist_out = {3, 1, 2, 0, 0, 0, 0, 1, 1, 1}
    //! [ex_image_hist_nominmax]

    vector<unsigned> h_out(nbins);
    hist_out.host((void*)h_out.data());

    if( false == equal(h_out.begin(), h_out.end(), output) ) {
        cout << "Expected: ";
        copy(output, output + nbins, ostream_iterator<unsigned>(cout, ", "));
        cout << endl << "Actual: ";
        copy(h_out.begin(), h_out.end(), ostream_iterator<unsigned>(cout, ", "));
        FAIL() << "Output did not match";
    }
}

TEST(Histogram, SNIPPET_hist_minmax)
{
    using af::array;
    using af::histogram;
    using std::ostream_iterator;
    using std::cout;
    using std::endl;

    unsigned output[] = {0, 3, 1, 2, 0, 0, 1, 1, 1, 0};

    //! [ex_image_hist_minmax]
    float input[]  = {1, 2, 1, 1, 3, 6, 7, 8, 3};
    int nbins = 10;

    size_t nElems = sizeof(input)/sizeof(float);
    array hist_in(nElems, input);

    array hist_out = histogram(hist_in, nbins, 0, 9);
    // hist_out = {0, 3, 1, 2, 0, 0, 1, 1, 1, 0}
    //! [ex_image_hist_minmax]

    vector<unsigned> h_out(nbins);
    hist_out.host((void*)h_out.data());

    if( false == equal(h_out.begin(), h_out.end(), output) ) {
        cout << "Expected: ";
        copy(output, output + nbins, ostream_iterator<unsigned>(cout, ", "));
        cout << endl << "Actual: ";
        copy(h_out.begin(), h_out.end(), ostream_iterator<unsigned>(cout, ", "));
        FAIL() << "Output did not match";
    }
}

TEST(Histogram, SNIPPET_histequal)
{
    using af::array;
    using af::histogram;
    using std::ostream_iterator;
    using std::cout;
    using std::endl;

    float output[] = { 1.5, 4.5,  1.5, 1.5, 4.5, 4.5, 6.0, 7.5, 4.5 };

    //! [ex_image_histequal]
    float input[]  = {1, 2, 1, 1, 3, 6, 7, 8, 3};
    int nbins = 10;

    size_t nElems = sizeof(input)/sizeof(float);
    array hist_in(nElems, input);

    array hist_out = histogram(hist_in, nbins);

    // input after histogram equalization or normalization
    // based on histogram provided
    array eq_out = histEqual(hist_in, hist_out);
    // eq_out = { 1.5, 4.5,  1.5, 1.5, 4.5, 4.5, 6.0, 7.5, 4.5 }
    //! [ex_image_histequal]

    vector<float> h_out(nElems);
    eq_out.host((void*)h_out.data());

    if( false == equal(h_out.begin(), h_out.end(), output) ) {
        cout << "Expected: ";
        copy(output, output + nbins, ostream_iterator<float>(cout, ", "));
        cout << endl << "Actual: ";
        copy(h_out.begin(), h_out.end(), ostream_iterator<float>(cout, ", "));
        FAIL() << "Output did not match";
    }
}

TEST(histogram, GFOR)
{
    using namespace af;

    dim4 dims = dim4(100, 100, 3);
    array A = round(100 * randu(dims));
    array B = constant(0, 100, 1, 3);

    gfor(seq ii, 3) {
        B(span, span, ii) = histogram(A(span, span, ii), 100);
    }

    for(int ii = 0; ii < 3; ii++) {
        array c_ii = histogram(A(span, span, ii), 100);
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}
