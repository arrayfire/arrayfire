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
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>

using af::dim4;
using af::dtype_traits;
using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class MedianFilter : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

template<typename T>
class MedianFilter1d : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(MedianFilter, TestTypes);
TYPED_TEST_SUITE(MedianFilter1d, TestTypes);

template<typename T>
void medfiltTest(string pTestFile, dim_t w_len, dim_t w_wid,
                 af_border_type pad) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;

    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 dims         = numDims[0];
    af_array outArray = 0;
    af_array inArray  = 0;

    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS_CHECK_SUPRT(af_medfilt2(&outArray, inArray, w_len, w_wid, pad));

    vector<T> outData(dims.elements());

    ASSERT_SUCCESS(af_get_data_ptr((void*)outData.data(), outArray));

    vector<T> currGoldBar = tests[0];
    size_t nElems         = currGoldBar.size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }

    // cleanup
    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TYPED_TEST(MedianFilter, ZERO_PAD_3x3) {
    medfiltTest<TypeParam>(
        string(TEST_DIR "/medianfilter/zero_pad_3x3_window.test"), 3, 3,
        AF_PAD_ZERO);
}

TYPED_TEST(MedianFilter, SYMMETRIC_PAD_3x3) {
    medfiltTest<TypeParam>(
        string(TEST_DIR "/medianfilter/symmetric_pad_3x3_window.test"), 3, 3,
        AF_PAD_SYM);
}

TYPED_TEST(MedianFilter, BATCH_ZERO_PAD_3x3) {
    medfiltTest<TypeParam>(
        string(TEST_DIR "/medianfilter/batch_zero_pad_3x3_window.test"), 3, 3,
        AF_PAD_ZERO);
}

TYPED_TEST(MedianFilter, BATCH_SYMMETRIC_PAD_3x3) {
    medfiltTest<TypeParam>(
        string(TEST_DIR "/medianfilter/batch_symmetric_pad_3x3_window.test"), 3,
        3, AF_PAD_SYM);
}

template<typename T>
void medfilt1_Test(string pTestFile, dim_t w_wid, af_border_type pad) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;

    readTests<T, T, int>(pTestFile, numDims, in, tests);

    dim4 dims         = numDims[0];
    af_array outArray = 0;
    af_array inArray  = 0;

    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS_CHECK_SUPRT(af_medfilt1(&outArray, inArray, w_wid, pad));

    vector<T> outData(dims.elements());

    ASSERT_SUCCESS(af_get_data_ptr((void*)outData.data(), outArray));

    vector<T> currGoldBar = tests[0];
    size_t nElems         = currGoldBar.size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }

    // cleanup
    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TYPED_TEST(MedianFilter1d, ZERO_PAD_3) {
    medfilt1_Test<TypeParam>(
        string(TEST_DIR "/medianfilter/zero_pad_3x1_window.test"), 3,
        AF_PAD_ZERO);
}

TYPED_TEST(MedianFilter1d, SYMMETRIC_PAD_3) {
    medfilt1_Test<TypeParam>(
        string(TEST_DIR "/medianfilter/symmetric_pad_3x1_window.test"), 3,
        AF_PAD_SYM);
}

TYPED_TEST(MedianFilter1d, BATCH_ZERO_PAD_3) {
    medfilt1_Test<TypeParam>(
        string(TEST_DIR "/medianfilter/batch_zero_pad_3x1_window.test"), 3,
        AF_PAD_ZERO);
}

TYPED_TEST(MedianFilter1d, BATCH_SYMMETRIC_PAD_3) {
    medfilt1_Test<TypeParam>(
        string(TEST_DIR "/medianfilter/batch_symmetric_pad_3x1_window.test"), 3,
        AF_PAD_SYM);
}

template<typename T, bool isColor>
void medfiltImageTest(string pTestFile, dim_t w_len, dim_t w_wid) {
    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        af_array inArray   = 0;
        af_array outArray  = 0;
        af_array goldArray = 0;
        dim_t nElems       = 0;

        inFiles[testId].insert(0, string(TEST_DIR "/medianfilter/"));
        outFiles[testId].insert(0, string(TEST_DIR "/medianfilter/"));

        ASSERT_SUCCESS(
            af_load_image(&inArray, inFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(
            af_load_image(&goldArray, outFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(af_get_elements(&nElems, goldArray));

        ASSERT_SUCCESS_CHECK_SUPRT(
            af_medfilt2(&outArray, inArray, w_len, w_wid, AF_PAD_ZERO));

        ASSERT_IMAGES_NEAR(goldArray, outArray, 0.018f);

        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(outArray));
        ASSERT_SUCCESS(af_release_array(goldArray));
    }
}

template<typename T>
void medfiltInputTest(void) {
    SUPPORTED_TYPE_CHECK(T);

    af_array inArray  = 0;
    af_array outArray = 0;

    vector<T> in(100, 1);

    // Check for 1D inputs -> medfilt1
    dim4 dims = dim4(100, 1, 1, 1);

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS_CHECK_SUPRT(af_medfilt2(&outArray, inArray, 1, 1, AF_PAD_ZERO));

    bool medfilt1;
    ASSERT_SUCCESS(af_is_vector(&medfilt1, outArray));

    ASSERT_EQ(true, medfilt1);

    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TYPED_TEST(MedianFilter, InvalidArray) { medfiltInputTest<TypeParam>(); }

template<typename T>
void medfiltWindowTest(void) {
    SUPPORTED_TYPE_CHECK(T);

    af_array inArray  = 0;
    af_array outArray = 0;

    vector<T> in(100, 1);

    // Check for 4D inputs
    dim4 dims(10, 10, 1, 1);

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_medfilt2(&outArray, inArray, 3, 5, AF_PAD_ZERO));

    ASSERT_SUCCESS(af_release_array(inArray));
}

TYPED_TEST(MedianFilter, InvalidWindow) { medfiltWindowTest<TypeParam>(); }

template<typename T>
void medfilt1d_WindowTest(void) {
    SUPPORTED_TYPE_CHECK(T);

    af_array inArray  = 0;
    af_array outArray = 0;

    vector<T> in(100, 1);

    // Check for 4D inputs
    dim4 dims(10, 10, 1, 1);

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_EQ(AF_ERR_ARG, af_medfilt1(&outArray, inArray, -1, AF_PAD_ZERO));

    ASSERT_SUCCESS(af_release_array(inArray));
}

TYPED_TEST(MedianFilter1d, InvalidWindow) { medfilt1d_WindowTest<TypeParam>(); }

template<typename T>
void medfiltPadTest(void) {
    SUPPORTED_TYPE_CHECK(T);

    af_array inArray  = 0;
    af_array outArray = 0;

    vector<T> in(100, 1);

    // Check for 4D inputs
    dim4 dims(10, 10, 1, 1);

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_EQ(AF_ERR_ARG,
              af_medfilt2(&outArray, inArray, 3, 3, af_border_type(3)));

    ASSERT_EQ(AF_ERR_ARG,
              af_medfilt2(&outArray, inArray, 3, 3, af_border_type(-1)));

    ASSERT_SUCCESS(af_release_array(inArray));
}

TYPED_TEST(MedianFilter, InvalidPadType) { medfiltPadTest<TypeParam>(); }

template<typename T>
void medfilt1d_PadTest(void) {
    SUPPORTED_TYPE_CHECK(T);

    af_array inArray  = 0;
    af_array outArray = 0;

    vector<T> in(100, 1);

    // Check for 4D inputs
    dim4 dims(10, 10, 1, 1);

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_EQ(AF_ERR_ARG,
              af_medfilt1(&outArray, inArray, 3, af_border_type(3)));

    ASSERT_EQ(AF_ERR_ARG,
              af_medfilt1(&outArray, inArray, 3, af_border_type(-1)));

    ASSERT_SUCCESS(af_release_array(inArray));
}

TYPED_TEST(MedianFilter1d, InvalidPadType) { medfilt1d_PadTest<TypeParam>(); }

//////////////////////////////////// CPP ////////////////////////////////////
//

using af::array;

TEST(MedianFilter, CPP) {
    const dim_t w_len = 3;
    const dim_t w_wid = 3;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTests<float, float, int>(
        string(TEST_DIR "/medianfilter/batch_symmetric_pad_3x3_window.test"),
        numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));
    array output;
    try { output = medfilt(input, w_len, w_wid, AF_PAD_SYM); } catch FUNCTION_UNSUPPORTED

    vector<float> outData(dims.elements());
    output.host((void*)outData.data());

    vector<float> currGoldBar = tests[0];
    size_t nElems             = currGoldBar.size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }
}

TEST(MedianFilter1d, CPP) {
    const dim_t w_wid = 3;

    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTests<float, float, int>(
        string(TEST_DIR "/medianfilter/batch_symmetric_pad_3x1_window.test"),
        numDims, in, tests);

    dim4 dims = numDims[0];
    array input(dims, &(in[0].front()));
    array output;
    try { output = medfilt1(input, w_wid, AF_PAD_SYM); } catch FUNCTION_UNSUPPORTED

    vector<float> outData(dims.elements());
    output.host((void*)outData.data());

    vector<float> currGoldBar = tests[0];
    size_t nElems             = currGoldBar.size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }
}

TEST(MedianFilter, Docs) {
    float input[] = {1.0000,  2.0000,  3.0000,  4.0000,  5.0000,  6.0000,
                     7.0000,  8.0000,  9.0000,  10.0000, 11.0000, 12.0000,
                     13.0000, 14.0000, 15.0000, 16.0000};

    float gold[] = {0.0000, 2.0000,  3.0000,  0.0000,  2.0000,  6.0000,
                    7.0000, 4.0000,  6.0000,  10.0000, 11.0000, 8.0000,
                    0.0000, 10.0000, 11.0000, 0.0000};

    //![ex_image_medfilt]
    array a = array(4, 4, input);
    // af_print(a);
    // a = 1.0000        5.0000        9.0000       13.0000
    //    2.0000        6.0000       10.0000       14.0000
    //    3.0000        7.0000       11.0000       15.0000
    //    4.0000        8.0000       12.0000       16.0000
    array b;
    try { b = medfilt(a, 3, 3, AF_PAD_ZERO); } catch FUNCTION_UNSUPPORTED
    // af_print(b);
    // b=  0.0000        2.0000        6.0000        0.0000
    //    2.0000        6.0000       10.0000       10.0000
    //    3.0000        7.0000       11.0000       11.0000
    //    0.0000        4.0000        8.0000        0.0000
    //![ex_image_medfilt]

    float output[16];
    b.host((void*)output);

    for (int i = 0; i < 16; ++i) {
        ASSERT_EQ(output[i], gold[i]) << "output mismatch at i = " << i << endl;
    }
}

using af::constant;
using af::iota;
using af::max;
using af::medfilt;
using af::medfilt1;
using af::seq;
using af::span;

TEST(MedianFilter, GFOR) {
    dim4 dims = dim4(10, 10, 3);
    array A   = iota(dims);
    array B   = constant(0, dims);

    try {
        gfor(seq ii, 3) { B(span, span, ii) = medfilt(A(span, span, ii)); }
    } catch FUNCTION_UNSUPPORTED

    for (int ii = 0; ii < 3; ii++) {
        array c_ii;
        try { c_ii = medfilt(A(span, span, ii)); } catch FUNCTION_UNSUPPORTED
        array b_ii = B(span, span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}

TEST(MedianFilter1d, GFOR) {
    dim4 dims = dim4(10, 10, 3);
    array A   = iota(dims);
    array B   = constant(0, dims);

    try {
        gfor(seq ii, 3) { B(span, ii) = medfilt1(A(span, ii)); }
    } catch FUNCTION_UNSUPPORTED

    for (int ii = 0; ii < 3; ii++) {
        array c_ii;
        try { c_ii = medfilt1(A(span, ii)); } catch FUNCTION_UNSUPPORTED
        array b_ii = B(span, ii);
        ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
    }
}
