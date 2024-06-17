/*******************************************************
 * Copyright (c) 2017, ArrayFire
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
using std::endl;
using std::string;
using std::vector;

template<typename T>
class CannyEdgeDetector : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, int, uint, short, ushort, uchar, double>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(CannyEdgeDetector, TestTypes);

template<typename T>
void cannyTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<char>> tests;

    readTests<T, char, int>(pTestFile, numDims, in, tests);

    dim4 sDims        = numDims[0];
    af_array outArray = 0;
    af_array sArray   = 0;

    ASSERT_SUCCESS(af_create_array(&sArray, &(in[0].front()), sDims.ndims(),
                                   sDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS_CHECK_SUPRT(af_canny(&outArray, sArray, AF_CANNY_THRESHOLD_MANUAL,
                                        0.4147f, 0.8454f, 3, true));

    vector<char> outData(sDims.elements());

    ASSERT_SUCCESS(af_get_data_ptr((void*)outData.data(), outArray));

    vector<char> currGoldBar = tests[0];
    size_t nElems            = currGoldBar.size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(currGoldBar[elIter], outData[elIter])
            << "at: " << elIter << endl;
    }

    // cleanup
    ASSERT_SUCCESS(af_release_array(sArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TYPED_TEST(CannyEdgeDetector, ArraySizeLessThanBlockSize10x10) {
    cannyTest<TypeParam>(string(TEST_DIR "/CannyEdgeDetector/fast10x10.test"));
}

TYPED_TEST(CannyEdgeDetector, ArraySizeEqualBlockSize16x16) {
    cannyTest<TypeParam>(string(TEST_DIR "/CannyEdgeDetector/fast16x16.test"));
}

TEST(Canny, DISABLED_Exact) {
    using namespace af;
    array img = loadImage(TEST_DIR "/CannyEdgeDetector/woman.jpg", false);

    array out = canny(img, AF_CANNY_THRESHOLD_AUTO_OTSU, 0.08, 0.32, 3, false);
    array gold =
        loadImage(TEST_DIR "/CannyEdgeDetector/woman_edges.jpg", false) > 3;

    ASSERT_ARRAYS_EQ(gold, out);
}

template<typename T>
void cannyImageOtsuTest(string pTestFile, bool isColor) {
    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    using af::dim4;

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        af_array _inArray  = 0;
        af_array inArray   = 0;
        af_array _outArray = 0;
        af_array cstArray  = 0;
        af_array mulArray  = 0;
        af_array outArray  = 0;
        af_array goldArray = 0;

        inFiles[testId].insert(0, string(TEST_DIR "/CannyEdgeDetector/"));
        outFiles[testId].insert(0, string(TEST_DIR "/CannyEdgeDetector/"));

        af_dtype type = (af_dtype)dtype_traits<T>::af_type;

        ASSERT_SUCCESS(
            af_load_image(&_inArray, inFiles[testId].c_str(), isColor));

        ASSERT_SUCCESS(af_cast(&inArray, _inArray, type));

        ASSERT_SUCCESS(
            af_load_image_native(&goldArray, outFiles[testId].c_str()));

        ASSERT_SUCCESS_CHECK_SUPRT(af_canny(&_outArray, inArray,
                                            AF_CANNY_THRESHOLD_AUTO_OTSU,
                                            0.08, 0.32, 3, false));

        unsigned ndims = 0;
        dim_t dims[4];

        ASSERT_SUCCESS(af_get_numdims(&ndims, _outArray));
        ASSERT_SUCCESS(
            af_get_dims(dims, dims + 1, dims + 2, dims + 3, _outArray));

        ASSERT_SUCCESS(af_constant(&cstArray, 255.0, ndims, dims, f32));

        ASSERT_SUCCESS(af_mul(&mulArray, cstArray, _outArray, false));
        ASSERT_SUCCESS(af_cast(&outArray, mulArray, u8));

        ASSERT_IMAGES_NEAR(goldArray, outArray, 1.0e-3);

        ASSERT_SUCCESS(af_release_array(_inArray));
        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(cstArray));
        ASSERT_SUCCESS(af_release_array(mulArray));
        ASSERT_SUCCESS(af_release_array(_outArray));
        ASSERT_SUCCESS(af_release_array(outArray));
        ASSERT_SUCCESS(af_release_array(goldArray));
    }
}

TEST(CannyEdgeDetector, OtsuThreshold) {
    cannyImageOtsuTest<float>(string(TEST_DIR "/CannyEdgeDetector/gray.test"),
                              false);
}

TEST(CannyEdgeDetector, InvalidSizeArray) {
    af_array inArray  = 0;
    af_array outArray = 0;

    vector<float> in(100, 1);

    dim4 sDims(100, 1, 1, 1);

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), sDims.ndims(),
                                   sDims.get(),
                                   (af_dtype)dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_SIZE,
              af_canny(&outArray, inArray, AF_CANNY_THRESHOLD_MANUAL, 0.24,
                       0.72, 3, true));

    ASSERT_SUCCESS(af_release_array(inArray));
}

TEST(CannyEdgeDetector, Array4x4_Invalid) {
    af_array inArray  = 0;
    af_array outArray = 0;

    vector<float> in(16, 1);

    dim4 sDims(4, 4, 1, 1);

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), sDims.ndims(),
                                   sDims.get(),
                                   (af_dtype)dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_SIZE,
              af_canny(&outArray, inArray, AF_CANNY_THRESHOLD_MANUAL, 0.24,
                       0.72, 3, true));

    ASSERT_SUCCESS(af_release_array(inArray));
}

TEST(CannyEdgeDetector, Sobel5x5_Invalid) {
    af_array inArray  = 0;
    af_array outArray = 0;

    vector<float> in(25, 1);

    dim4 sDims(5, 5, 1, 1);

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), sDims.ndims(),
                                   sDims.get(),
                                   (af_dtype)dtype_traits<float>::af_type));

    ASSERT_EQ(AF_ERR_ARG,
              af_canny(&outArray, inArray, AF_CANNY_THRESHOLD_MANUAL, 0.24,
                       0.72, 5, true));

    ASSERT_SUCCESS(af_release_array(inArray));
}

template<typename T>
void cannyImageOtsuBatchTest(string pTestFile, const dim_t targetBatchCount) {
    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    using af::array;
    using af::canny;
    using af::loadImage;
    using af::loadImageNative;
    using af::tile;

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        inFiles[testId].insert(0, string(TEST_DIR "/CannyEdgeDetector/"));
        outFiles[testId].insert(0, string(TEST_DIR "/CannyEdgeDetector/"));

        af_dtype type  = (af_dtype)dtype_traits<T>::af_type;
        array readGold = loadImageNative(outFiles[testId].c_str());
        array goldIm   = tile(readGold, 1, 1, targetBatchCount);
        array readImg  = loadImage(inFiles[testId].c_str(), false).as(type);
        array inputIm  = tile(readImg, 1, 1, targetBatchCount);

        array outIm;
        try { outIm =
              canny(inputIm, AF_CANNY_THRESHOLD_AUTO_OTSU, 0.08, 0.32, 3, false);
        } catch FUNCTION_UNSUPPORTED
        outIm *= 255.0;

        ASSERT_IMAGES_NEAR(goldIm, outIm.as(u8), 1.0e-3);
    }
}

TEST(CannyEdgeDetector, BatchofImagesUsingCPPAPI) {
    // DO NOT INCREASE BATCH COUNT BEYOND 4
    // This is a limitation on the test assert macro that is saving
    // images to disk which can't handle a batch of images.
    cannyImageOtsuBatchTest<float>(
        string(TEST_DIR "/CannyEdgeDetector/gray.test"), 3);
}
