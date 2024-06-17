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
#include <af/data.h>
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
class Morph : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, char, uchar, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Morph, TestTypes);

template<typename inType, bool isDilation, bool isVolume>
void morphTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(inType);

    vector<dim4> numDims;
    vector<vector<inType>> in;
    vector<vector<inType>> tests;

    readTests<inType, inType, int>(pTestFile, numDims, in, tests);

    dim4 dims          = numDims[0];
    dim4 maskDims      = numDims[1];
    af_array outArray  = 0;
    af_array inArray   = 0;
    af_array maskArray = 0;

    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<inType>::af_type));
    ASSERT_SUCCESS(af_create_array(&maskArray, &(in[1].front()),
                                   maskDims.ndims(), maskDims.get(),
                                   (af_dtype)dtype_traits<inType>::af_type));

    af_err af_stat;
    if (isDilation) {
        if (isVolume) {
            ASSERT_SUCCESS_CHECK_SUPRT(af_dilate3(&outArray, inArray, maskArray));
        } else {
            ASSERT_SUCCESS_CHECK_SUPRT(af_dilate(&outArray, inArray, maskArray));
        }
    } else {
        if (isVolume) {
            ASSERT_SUCCESS_CHECK_SUPRT(af_erode3(&outArray, inArray, maskArray));
        } else {
            ASSERT_SUCCESS_CHECK_SUPRT(af_erode(&outArray, inArray, maskArray));
        }
    }

    for (size_t testIter = 0; testIter < tests.size(); ++testIter) {
        vector<inType> currGoldBar = tests[testIter];
        ASSERT_VEC_ARRAY_EQ(currGoldBar, dims, outArray);
    }

    // cleanup
    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(maskArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

TYPED_TEST(Morph, Dilate3x3) {
    morphTest<TypeParam, true, false>(string(TEST_DIR "/morph/dilate3x3.test"));
}

TYPED_TEST(Morph, Erode3x3) {
    morphTest<TypeParam, false, false>(string(TEST_DIR "/morph/erode3x3.test"));
}

TYPED_TEST(Morph, Dilate4x4) {
    morphTest<TypeParam, true, false>(string(TEST_DIR "/morph/dilate4x4.test"));
}

TYPED_TEST(Morph, Dilate12x12) {
    morphTest<TypeParam, true, false>(
        string(TEST_DIR "/morph/dilate12x12.test"));
}

TYPED_TEST(Morph, Erode4x4) {
    morphTest<TypeParam, false, false>(string(TEST_DIR "/morph/erode4x4.test"));
}

TYPED_TEST(Morph, Dilate3x3_Batch) {
    morphTest<TypeParam, true, false>(
        string(TEST_DIR "/morph/dilate3x3_batch.test"));
}

TYPED_TEST(Morph, Erode3x3_Batch) {
    morphTest<TypeParam, false, false>(
        string(TEST_DIR "/morph/erode3x3_batch.test"));
}

TYPED_TEST(Morph, Dilate3x3x3) {
    morphTest<TypeParam, true, true>(
        string(TEST_DIR "/morph/dilate3x3x3.test"));
}

TYPED_TEST(Morph, Erode3x3x3) {
    morphTest<TypeParam, false, true>(
        string(TEST_DIR "/morph/erode3x3x3.test"));
}

TYPED_TEST(Morph, Dilate4x4x4) {
    morphTest<TypeParam, true, true>(
        string(TEST_DIR "/morph/dilate4x4x4.test"));
}

TYPED_TEST(Morph, Erode4x4x4) {
    morphTest<TypeParam, false, true>(
        string(TEST_DIR "/morph/erode4x4x4.test"));
}

template<typename T, bool isDilation, bool isColor>
void morphImageTest(string pTestFile, dim_t seLen) {
    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        af_array _inArray   = 0;
        af_array inArray    = 0;
        af_array maskArray  = 0;
        af_array outArray   = 0;
        af_array _goldArray = 0;
        af_array goldArray  = 0;
        dim_t nElems        = 0;

        inFiles[testId].insert(0, string(TEST_DIR "/morph/"));
        outFiles[testId].insert(0, string(TEST_DIR "/morph/"));

        af_dtype targetType = static_cast<af_dtype>(dtype_traits<T>::af_type);

        dim4 mdims(seLen, seLen, 1, 1);
        ASSERT_SUCCESS(af_constant(&maskArray, 1.0, mdims.ndims(), mdims.get(),
                                   targetType));

        ASSERT_SUCCESS(
            af_load_image(&_inArray, inFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(af_cast(&inArray, _inArray, targetType));

        ASSERT_SUCCESS(
            af_load_image(&_goldArray, outFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(af_cast(&goldArray, _goldArray, targetType));

        ASSERT_SUCCESS(af_get_elements(&nElems, goldArray));

        af_err error_code = AF_SUCCESS;
        try {
            if (isDilation) {
                error_code = af_dilate(&outArray, inArray, maskArray);
            } else {
                error_code = af_erode(&outArray, inArray, maskArray);
            }
        } catch FUNCTION_UNSUPPORTED

#if defined(AF_CPU)
        ASSERT_SUCCESS_CHECK_SUPRT(error_code);
        ASSERT_IMAGES_NEAR(goldArray, outArray, 0.018f);
#else
        if (targetType != b8 && seLen > 19) {
            ASSERT_EQ(error_code, AF_ERR_NOT_SUPPORTED);
        } else {
            ASSERT_SUCCESS_CHECK_SUPRT(error_code);
            ASSERT_IMAGES_NEAR(goldArray, outArray, 0.018f);
        }
#endif

        ASSERT_SUCCESS(af_release_array(_inArray));
        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(maskArray));
        ASSERT_SUCCESS(af_release_array(outArray));
        ASSERT_SUCCESS(af_release_array(_goldArray));
        ASSERT_SUCCESS(af_release_array(goldArray));
    }
}

TEST(Morph, GrayscaleDilation3x3StructuringElement) {
    morphImageTest<float, true, false>(string(TEST_DIR "/morph/gray.test"), 3);
}

TEST(Morph, ColorImageErosion3x3StructuringElement) {
    morphImageTest<float, false, true>(string(TEST_DIR "/morph/color.test"), 3);
}

TEST(Morph, BinaryImageDilationBy33x33Kernel) {
    morphImageTest<char, true, false>(
        string(TEST_DIR "/morph/zag_dilation.test"), 33);
}

TEST(Morph, BinaryImageErosionBy33x33Kernel) {
    morphImageTest<char, false, false>(
        string(TEST_DIR "/morph/zag_erosion.test"), 33);
}

TEST(Morph, DilationBy33x33Kernel) {
    morphImageTest<float, true, true>(
        string(TEST_DIR "/morph/baboon_dilation.test"), 33);
}

TEST(Morph, ErosionBy33x33Kernel) {
    morphImageTest<float, false, true>(
        string(TEST_DIR "/morph/baboon_erosion.test"), 33);
}

template<typename T, bool isDilation>
void morphInputTest(void) {
    SUPPORTED_TYPE_CHECK(T);

    af_array inArray   = 0;
    af_array maskArray = 0;
    af_array outArray  = 0;

    vector<T> in(100, 1);
    vector<T> mask(9, 1);

    // Check for 1D inputs
    dim4 dims = dim4(100, 1, 1, 1);
    dim4 mdims(3, 3, 1, 1);

    ASSERT_SUCCESS(af_create_array(&maskArray, &mask.front(), mdims.ndims(),
                                   mdims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    if (isDilation)
        ASSERT_EQ(AF_ERR_SIZE, af_dilate(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(AF_ERR_SIZE, af_erode(&outArray, inArray, maskArray));

    ASSERT_SUCCESS(af_release_array(inArray));

    ASSERT_SUCCESS(af_release_array(maskArray));
}

TYPED_TEST(Morph, DilateInvalidInput) { morphInputTest<TypeParam, true>(); }

TYPED_TEST(Morph, ErodeInvalidInput) { morphInputTest<TypeParam, false>(); }

template<typename T, bool isDilation>
void morphMaskTest(void) {
    SUPPORTED_TYPE_CHECK(T);

    af_array inArray   = 0;
    af_array maskArray = 0;
    af_array outArray  = 0;

    vector<T> in(100, 1);
    vector<T> mask(16, 1);

    // Check for 4D mask
    dim4 dims(10, 10, 1, 1);
    dim4 mdims(2, 2, 2, 2);

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&maskArray, &mask.front(), mdims.ndims(),
                                   mdims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    if (isDilation)
        ASSERT_EQ(AF_ERR_SIZE, af_dilate(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(AF_ERR_SIZE, af_erode(&outArray, inArray, maskArray));

    ASSERT_SUCCESS(af_release_array(maskArray));

    // Check for 1D mask
    mdims = dim4(16, 1, 1, 1);

    ASSERT_SUCCESS(af_create_array(&maskArray, &mask.front(), mdims.ndims(),
                                   mdims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    if (isDilation)
        ASSERT_EQ(AF_ERR_SIZE, af_dilate(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(AF_ERR_SIZE, af_erode(&outArray, inArray, maskArray));

    ASSERT_SUCCESS(af_release_array(maskArray));

    ASSERT_SUCCESS(af_release_array(inArray));
}

TYPED_TEST(Morph, DilateInvalidMask) { morphMaskTest<TypeParam, true>(); }

TYPED_TEST(Morph, ErodeInvalidMask) { morphMaskTest<TypeParam, false>(); }

template<typename T, bool isDilation>
void morph3DMaskTest(void) {
    SUPPORTED_TYPE_CHECK(T);

    af_array inArray   = 0;
    af_array maskArray = 0;
    af_array outArray  = 0;

    vector<T> in(1000, 1);
    vector<T> mask(81, 1);

    // Check for 2D mask
    dim4 dims(10, 10, 10, 1);
    dim4 mdims(9, 9, 1, 1);

    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_create_array(&maskArray, &mask.front(), mdims.ndims(),
                                   mdims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    if (isDilation)
        ASSERT_EQ(AF_ERR_SIZE, af_dilate3(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(AF_ERR_SIZE, af_erode3(&outArray, inArray, maskArray));

    ASSERT_SUCCESS(af_release_array(maskArray));

    // Check for 4D mask
    mdims = dim4(3, 3, 3, 3);

    ASSERT_SUCCESS(af_create_array(&maskArray, &mask.front(), mdims.ndims(),
                                   mdims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    if (isDilation)
        ASSERT_EQ(AF_ERR_SIZE, af_dilate3(&outArray, inArray, maskArray));
    else
        ASSERT_EQ(AF_ERR_SIZE, af_erode3(&outArray, inArray, maskArray));

    ASSERT_SUCCESS(af_release_array(maskArray));

    ASSERT_SUCCESS(af_release_array(inArray));
}

TYPED_TEST(Morph, DilateVolumeInvalidMask) {
    morph3DMaskTest<TypeParam, true>();
}

TYPED_TEST(Morph, ErodeVolumeInvalidMask) {
    morph3DMaskTest<TypeParam, false>();
}

////////////////////////////////////// CPP //////////////////////////////////
//

using af::array;
using af::constant;
using af::erode;
using af::iota;
using af::loadImage;
using af::max;
using af::randu;
using af::seq;
using af::span;

template<typename T, bool isDilation, bool isColor>
void cppMorphImageTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        inFiles[testId].insert(0, string(TEST_DIR "/morph/"));
        outFiles[testId].insert(0, string(TEST_DIR "/morph/"));

        array mask   = constant(1.0, 3, 3);
        array img    = loadImage(inFiles[testId].c_str(), isColor);
        array gold   = loadImage(outFiles[testId].c_str(), isColor);
        dim_t nElems = gold.elements();
        array output;

        try {
            if (isDilation)
                output = dilate(img, mask);
            else
                output = erode(img, mask);
        } catch FUNCTION_UNSUPPORTED

        vector<T> outData(nElems);
        output.host((void*)outData.data());

        vector<T> goldData(nElems);
        gold.host((void*)goldData.data());

        ASSERT_EQ(true, compareArraysRMSD(nElems, goldData.data(),
                                          outData.data(), 0.018f));
    }
}

TEST(Morph, Grayscale_CPP) {
    cppMorphImageTest<float, true, false>(string(TEST_DIR "/morph/gray.test"));
}

TEST(Morph, ColorImage_CPP) {
    cppMorphImageTest<float, false, true>(string(TEST_DIR "/morph/color.test"));
}

TEST(Morph, GFOR) {
    dim4 dims  = dim4(10, 10, 3);
    array A    = iota(dims);
    array B    = constant(0, dims);
    array mask = randu(3, 3) > 0.3;

    try {
        gfor(seq ii, 3) { B(span, span, ii) = erode(A(span, span, ii), mask); }
    
        for (int ii = 0; ii < 3; ii++) {
            array c_ii = erode(A(span, span, ii), mask);
            array b_ii = B(span, span, ii);
            ASSERT_EQ(max<double>(abs(c_ii - b_ii)) < 1E-5, true);
        }
    } catch FUNCTION_UNSUPPORTED
}

TEST(Morph, EdgeIssue1564) {
    int inputData[10 * 10] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                              0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    int goldData[10 * 10]  = {0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                              0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1,
                              1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                              1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    array input(10, 10, inputData);
    int maskData[3 * 3] = {1, 1, 1, 1, 0, 1, 1, 1, 1};
    array mask(3, 3, maskData);
    try {
        array dilated = dilate(input.as(b8), mask.as(b8));

        size_t nElems = dilated.elements();
        vector<char> outData(nElems);
        dilated.host((void*)outData.data());
    
        for (size_t i = 0; i < nElems; ++i) {
            ASSERT_EQ((int)outData[i], goldData[i]);
        }
    } catch FUNCTION_UNSUPPORTED
}

TEST(Morph, UnsupportedKernel2D) {
    const unsigned ndims = 2;
    const dim_t dims[2]  = {10, 10};
    const dim_t kdims[2] = {32, 32};

    af_array in, mask, out;

    ASSERT_SUCCESS(af_constant(&mask, 1.0, ndims, kdims, f32));
    ASSERT_SUCCESS(af_randu(&in, ndims, dims, f32));

#if defined(AF_CPU)
    ASSERT_SUCCESS(af_dilate(&out, in, mask));
    ASSERT_SUCCESS(af_release_array(out));
#else
    ASSERT_EQ(AF_ERR_NOT_SUPPORTED, af_dilate(&out, in, mask));
#endif
    ASSERT_SUCCESS(af_release_array(in));
    ASSERT_SUCCESS(af_release_array(mask));
}
