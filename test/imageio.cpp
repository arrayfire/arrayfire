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
class ImageIO : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

typedef ::testing::Types<float> TestTypes;

// register the type list
TYPED_TEST_CASE(ImageIO, TestTypes);

// Disable tests if FreeImage is not found
#if defined(WITH_FREEIMAGE)
void loadImageTest(string pTestFile, string pImageFile, const bool isColor)
{
    if (noDoubleTests<float>()) return;

    vector<af::dim4> numDims;

    vector<vector<float> >   in;
    vector<vector<float> >   tests;
    readTests<float, float, float>(pTestFile,numDims,in,tests);
    af::dim4 dims       = numDims[0];

    af_array imgArray = 0;
    ASSERT_EQ(AF_SUCCESS, af_load_image(&imgArray, pImageFile.c_str(), isColor));

    // Get result
    float *imgData = new float[dims.elements()];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*) imgData, imgArray));

    // Compare result
    size_t nElems = in[0].size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(in[0][elIter], imgData[elIter]) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] imgData;

    if(imgArray != 0) af_release_array(imgArray);
}

TYPED_TEST(ImageIO, ColorSmall)
{
    loadImageTest(string(TEST_DIR"/imageio/color_small.test"), string(TEST_DIR"/imageio/color_small.png"), true);
}

TYPED_TEST(ImageIO, GraySmall)
{
    loadImageTest(string(TEST_DIR"/imageio/gray_small.test"), string(TEST_DIR"/imageio/gray_small.jpg"), false);
}

TYPED_TEST(ImageIO, GraySeq)
{
    loadImageTest(string(TEST_DIR"/imageio/gray_seq.test"), string(TEST_DIR"/imageio/gray_seq.png"), false);
}

TYPED_TEST(ImageIO, ColorSeq)
{
    loadImageTest(string(TEST_DIR"/imageio/color_seq.test"), string(TEST_DIR"/imageio/color_seq.png"), true);
}

void loadimageArgsTest(string pImageFile, const bool isColor, af_err err)
{
    af_array imgArray = 0;

    ASSERT_EQ(err, af_load_image(&imgArray, pImageFile.c_str(), isColor));

    if(imgArray != 0) af_release_array(imgArray);
}

TYPED_TEST(ImageIO,InvalidArgsMissingFile)
{
    loadimageArgsTest(string(TEST_DIR"/imageio/nofile.png"), false, AF_ERR_RUNTIME);
}

TYPED_TEST(ImageIO,InvalidArgsWrongExt)
{
    loadimageArgsTest(string(TEST_DIR"/imageio/image.wrongext"), true, AF_ERR_NOT_SUPPORTED);
}

////////////////////////////////// CPP //////////////////////////////////////
TEST(ImageIO, CPP)
{
    if (noDoubleTests<float>()) return;

    vector<af::dim4> numDims;

    vector<vector<float> >   in;
    vector<vector<float> >   tests;
    readTests<float, float, float>(string(TEST_DIR"/imageio/color_small.test"),numDims,in,tests);

    af::dim4 dims = numDims[0];
    af::array img = af::loadImage(string(TEST_DIR"/imageio/color_small.png").c_str(), true);

    // Get result
    float *imgData = new float[dims.elements()];
    img.host((void*)imgData);

    // Compare result
    size_t nElems = in[0].size();
    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_EQ(in[0][elIter], imgData[elIter]) << "at: " << elIter << std::endl;
    }

    // Delete
    delete[] imgData;
}

TEST(ImageIO, SavePNGCPP) {

    af::array input(10, 10, 3, f32);

    input(af::span, af::span, af::span) = 0;
    input(0, 0, 0) = 255;
    input(0, 9, 1) = 255;
    input(9, 0, 2) = 255;
    input(9, 9, af::span) = 255;

    saveImage("SaveCPP.png", input);
    af::array out = af::loadImage("SaveCPP.png", true);

    ASSERT_FALSE(af::anyTrue<bool>(out - input));
}

TEST(ImageIO, SaveBMPCPP) {

    af::array input(10, 10, 3, f32);

    input(af::span, af::span, af::span) = 0;
    input(0, 0, 0) = 255;
    input(0, 9, 1) = 255;
    input(9, 0, 2) = 255;
    input(9, 9, af::span) = 255;

    saveImage("SaveCPP.bmp", input);
    af::array out = af::loadImage("SaveCPP.bmp", true);

    ASSERT_FALSE(af::anyTrue<bool>(out - input));
}

TEST(ImageMem, SaveMemPNG)
{
    if (noDoubleTests<float>()) return;

    af::array img = af::loadImage(string(TEST_DIR"/imageio/color_seq.png").c_str(), true);

    void* savedMem = af::saveImageMem(img, AF_FIF_PNG);

    af::array loadMem = af::loadImageMem(savedMem);

    ASSERT_FALSE(af::anyTrue<bool>(img - loadMem));

    af::deleteImageMem(savedMem);
}

TEST(ImageMem, SaveMemJPG1)
{
    if (noDoubleTests<float>()) return;

    af::array img = af::loadImage(string(TEST_DIR"/imageio/color_seq.png").c_str(), false);
    af::saveImage("color_seq1.jpg", img);

    void* savedMem = af::saveImageMem(img, AF_FIF_JPEG);

    af::array loadMem = af::loadImageMem(savedMem);
    af::array imgJPG = af::loadImage("color_seq1.jpg", false);

    ASSERT_FALSE(af::anyTrue<bool>(imgJPG - loadMem));

    af::deleteImageMem(savedMem);
}

TEST(ImageMem, SaveMemJPG3)
{
    if (noDoubleTests<float>()) return;

    af::array img = af::loadImage(string(TEST_DIR"/imageio/color_seq.png").c_str(), true);
    af::saveImage("color_seq3.jpg", img);

    void* savedMem = af::saveImageMem(img, AF_FIF_JPEG);

    af::array loadMem = af::loadImageMem(savedMem);
    af::array imgJPG = af::loadImage("color_seq3.jpg", true);

    ASSERT_FALSE(af::anyTrue<bool>(imgJPG - loadMem));

    af::deleteImageMem(savedMem);
}

TEST(ImageMem, SaveMemBMP)
{
    if (noDoubleTests<float>()) return;

    af::array img = af::loadImage(string(TEST_DIR"/imageio/color_rand.png").c_str(), true);

    void* savedMem = af::saveImageMem(img, AF_FIF_BMP);

    af::array loadMem = af::loadImageMem(savedMem);

    ASSERT_FALSE(af::anyTrue<bool>(img - loadMem));

    af::deleteImageMem(savedMem);
}

#endif // WITH_FREEIMAGE
