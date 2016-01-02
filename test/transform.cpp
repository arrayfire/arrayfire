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
using std::abs;
using std::cout;
using std::endl;

template<typename T>
class Transform : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

template<typename T>
class TransformInt : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

typedef ::testing::Types<float, double> TestTypes;
typedef ::testing::Types<int, intl, uint, uintl, short, ushort, uchar> TestTypesInt;

TYPED_TEST_CASE(Transform, TestTypes);
TYPED_TEST_CASE(TransformInt, TestTypesInt);

template<typename T>
void transformTest(string pTestFile, string pHomographyFile, const af_interp_type method, const bool invert)
{
    if (noDoubleTests<T>()) return;

    vector<af::dim4> inNumDims;
    vector<string>   inFiles;
    vector<dim_t>    goldNumDims;
    vector<string>   goldFiles;

    readImageTests(pTestFile, inNumDims, inFiles, goldNumDims, goldFiles);

    inFiles[0].insert(0,string(TEST_DIR"/transform/"));
    inFiles[1].insert(0,string(TEST_DIR"/transform/"));
    goldFiles[0].insert(0,string(TEST_DIR"/transform/"));

    af::dim4 objDims = inNumDims[0];

    vector<af::dim4>       HNumDims;
    vector<vector<float> > HIn;
    vector<vector<float> > HTests;
    readTests<float, float, float>(pHomographyFile, HNumDims, HIn, HTests);

    af::dim4 HDims = HNumDims[0];

    af_array sceneArray_f32 = 0;
    af_array goldArray_f32 = 0;
    af_array outArray_f32 = 0;
    af_array sceneArray = 0;
    af_array goldArray = 0;
    af_array outArray = 0;
    af_array HArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_load_image(&sceneArray_f32, inFiles[1].c_str(), false));
    ASSERT_EQ(AF_SUCCESS, af_load_image(&goldArray_f32, goldFiles[0].c_str(), false));

    ASSERT_EQ(AF_SUCCESS, conv_image<T>(&sceneArray, sceneArray_f32));
    ASSERT_EQ(AF_SUCCESS, conv_image<T>(&goldArray, goldArray_f32));

    ASSERT_EQ(AF_SUCCESS, af_create_array(&HArray, &(HIn[0].front()), HDims.ndims(), HDims.get(), f32));

    ASSERT_EQ(AF_SUCCESS, af_transform(&outArray, sceneArray, HArray, objDims[0], objDims[1], method, invert));

    // Get gold data
    dim_t goldEl = 0;
    ASSERT_EQ(AF_SUCCESS, af_get_elements(&goldEl, goldArray));
    T* goldData = new T[goldEl];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)goldData, goldArray));

    // Get result
    dim_t outEl = 0;
    ASSERT_EQ(AF_SUCCESS, af_get_elements(&outEl, outArray));
    T* outData = new T[outEl];
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData, outArray));

    const float thr = 1.1f;

    // Maximum number of wrong pixels must be <= 0.01% of number of elements,
    // this metric is necessary due to rounding errors between different
    // backends for AF_INTERP_NEAREST and AF_INTERP_LOWER
    const size_t maxErr = goldEl * 0.0001f;
    size_t err = 0;

    for (dim_t elIter = 0; elIter < goldEl; elIter++) {
        err += fabs((float)floor(outData[elIter]) - (float)floor(goldData[elIter])) > thr;
        if (err > maxErr)
            ASSERT_LE(err, maxErr) << "at: " << elIter << std::endl;
    }

    delete[] goldData;
    delete[] outData;

    if(sceneArray_f32 != 0) af_release_array(sceneArray_f32);
    if(goldArray_f32  != 0) af_release_array(goldArray_f32);
    if(outArray_f32   != 0) af_release_array(outArray_f32);
    if(sceneArray     != 0) af_release_array(sceneArray);
    if(goldArray      != 0) af_release_array(goldArray);
    if(outArray       != 0) af_release_array(outArray);
    if(HArray         != 0) af_release_array(HArray);
}

TYPED_TEST(Transform, PerspectiveNearest)
{
    transformTest<TypeParam>(string(TEST_DIR"/transform/tux_nearest.test"),
                             string(TEST_DIR"/transform/tux_tmat.test"),
                             AF_INTERP_NEAREST, false);
}

TYPED_TEST(Transform, PerspectiveBilinear)
{
    transformTest<TypeParam>(string(TEST_DIR"/transform/tux_bilinear.test"),
                             string(TEST_DIR"/transform/tux_tmat.test"),
                             AF_INTERP_BILINEAR, false);
}

TYPED_TEST(Transform, PerspectiveLower)
{
    transformTest<TypeParam>(string(TEST_DIR"/transform/tux_lower.test"),
                             string(TEST_DIR"/transform/tux_tmat.test"),
                             AF_INTERP_LOWER, false);
}

TYPED_TEST(Transform, PerspectiveNearestInvert)
{
    transformTest<TypeParam>(string(TEST_DIR"/transform/tux_nearest.test"),
                             string(TEST_DIR"/transform/tux_tmat_inverse.test"),
                             AF_INTERP_NEAREST, true);
}

TYPED_TEST(Transform, PerspectiveBilinearInvert)
{
    transformTest<TypeParam>(string(TEST_DIR"/transform/tux_bilinear.test"),
                             string(TEST_DIR"/transform/tux_tmat_inverse.test"),
                             AF_INTERP_BILINEAR, true);
}

TYPED_TEST(Transform, PerspectiveLowerInvert)
{
    transformTest<TypeParam>(string(TEST_DIR"/transform/tux_lower.test"),
                             string(TEST_DIR"/transform/tux_tmat_inverse.test"),
                             AF_INTERP_LOWER, true);
}

TYPED_TEST(TransformInt, PerspectiveNearest)
{
    transformTest<TypeParam>(string(TEST_DIR"/transform/tux_nearest.test"),
                             string(TEST_DIR"/transform/tux_tmat.test"),
                             AF_INTERP_NEAREST, false);
}

TYPED_TEST(TransformInt, PerspectiveBilinear)
{
    transformTest<TypeParam>(string(TEST_DIR"/transform/tux_bilinear.test"),
                             string(TEST_DIR"/transform/tux_tmat.test"),
                             AF_INTERP_BILINEAR, false);
}

TYPED_TEST(TransformInt, PerspectiveLower)
{
    transformTest<TypeParam>(string(TEST_DIR"/transform/tux_lower.test"),
                             string(TEST_DIR"/transform/tux_tmat.test"),
                             AF_INTERP_LOWER, false);
}

TYPED_TEST(TransformInt, PerspectiveNearestInvert)
{
    transformTest<TypeParam>(string(TEST_DIR"/transform/tux_nearest.test"),
                             string(TEST_DIR"/transform/tux_tmat_inverse.test"),
                             AF_INTERP_NEAREST, true);
}

TYPED_TEST(TransformInt, PerspectiveBilinearInvert)
{
    transformTest<TypeParam>(string(TEST_DIR"/transform/tux_bilinear.test"),
                             string(TEST_DIR"/transform/tux_tmat_inverse.test"),
                             AF_INTERP_BILINEAR, true);
}

TYPED_TEST(TransformInt, PerspectiveLowerInvert)
{
    transformTest<TypeParam>(string(TEST_DIR"/transform/tux_lower.test"),
                             string(TEST_DIR"/transform/tux_tmat_inverse.test"),
                             AF_INTERP_LOWER, true);
}


///////////////////////////////////// CPP ////////////////////////////////
//
TEST(Transform, CPP)
{
    vector<af::dim4>   inDims;
    vector<string> inFiles;
    vector<dim_t>  goldDim;
    vector<string> goldFiles;

    vector<af::dim4> HDims;
    vector<vector<float> >   HIn;
    vector<vector<float> >   HTests;
    readTests<float, float, float>(TEST_DIR"/transform/tux_tmat.test",HDims,HIn,HTests);

    readImageTests(string(TEST_DIR"/transform/tux_nearest.test"), inDims, inFiles, goldDim, goldFiles);

    inFiles[0].insert(0,string(TEST_DIR"/transform/"));
    inFiles[1].insert(0,string(TEST_DIR"/transform/"));

    goldFiles[0].insert(0,string(TEST_DIR"/transform/"));

    af::array H = af::array(HDims[0][0], HDims[0][1], &(HIn[0].front()));
    af::array IH = af::array(HDims[0][0], HDims[0][1], &(HIn[0].front()));

    af::array scene_img = af::loadImage(inFiles[1].c_str(), false);

    af::array gold_img = af::loadImage(goldFiles[0].c_str(), false);

    af::array out_img = af::transform(scene_img, IH, inDims[0][0], inDims[0][1], AF_INTERP_NEAREST, false);

    af::dim4 outDims = out_img.dims();
    af::dim4 goldDims = gold_img.dims();

    float* h_out_img = new float[outDims[0] * outDims[1]];
    out_img.host(h_out_img);
    float* h_gold_img = new float[goldDims[0] * goldDims[1]];
    gold_img.host(h_gold_img);

    const dim_t n = gold_img.elements();

    const float thr = 1.0f;

    // Maximum number of wrong pixels must be <= 0.01% of number of elements,
    // this metric is necessary due to rounding errors between different
    // backends for AF_INTERP_NEAREST and AF_INTERP_LOWER
    const size_t maxErr = n * 0.0001f;
    size_t err = 0;

    for (dim_t elIter = 0; elIter < n; elIter++) {
        err += fabs((int)h_out_img[elIter] - h_gold_img[elIter]) > thr;
        if (err > maxErr)
            ASSERT_LE(err, maxErr) << "at: " << elIter << std::endl;
    }

    delete[] h_gold_img;
    delete[] h_out_img;
}
