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
    if (noImageIOTests()) return;

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
    vector<T> goldData(goldEl);
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)&goldData.front(), goldArray));

    // Get result
    dim_t outEl = 0;
    ASSERT_EQ(AF_SUCCESS, af_get_elements(&outEl, outArray));
    vector<T> outData(outEl);
    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)&outData.front(), outArray));

    const float thr = 1.1f;

    // Maximum number of wrong pixels must be <= 0.01% of number of elements,
    // this metric is necessary due to rounding errors between different
    // backends for AF_INTERP_NEAREST and AF_INTERP_LOWER
    const size_t maxErr = goldEl * 0.0001f;
    size_t err = 0;

    for (dim_t elIter = 0; elIter < goldEl; elIter++) {
        err += fabs((float)floor(outData[elIter]) - (float)floor(goldData[elIter])) > thr;
        if (err > maxErr) {
            ASSERT_LE(err, maxErr) << "at: " << elIter << std::endl;
        }
    }

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
    if (noImageIOTests()) return;

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

    vector<float> h_out_img(outDims[0] * outDims[1]);
    out_img.host(&h_out_img.front());
    vector<float> h_gold_img(goldDims[0] * goldDims[1]);
    gold_img.host(&h_gold_img.front());

    const dim_t n = gold_img.elements();
    const float thr = 1.0f;

    // Maximum number of wrong pixels must be <= 0.01% of number of elements,
    // this metric is necessary due to rounding errors between different
    // backends for AF_INTERP_NEAREST and AF_INTERP_LOWER
    const size_t maxErr = n * 0.0001f;
    size_t err = 0;

    for (dim_t elIter = 0; elIter < n; elIter++) {
        err += fabs((int)h_out_img[elIter] - h_gold_img[elIter]) > thr;
        if (err > maxErr) {
            ASSERT_LE(err, maxErr) << "at: " << elIter << std::endl;
        }
    }
}

// This tests batching of different forms
// tf0 rotates by 90 clockwise
// tf1 rotates by 90 counter clockwise
// This test simply makes sure the batching is working correctly
TEST(TransformBatching, CPP)
{
    vector<af::dim4>        vDims;
    vector<vector<float> >  in;
    vector<vector<float> >  gold;

    readTests<float, float, int>(string(TEST_DIR"/transform/transform_batching.test"), vDims, in, gold);

    af::array img0     (vDims[0], &(in[0].front()));
    af::array img1     (vDims[1], &(in[1].front()));
    af::array ip_tile  (vDims[2], &(in[2].front()));
    af::array ip_quad  (vDims[3], &(in[3].front()));
    af::array ip_mult  (vDims[4], &(in[4].front()));
    af::array ip_tile3 (vDims[5], &(in[5].front()));
    af::array ip_quad3 (vDims[6], &(in[6].front()));

    af::array tf0      (vDims[7 + 0], &(in[7 + 0].front()));
    af::array tf1      (vDims[7 + 1], &(in[7 + 1].front()));
    af::array tf_tile  (vDims[7 + 2], &(in[7 + 2].front()));
    af::array tf_quad  (vDims[7 + 3], &(in[7 + 3].front()));
    af::array tf_mult  (vDims[7 + 4], &(in[7 + 4].front()));
    af::array tf_mult3 (vDims[7 + 5], &(in[7 + 5].front()));
    af::array tf_mult3x(vDims[7 + 6], &(in[7 + 6].front()));

    const int X = img0.dims(0);
    const int Y = img0.dims(1);

    ASSERT_EQ(gold.size(), 21u);
    vector<af::array> out(gold.size());
    out[0 ] = transform(img0    , tf0      , Y, X, AF_INTERP_NEAREST);  // 1,1 x 1,1
    out[1 ] = transform(img0    , tf1      , Y, X, AF_INTERP_NEAREST);  // 1,1 x 1,1
    out[2 ] = transform(img1    , tf0      , Y, X, AF_INTERP_NEAREST);  // 1,1 x 1,1
    out[3 ] = transform(img1    , tf1      , Y, X, AF_INTERP_NEAREST);  // 1,1 x 1,1

    out[4 ] = transform(img0    , tf_tile  , Y, X, AF_INTERP_NEAREST);  // 1,1 x N,1
    out[5 ] = transform(img0    , tf_mult  , Y, X, AF_INTERP_NEAREST);  // 1,1 x N,N
    out[6 ] = transform(img0    , tf_quad  , Y, X, AF_INTERP_NEAREST);  // 1,1 x 1,N

    out[7 ] = transform(ip_tile , tf0      , Y, X, AF_INTERP_NEAREST);  // N,1 x 1,1
    out[8 ] = transform(ip_tile , tf_tile  , Y, X, AF_INTERP_NEAREST);  // N,1 x N,1
    out[9 ] = transform(ip_tile , tf_mult  , Y, X, AF_INTERP_NEAREST);  // N,N x N,N
    out[10] = transform(ip_tile , tf_quad  , Y, X, AF_INTERP_NEAREST);  // N,1 x 1,N

    out[11] = transform(ip_quad , tf0      , Y, X, AF_INTERP_NEAREST);  // 1,N x 1,1
    out[12] = transform(ip_quad , tf_quad  , Y, X, AF_INTERP_NEAREST);  // 1,N x 1,N
    out[13] = transform(ip_quad , tf_mult  , Y, X, AF_INTERP_NEAREST);  // 1,N x N,N
    out[14] = transform(ip_quad , tf_tile  , Y, X, AF_INTERP_NEAREST);  // 1,N x N,1

    out[15] = transform(ip_mult , tf0      , Y, X, AF_INTERP_NEAREST);  // N,N x 1,1
    out[16] = transform(ip_mult , tf_tile  , Y, X, AF_INTERP_NEAREST);  // N,N x N,1
    out[17] = transform(ip_mult , tf_mult  , Y, X, AF_INTERP_NEAREST);  // N,N x N,N
    out[18] = transform(ip_mult , tf_quad  , Y, X, AF_INTERP_NEAREST);  // N,N x 1,N

    out[19] = transform(ip_tile3, tf_mult3 , Y, X, AF_INTERP_NEAREST);  // N,1 x N,N
    out[20] = transform(ip_quad3, tf_mult3x, Y, X, AF_INTERP_NEAREST);  // 1,N x N,N

    af::array x_(af::dim4(35, 40, 1, 1), &(gold[1].front()));

    for(int i = 0; i < (int)gold.size(); i++) {
        // Get result
        vector<float> outData(out[i].elements());
        out[i].host((void*)&outData.front());

        for(int iter = 0; iter < (int)gold[i].size(); iter++) {
            ASSERT_EQ(gold[i][iter], outData[iter]) << "at: " << iter << std::endl
                    << "for " << i << "-th operation"<< std::endl;
        }
    }
}
