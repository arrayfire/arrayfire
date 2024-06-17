/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/compatible.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <cmath>
#include <string>
#include <typeinfo>
#include <vector>

using af::dim4;
using std::abs;
using std::endl;
using std::string;
using std::vector;

typedef struct {
    float f[5];
} feat_t;

static bool feat_cmp(feat_t i, feat_t j) {
    for (int k = 0; k < 5; k++)
        if (i.f[k] != j.f[k]) return (i.f[k] < j.f[k]);

    return false;
}

static void array_to_feat(vector<feat_t> &feat, float *x, float *y,
                          float *score, float *orientation, float *size,
                          unsigned nfeat) {
    feat.resize(nfeat);
    for (unsigned i = 0; i < feat.size(); i++) {
        feat[i].f[0] = x[i];
        feat[i].f[1] = y[i];
        feat[i].f[2] = score[i];
        feat[i].f[3] = orientation[i];
        feat[i].f[4] = size[i];
    }
}

template<typename T>
class Harris : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

typedef ::testing::Types<float, double> TestTypes;

TYPED_TEST_SUITE(Harris, TestTypes);

template<typename T>
void harrisTest(string pTestFile, float sigma, unsigned block_size) {
    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<vector<float>> gold;

    readImageTests(pTestFile, inDims, inFiles, gold);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        dim_t nElems         = 0;
        af_array inArray_f32 = 0;
        af_array inArray     = 0;
        af_features out;

        inFiles[testId].insert(0, string(TEST_DIR "/harris/"));

        ASSERT_SUCCESS(
            af_load_image(&inArray_f32, inFiles[testId].c_str(), false));

        ASSERT_SUCCESS(conv_image<T>(&inArray, inArray_f32));

        ASSERT_SUCCESS_CHECK_SUPRT(
            af_harris(&out, inArray, 500, 1e5f, sigma, block_size, 0.04f));

        dim_t n = 0;
        af_array x, y, score, orientation, size;

        ASSERT_SUCCESS(af_get_features_num(&n, out));
        ASSERT_SUCCESS(af_get_features_xpos(&x, out));
        ASSERT_SUCCESS(af_get_features_ypos(&y, out));
        ASSERT_SUCCESS(af_get_features_score(&score, out));
        ASSERT_SUCCESS(af_get_features_orientation(&orientation, out));
        ASSERT_SUCCESS(af_get_features_size(&size, out));

        ASSERT_SUCCESS(af_get_elements(&nElems, x));

        vector<float> outX(gold[0].size());
        vector<float> outY(gold[1].size());
        vector<float> outScore(gold[2].size());
        vector<float> outOrientation(gold[3].size());
        vector<float> outSize(gold[4].size());
        ASSERT_SUCCESS(af_get_data_ptr((void *)&outX.front(), x));
        ASSERT_SUCCESS(af_get_data_ptr((void *)&outY.front(), y));
        ASSERT_SUCCESS(af_get_data_ptr((void *)&outScore.front(), score));
        ASSERT_SUCCESS(
            af_get_data_ptr((void *)&outOrientation.front(), orientation));
        ASSERT_SUCCESS(af_get_data_ptr((void *)&outSize.front(), size));

        vector<feat_t> out_feat;
        array_to_feat(out_feat, &outX.front(), &outY.front(), &outScore.front(),
                      &outOrientation.front(), &outSize.front(), n);

        vector<feat_t> gold_feat;
        array_to_feat(gold_feat, &gold[0].front(), &gold[1].front(),
                      &gold[2].front(), &gold[3].front(), &gold[4].front(),
                      gold[0].size());

        std::sort(out_feat.begin(), out_feat.end(), feat_cmp);
        std::sort(gold_feat.begin(), gold_feat.end(), feat_cmp);

        for (int elIter = 0; elIter < (int)nElems; elIter++) {
            ASSERT_EQ(out_feat[elIter].f[0], gold_feat[elIter].f[0])
                << "at: " << elIter << endl;
            ASSERT_EQ(out_feat[elIter].f[1], gold_feat[elIter].f[1])
                << "at: " << elIter << endl;
            ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]), 1e2)
                << "at: " << elIter << endl;
            ASSERT_EQ(out_feat[elIter].f[3], gold_feat[elIter].f[3])
                << "at: " << elIter << endl;
            ASSERT_EQ(out_feat[elIter].f[4], gold_feat[elIter].f[4])
                << "at: " << elIter << endl;
        }

        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(inArray_f32));

        ASSERT_SUCCESS(af_release_features(out));
    }
}

#define HARRIS_INIT(desc, image, sigma, block_size)                        \
    TYPED_TEST(Harris, desc) {                                             \
        harrisTest<TypeParam>(string(TEST_DIR "/harris/" #image "_" #sigma \
                                              "_" #block_size ".test"),    \
                              sigma, block_size);                          \
    }

HARRIS_INIT(square_0_3, square, 0, 3);
HARRIS_INIT(square_0_7, square, 0, 7);
HARRIS_INIT(square_1_0, square, 1, 0);
HARRIS_INIT(square_5_0, square, 5, 0);
HARRIS_INIT(lena_0_3, lena, 0, 3);
HARRIS_INIT(lena_0_7, lena, 0, 7);
HARRIS_INIT(lena_1_0, lena, 1, 0);
HARRIS_INIT(lena_5_0, lena, 5, 0);

/////////////////////////////////// CPP ////////////////////////////////

using af::array;
using af::features;
using af::harris;
using af::loadImage;

TEST(FloatHarris, CPP) {
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<vector<float>> gold;

    readImageTests(string(TEST_DIR "/harris/square_0_3.test"), inDims, inFiles,
                   gold);
    inFiles[0].insert(0, string(TEST_DIR "/harris/"));

    array in = loadImage(inFiles[0].c_str(), false);

    features out;
    try { out = harris(in, 500, 1e5f, 0.0f, 3, 0.04f);
    } catch FUNCTION_UNSUPPORTED

    vector<float> outX(gold[0].size());
    vector<float> outY(gold[1].size());
    vector<float> outScore(gold[2].size());
    vector<float> outOrientation(gold[3].size());
    vector<float> outSize(gold[4].size());
    out.getX().host(&outX.front());
    out.getY().host(&outY.front());
    out.getScore().host(&outScore.front());
    out.getOrientation().host(&outOrientation.front());
    out.getSize().host(&outSize.front());

    vector<feat_t> out_feat;
    array_to_feat(out_feat, &outX.front(), &outY.front(), &outScore.front(),
                  &outOrientation.front(), &outSize.front(),
                  out.getNumFeatures());

    vector<feat_t> gold_feat;
    array_to_feat(gold_feat, &gold[0].front(), &gold[1].front(),
                  &gold[2].front(), &gold[3].front(), &gold[4].front(),
                  gold[0].size());

    std::sort(out_feat.begin(), out_feat.end(), feat_cmp);
    std::sort(gold_feat.begin(), gold_feat.end(), feat_cmp);

    for (unsigned elIter = 0; elIter < out.getNumFeatures(); elIter++) {
        ASSERT_EQ(out_feat[elIter].f[0], gold_feat[elIter].f[0])
            << "at: " << elIter << endl;
        ASSERT_EQ(out_feat[elIter].f[1], gold_feat[elIter].f[1])
            << "at: " << elIter << endl;
        ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]), 1e2)
            << "at: " << elIter << endl;
        ASSERT_EQ(out_feat[elIter].f[3], gold_feat[elIter].f[3])
            << "at: " << elIter << endl;
        ASSERT_EQ(out_feat[elIter].f[4], gold_feat[elIter].f[4])
            << "at: " << elIter << endl;
    }
}
