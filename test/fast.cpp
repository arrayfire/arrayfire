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
class FloatFAST : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

template<typename T>
class FixedFAST : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

typedef ::testing::Types<float, double> FloatTestTypes;
typedef ::testing::Types<int, unsigned, short, ushort> FixedTestTypes;

TYPED_TEST_SUITE(FloatFAST, FloatTestTypes);
TYPED_TEST_SUITE(FixedFAST, FixedTestTypes);

template<typename T>
void fastTest(string pTestFile, bool nonmax) {
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

        inFiles[testId].insert(0, string(TEST_DIR "/fast/"));

        ASSERT_SUCCESS(
            af_load_image(&inArray_f32, inFiles[testId].c_str(), false));

        ASSERT_SUCCESS(conv_image<T>(&inArray, inArray_f32));

        ASSERT_SUCCESS_CHECK_SUPRT(af_fast(&out, inArray, 20.0f, 9, nonmax, 0.05f, 3));

        dim_t n = 0;
        af_array x, y, score, orientation, size;

        ASSERT_SUCCESS(af_get_features_num(&n, out));
        ASSERT_SUCCESS(af_get_features_xpos(&x, out));
        ASSERT_SUCCESS(af_get_features_ypos(&y, out));
        ASSERT_SUCCESS(af_get_features_score(&score, out));
        ASSERT_SUCCESS(af_get_features_orientation(&orientation, out));
        ASSERT_SUCCESS(af_get_features_size(&size, out));

        ASSERT_SUCCESS(af_get_elements(&nElems, x));

        float *outX           = new float[gold[0].size()];
        float *outY           = new float[gold[1].size()];
        float *outScore       = new float[gold[2].size()];
        float *outOrientation = new float[gold[3].size()];
        float *outSize        = new float[gold[4].size()];
        ASSERT_SUCCESS(af_get_data_ptr((void *)outX, x));
        ASSERT_SUCCESS(af_get_data_ptr((void *)outY, y));
        ASSERT_SUCCESS(af_get_data_ptr((void *)outScore, score));
        ASSERT_SUCCESS(af_get_data_ptr((void *)outOrientation, orientation));
        ASSERT_SUCCESS(af_get_data_ptr((void *)outSize, size));

        vector<feat_t> out_feat;
        array_to_feat(out_feat, outX, outY, outScore, outOrientation, outSize,
                      n);

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
            ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]),
                      1e-3)
                << "at: " << elIter << endl;
            ASSERT_EQ(out_feat[elIter].f[3], gold_feat[elIter].f[3])
                << "at: " << elIter << endl;
            ASSERT_EQ(out_feat[elIter].f[4], gold_feat[elIter].f[4])
                << "at: " << elIter << endl;
        }

        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(inArray_f32));

        ASSERT_SUCCESS(af_release_features(out));

        delete[] outX;
        delete[] outY;
        delete[] outScore;
        delete[] outOrientation;
        delete[] outSize;
    }
}

#define FLOAT_FAST_INIT(desc, image, nonmax)                                \
    TYPED_TEST(FloatFAST, desc) {                                           \
        fastTest<TypeParam>(string(TEST_DIR "/fast/" #image "_float.test"), \
                            nonmax);                                        \
    }

#define FIXED_FAST_INIT(desc, image, nonmax)                                \
    TYPED_TEST(FixedFAST, desc) {                                           \
        fastTest<TypeParam>(string(TEST_DIR "/fast/" #image "_fixed.test"), \
                            nonmax);                                        \
    }

FLOAT_FAST_INIT(square, square, false);
FLOAT_FAST_INIT(square_nonmax, square_nonmax, true);
FIXED_FAST_INIT(square, square, false);
FIXED_FAST_INIT(square_nonmax, square_nonmax, true);

/////////////////////////////////// CPP ////////////////////////////////

using af::array;
using af::features;
using af::loadImage;

TEST(FloatFAST, CPP) {
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<vector<float>> gold;

    readImageTests(string(TEST_DIR "/fast/square_nonmax_float.test"), inDims,
                   inFiles, gold);
    inFiles[0].insert(0, string(TEST_DIR "/fast/"));

    array in = loadImage(inFiles[0].c_str(), false);

    features out;
    try { out = fast(in, 20.0f, 9, true, 0.05f, 3); } catch FUNCTION_UNSUPPORTED

    float *outX           = new float[gold[0].size()];
    float *outY           = new float[gold[1].size()];
    float *outScore       = new float[gold[2].size()];
    float *outOrientation = new float[gold[3].size()];
    float *outSize        = new float[gold[4].size()];
    out.getX().host(outX);
    out.getY().host(outY);
    out.getScore().host(outScore);
    out.getOrientation().host(outOrientation);
    out.getSize().host(outSize);

    vector<feat_t> out_feat;
    array_to_feat(out_feat, outX, outY, outScore, outOrientation, outSize,
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
        ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]), 1e-3)
            << "at: " << elIter << endl;
        ASSERT_EQ(out_feat[elIter].f[3], gold_feat[elIter].f[3])
            << "at: " << elIter << endl;
        ASSERT_EQ(out_feat[elIter].f[4], gold_feat[elIter].f[4])
            << "at: " << elIter << endl;
    }

    delete[] outX;
    delete[] outY;
    delete[] outScore;
    delete[] outOrientation;
    delete[] outSize;
}
