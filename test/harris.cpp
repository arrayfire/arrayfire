/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <af/compatible.h>
#include <string>
#include <vector>
#include <cmath>
#include <testHelpers.hpp>
#include <typeinfo>

using std::string;
using std::vector;
using af::dim4;

typedef struct
{
    float f[5];
} feat_t;

bool feat_cmp(feat_t i, feat_t j)
{
    for (int k = 0; k < 5; k++)
        if (i.f[k] != j.f[k])
            return (i.f[k] < j.f[k]);

    return false;
}

void array_to_feat(vector<feat_t>& feat, float *x, float *y, float *score, float *orientation, float *size, unsigned nfeat)
{
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
class Harris : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

typedef ::testing::Types<float, double> TestTypes;

TYPED_TEST_CASE(Harris, TestTypes);

template<typename T>
void harrisTest(string pTestFile, float sigma, unsigned block_size)
{
    if (noDoubleTests<T>()) return;

    vector<dim4>        inDims;
    vector<string>     inFiles;
    vector<vector<float> > gold;

    readImageTests(pTestFile, inDims, inFiles, gold);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {
        dim_t nElems       = 0;
        af_array inArray_f32  = 0;
        af_array inArray      = 0;
        af_features out;

        inFiles[testId].insert(0,string(TEST_DIR"/harris/"));

        ASSERT_EQ(AF_SUCCESS, af_load_image(&inArray_f32, inFiles[testId].c_str(), false));

        ASSERT_EQ(AF_SUCCESS, conv_image<T>(&inArray, inArray_f32));

        ASSERT_EQ(AF_SUCCESS, af_harris(&out, inArray, 500, 1e5f, sigma, block_size, 0.04f));

        dim_t n = 0;
        af_array x, y, score, orientation, size;

        ASSERT_EQ(AF_SUCCESS, af_get_features_num(&n, out));
        ASSERT_EQ(AF_SUCCESS, af_get_features_xpos(&x, out));
        ASSERT_EQ(AF_SUCCESS, af_get_features_ypos(&y, out));
        ASSERT_EQ(AF_SUCCESS, af_get_features_score(&score, out));
        ASSERT_EQ(AF_SUCCESS, af_get_features_orientation(&orientation, out));
        ASSERT_EQ(AF_SUCCESS, af_get_features_size(&size, out));


        ASSERT_EQ(AF_SUCCESS, af_get_elements(&nElems, x));

        float * outX           = new float[gold[0].size()];
        float * outY           = new float[gold[1].size()];
        float * outScore       = new float[gold[2].size()];
        float * outOrientation = new float[gold[3].size()];
        float * outSize        = new float[gold[4].size()];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outX, x));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outY, y));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outScore, score));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outOrientation, orientation));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outSize, size));

        vector<feat_t> out_feat;
        array_to_feat(out_feat, outX, outY, outScore, outOrientation, outSize, n);

        vector<feat_t> gold_feat;
        array_to_feat(gold_feat, &gold[0].front(), &gold[1].front(), &gold[2].front(), &gold[3].front(), &gold[4].front(), gold[0].size());

        std::sort(out_feat.begin(), out_feat.end(), feat_cmp);
        std::sort(gold_feat.begin(), gold_feat.end(), feat_cmp);

        for (int elIter = 0; elIter < (int)nElems; elIter++) {
            ASSERT_EQ(out_feat[elIter].f[0], gold_feat[elIter].f[0]) << "at: " << elIter << std::endl;
            ASSERT_EQ(out_feat[elIter].f[1], gold_feat[elIter].f[1]) << "at: " << elIter << std::endl;
            ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]), 1e2) << "at: " << elIter << std::endl;
            ASSERT_EQ(out_feat[elIter].f[3], gold_feat[elIter].f[3]) << "at: " << elIter << std::endl;
            ASSERT_EQ(out_feat[elIter].f[4], gold_feat[elIter].f[4]) << "at: " << elIter << std::endl;
        }

        ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(inArray_f32));

        ASSERT_EQ(AF_SUCCESS, af_release_array(x));
        ASSERT_EQ(AF_SUCCESS, af_release_array(y));
        ASSERT_EQ(AF_SUCCESS, af_release_array(score));
        ASSERT_EQ(AF_SUCCESS, af_release_array(orientation));
        ASSERT_EQ(AF_SUCCESS, af_release_array(size));

        delete [] outX;
        delete [] outY;
        delete [] outScore;
        delete [] outOrientation;
        delete [] outSize;
    }
}

#define HARRIS_INIT(desc, image, sigma, block_size) \
    TYPED_TEST(Harris, desc) \
    {   \
        harrisTest<TypeParam>(string(TEST_DIR"/harris/"#image"_"#sigma"_"#block_size".test"), sigma, block_size); \
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

TEST(FloatHarris, CPP)
{
    if (noDoubleTests<float>()) return;

    vector<dim4>        inDims;
    vector<string>     inFiles;
    vector<vector<float> > gold;

    readImageTests(string(TEST_DIR"/harris/square_0_3.test"), inDims, inFiles, gold);
    inFiles[0].insert(0,string(TEST_DIR"/harris/"));

    af::array in = af::loadImage(inFiles[0].c_str(), false);

    af::features out = harris(in, 500, 1e5f, 0.0f, 3, 0.04f);

    float * outX           = new float[gold[0].size()];
    float * outY           = new float[gold[1].size()];
    float * outScore       = new float[gold[2].size()];
    float * outOrientation = new float[gold[3].size()];
    float * outSize        = new float[gold[4].size()];
    out.getX().host(outX);
    out.getY().host(outY);
    out.getScore().host(outScore);
    out.getOrientation().host(outOrientation);
    out.getSize().host(outSize);

    vector<feat_t> out_feat;
    array_to_feat(out_feat, outX, outY, outScore, outOrientation, outSize, out.getNumFeatures());

    vector<feat_t> gold_feat;
    array_to_feat(gold_feat, &gold[0].front(), &gold[1].front(), &gold[2].front(), &gold[3].front(), &gold[4].front(), gold[0].size());

    std::sort(out_feat.begin(), out_feat.end(), feat_cmp);
    std::sort(gold_feat.begin(), gold_feat.end(), feat_cmp);

    for (unsigned elIter = 0; elIter < out.getNumFeatures(); elIter++) {
        ASSERT_EQ(out_feat[elIter].f[0], gold_feat[elIter].f[0]) << "at: " << elIter << std::endl;
        ASSERT_EQ(out_feat[elIter].f[1], gold_feat[elIter].f[1]) << "at: " << elIter << std::endl;
        ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]), 1e2) << "at: " << elIter << std::endl;
        ASSERT_EQ(out_feat[elIter].f[3], gold_feat[elIter].f[3]) << "at: " << elIter << std::endl;
        ASSERT_EQ(out_feat[elIter].f[4], gold_feat[elIter].f[4]) << "at: " << elIter << std::endl;
    }

    delete[] outX;
    delete[] outY;
    delete[] outScore;
    delete[] outOrientation;
    delete[] outSize;
}
