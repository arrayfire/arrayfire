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
class Susan : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

typedef ::testing::Types<float, double, int, uint, char, uchar> TestTypes;

TYPED_TEST_CASE(Susan, TestTypes);

template<typename T>
void susanTest(string pTestFile, float t, float g)
{
    if (noDoubleTests<T>()) return;

    vector<dim4>        inDims;
    vector<string>     inFiles;
    vector<vector<float> > gold;

    readImageTests(pTestFile, inDims, inFiles, gold);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {
        inFiles[testId].insert(0, string(TEST_DIR "/susan/"));

        af::array in = af::loadImage(inFiles[testId].c_str(), false);

        af::features out = af::susan(in, 3, t, g, 0.05f, 3);

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

        for (int elIter = 0; elIter < (int)out.getNumFeatures(); elIter++) {
            ASSERT_EQ(out_feat[elIter].f[0], gold_feat[elIter].f[0]) << "at: " << elIter << std::endl;
            ASSERT_EQ(out_feat[elIter].f[1], gold_feat[elIter].f[1]) << "at: " << elIter << std::endl;
            ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]), 1e2) << "at: " << elIter << std::endl;
            ASSERT_EQ(out_feat[elIter].f[3], gold_feat[elIter].f[3]) << "at: " << elIter << std::endl;
            ASSERT_EQ(out_feat[elIter].f[4], gold_feat[elIter].f[4]) << "at: " << elIter << std::endl;
        }

        delete [] outX;
        delete [] outY;
        delete [] outScore;
        delete [] outOrientation;
        delete [] outSize;
    }
}

#define SUSAN_TEST(image, tval, gval) \
    TYPED_TEST(Susan, image) \
    {   \
        susanTest<TypeParam>(string(TEST_DIR "/susan/"#image".test"), tval, gval);\
    }

SUSAN_TEST(man_t32_g10, 32, 10);
SUSAN_TEST(square_t32_g10, 32, 10);
SUSAN_TEST(square_t32_g20, 32, 20);

TEST(Susan, InvalidDims)
{
    try {
        af::array a = af::randu(256);
        af::features out = af::susan(a);
        EXPECT_TRUE(false);
    } catch (af::exception &e) {
        EXPECT_TRUE(true);
    }
}

TEST(Susan, InvalidRadius)
{
    try {
        af::array a = af::randu(256);
        af::features out = af::susan(a, 10);
        EXPECT_TRUE(false);
    } catch (af::exception &e) {
        EXPECT_TRUE(true);
    }
}

TEST(Susan, InvalidThreshold)
{
    try {
        af::array a = af::randu(256);
        af::features out = af::susan(a, 3, -32, 10, 0.05f, 3);
        EXPECT_TRUE(false);
    } catch (af::exception &e) {
        EXPECT_TRUE(true);
    }
}

TEST(Susan, InvalidFeatureRatio)
{
    try {
        af::array a = af::randu(256);
        af::features out = af::susan(a, 3, 32, 10, 1.3f, 3);
        EXPECT_TRUE(false);
    } catch (af::exception &e) {
        EXPECT_TRUE(true);
    }
}

TEST(Susan, InvalidEdge)
{
    try {
        af::array a = af::randu(128, 128);
        af::features out = af::susan(a, 3, 32, 10, 1.3f, 129);
        EXPECT_TRUE(false);
    } catch (af::exception &e) {
        EXPECT_TRUE(true);
    }
}
