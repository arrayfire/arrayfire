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
    unsigned d[8];
} feat_desc_t;

typedef struct
{
    float f[5];
} feat_t;

typedef struct
{
    unsigned d[8];
} desc_t;

bool feat_cmp(feat_desc_t i, feat_desc_t j)
{
    for (int k = 0; k < 5; k++)
        if (i.f[k] != j.f[k])
            return (i.f[k] < j.f[k]);

    return true;
}

void array_to_feat_desc(vector<feat_desc_t>& feat, float* x, float* y, float* score, float* ori, float* size, unsigned* desc, unsigned nfeat)
{
    feat.resize(nfeat);
    for (size_t i = 0; i < feat.size(); i++) {
        feat[i].f[0] = x[i];
        feat[i].f[1] = y[i];
        feat[i].f[2] = score[i];
        feat[i].f[3] = ori[i];
        feat[i].f[4] = size[i];
        for (unsigned j = 0; j < 8; j++)
            feat[i].d[j] = desc[i * 8 + j];
    }
}

void array_to_feat_desc(vector<feat_desc_t>& feat, float* x, float* y, float* score, float* ori, float* size, vector<vector<unsigned> >& desc, unsigned nfeat)
{
    feat.resize(nfeat);
    for (size_t i = 0; i < feat.size(); i++) {
        feat[i].f[0] = x[i];
        feat[i].f[1] = y[i];
        feat[i].f[2] = score[i];
        feat[i].f[3] = ori[i];
        feat[i].f[4] = size[i];
        for (unsigned j = 0; j < 8; j++)
            feat[i].d[j] = desc[i][j];
    }
}

void array_to_feat(vector<feat_t>& feat, float *x, float *y, float *score, float *ori, float *size, unsigned nfeat)
{
    feat.resize(nfeat);
    for (unsigned i = 0; i < feat.size(); i++) {
        feat[i].f[0] = x[i];
        feat[i].f[1] = y[i];
        feat[i].f[2] = score[i];
        feat[i].f[3] = ori[i];
        feat[i].f[4] = size[i];
    }
}

void split_feat_desc(vector<feat_desc_t>& fd, vector<feat_t>& f, vector<desc_t>& d)
{
    f.resize(fd.size());
    d.resize(fd.size());
    for (size_t i = 0; i < fd.size(); i++) {
        f[i].f[0] = fd[i].f[0];
        f[i].f[1] = fd[i].f[1];
        f[i].f[2] = fd[i].f[2];
        f[i].f[3] = fd[i].f[3];
        f[i].f[4] = fd[i].f[4];
        for (unsigned j = 0; j < 8; j++)
            d[i].d[j] = fd[i].d[j];
    }
}

unsigned popcount(unsigned x)
{
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x0000003F;
}

bool compareHamming(int data_size, unsigned *cpu, unsigned *gpu, unsigned thr = 1)
{
    bool ret = true;
    for(int i=0;i<data_size;i++)
    {
        unsigned x = (cpu[i] ^ gpu[i]);
        if(popcount(x) > thr) {
            ret = false;
            std::cout<<std::endl<<"@compareHamming: first mismatch."<<std::endl;
            std::cout<<"(cpu,gpu,cpu-gpu)["<<i<<"] : {"<<cpu[i]<<","<<gpu[i]<<","<<cpu[i]-gpu[i]<<"}"<<std::endl;
            std::cout<<std::endl;
            break;
        }
    }
    return ret;
}

template<typename T>
class ORB : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

typedef ::testing::Types<float, double> TestTypes;

TYPED_TEST_CASE(ORB, TestTypes);

template<typename T>
void orbTest(string pTestFile)
{
    if (noDoubleTests<T>()) return;

    vector<dim4>             inDims;
    vector<string>           inFiles;
    vector<vector<float> >    goldFeat;
    vector<vector<unsigned> > goldDesc;

    readImageFeaturesDescriptors<unsigned>(pTestFile, inDims, inFiles, goldFeat, goldDesc);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {
        af_array inArray_f32  = 0;
        af_array inArray      = 0;
        af_array desc         = 0;
        af_features feat;

        inFiles[testId].insert(0,string(TEST_DIR"/orb/"));

        ASSERT_EQ(AF_SUCCESS, af_load_image(&inArray_f32, inFiles[testId].c_str(), false));
        ASSERT_EQ(AF_SUCCESS, conv_image<T>(&inArray, inArray_f32));

        ASSERT_EQ(AF_SUCCESS, af_orb(&feat, &desc, inArray, 20.0f, 400, 1.2f, 8, true));

        dim_t n = 0;
        af_array x, y, score, orientation, size;

        ASSERT_EQ(AF_SUCCESS, af_get_features_num(&n, feat));
        ASSERT_EQ(AF_SUCCESS, af_get_features_xpos(&x, feat));
        ASSERT_EQ(AF_SUCCESS, af_get_features_ypos(&y, feat));
        ASSERT_EQ(AF_SUCCESS, af_get_features_score(&score, feat));
        ASSERT_EQ(AF_SUCCESS, af_get_features_orientation(&orientation, feat));
        ASSERT_EQ(AF_SUCCESS, af_get_features_size(&size, feat));

        float * outX           = new float[n];
        float * outY           = new float[n];
        float * outScore       = new float[n];
        float * outOrientation = new float[n];
        float * outSize        = new float[n];
        dim_t descSize;
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&descSize, desc));
        unsigned * outDesc     = new unsigned[descSize];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outX, x));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outY, y));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outScore, score));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outOrientation, orientation));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outSize, size));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outDesc, desc));

        vector<feat_desc_t> out_feat_desc;
        array_to_feat_desc(out_feat_desc, outX, outY, outScore, outOrientation, outSize, outDesc, n);

        vector<feat_desc_t> gold_feat_desc;
        array_to_feat_desc(gold_feat_desc, &goldFeat[0].front(), &goldFeat[1].front(), &goldFeat[2].front(), &goldFeat[3].front(), &goldFeat[4].front(), goldDesc, goldFeat[0].size());

        std::sort(out_feat_desc.begin(), out_feat_desc.end(), feat_cmp);
        std::sort(gold_feat_desc.begin(), gold_feat_desc.end(), feat_cmp);

        vector<feat_t> out_feat;
        vector<desc_t> v_out_desc;
        vector<feat_t> gold_feat;
        vector<desc_t> v_gold_desc;

        split_feat_desc(out_feat_desc, out_feat, v_out_desc);
        split_feat_desc(gold_feat_desc, gold_feat, v_gold_desc);

        for (int elIter = 0; elIter < (int)n; elIter++) {
            ASSERT_EQ(out_feat[elIter].f[0], gold_feat[elIter].f[0]) << "at: " << elIter << std::endl;
            ASSERT_EQ(out_feat[elIter].f[1], gold_feat[elIter].f[1]) << "at: " << elIter << std::endl;
            ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]), 1e-3) << "at: " << elIter << std::endl;
            ASSERT_LE(fabs(out_feat[elIter].f[3] - gold_feat[elIter].f[3]), 1e-3) << "at: " << elIter << std::endl;
            ASSERT_LE(fabs(out_feat[elIter].f[4] - gold_feat[elIter].f[4]), 1e-3) << "at: " << elIter << std::endl;
        }

        // TODO: improve distance for single/double-precision interchangeability
        EXPECT_TRUE(compareHamming(descSize, (unsigned*)&v_out_desc[0], (unsigned*)&v_gold_desc[0], 3));

        ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(inArray_f32));

        ASSERT_EQ(AF_SUCCESS, af_release_array(x));
        ASSERT_EQ(AF_SUCCESS, af_release_array(y));
        ASSERT_EQ(AF_SUCCESS, af_release_array(score));
        ASSERT_EQ(AF_SUCCESS, af_release_array(orientation));
        ASSERT_EQ(AF_SUCCESS, af_release_array(size));
        ASSERT_EQ(AF_SUCCESS, af_release_array(desc));

        delete[] outX;
        delete[] outY;
        delete[] outScore;
        delete[] outOrientation;
        delete[] outSize;
        delete[] outDesc;
    }
}

#define ORB_INIT(desc, image) \
    TYPED_TEST(ORB, desc) \
    {   \
        orbTest<TypeParam>(string(TEST_DIR"/orb/"#image".test"));   \
    }

    ORB_INIT(square, square);
    ORB_INIT(lena, lena);

///////////////////////////////////// CPP ////////////////////////////////
//
TEST(ORB, CPP)
{
    if (noDoubleTests<float>()) return;

    vector<dim4>             inDims;
    vector<string>           inFiles;
    vector<vector<float> >    goldFeat;
    vector<vector<unsigned> > goldDesc;

    readImageFeaturesDescriptors<unsigned>(string(TEST_DIR"/orb/square.test"), inDims, inFiles, goldFeat, goldDesc);
    inFiles[0].insert(0,string(TEST_DIR"/orb/"));

    af::array in = af::loadImage(inFiles[0].c_str(), false);

    af::features feat;
    af::array desc;
    af::orb(feat, desc, in, 20.0f, 400, 1.2f, 8, true);

    float * outX           = new float[feat.getNumFeatures()];
    float * outY           = new float[feat.getNumFeatures()];
    float * outScore       = new float[feat.getNumFeatures()];
    float * outOrientation = new float[feat.getNumFeatures()];
    float * outSize        = new float[feat.getNumFeatures()];
    unsigned * outDesc     = new unsigned[desc.elements()];
    feat.getX().host(outX);
    feat.getY().host(outY);
    feat.getScore().host(outScore);
    feat.getOrientation().host(outOrientation);
    feat.getSize().host(outSize);
    desc.host(outDesc);

    vector<feat_desc_t> out_feat_desc;
    array_to_feat_desc(out_feat_desc, outX, outY, outScore, outOrientation, outSize, outDesc, feat.getNumFeatures());

    vector<feat_desc_t> gold_feat_desc;
    array_to_feat_desc(gold_feat_desc, &goldFeat[0].front(), &goldFeat[1].front(), &goldFeat[2].front(), &goldFeat[3].front(), &goldFeat[4].front(), goldDesc, goldFeat[0].size());

    std::sort(out_feat_desc.begin(), out_feat_desc.end(), feat_cmp);
    std::sort(gold_feat_desc.begin(), gold_feat_desc.end(), feat_cmp);

    vector<feat_t> out_feat;
    vector<desc_t> v_out_desc;
    vector<feat_t> gold_feat;
    vector<desc_t> v_gold_desc;

    split_feat_desc(out_feat_desc, out_feat, v_out_desc);
    split_feat_desc(gold_feat_desc, gold_feat, v_gold_desc);

    for (int elIter = 0; elIter < (int)feat.getNumFeatures(); elIter++) {
        ASSERT_EQ(out_feat[elIter].f[0], gold_feat[elIter].f[0]) << "at: " << elIter << std::endl;
        ASSERT_EQ(out_feat[elIter].f[1], gold_feat[elIter].f[1]) << "at: " << elIter << std::endl;
        ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]), 1e-3) << "at: " << elIter << std::endl;
        ASSERT_LE(fabs(out_feat[elIter].f[3] - gold_feat[elIter].f[3]), 1e-3) << "at: " << elIter << std::endl;
        ASSERT_LE(fabs(out_feat[elIter].f[4] - gold_feat[elIter].f[4]), 1e-3) << "at: " << elIter << std::endl;
    }

    // TODO: improve distance for single/double-precision interchangeability
    EXPECT_TRUE(compareHamming(desc.elements(), (unsigned*)&v_out_desc[0], (unsigned*)&v_gold_desc[0], 3));

    delete[] outX;
    delete[] outY;
    delete[] outScore;
    delete[] outOrientation;
    delete[] outSize;
    delete[] outDesc;
}
