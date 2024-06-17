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

using af::array;
using af::dim4;
using af::features;
using af::loadImage;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

typedef struct {
    float f[5];
    unsigned d[272];
} feat_desc_t;

typedef struct {
    float f[5];
} feat_t;

typedef struct {
    float d[272];
} desc_t;

static bool feat_cmp(feat_desc_t i, feat_desc_t j) {
    for (int k = 0; k < 5; k++)
        if (round(i.f[k] * 1e1f) != round(j.f[k] * 1e1f))
            return (round(i.f[k] * 1e1f) < round(j.f[k] * 1e1f));

    return false;
}

static void array_to_feat_desc(vector<feat_desc_t>& feat, float* x, float* y,
                               float* score, float* ori, float* size,
                               float* desc, unsigned nfeat) {
    feat.resize(nfeat);
    for (size_t i = 0; i < feat.size(); i++) {
        feat[i].f[0] = x[i];
        feat[i].f[1] = y[i];
        feat[i].f[2] = score[i];
        feat[i].f[3] = ori[i];
        feat[i].f[4] = size[i];
        for (unsigned j = 0; j < 272; j++) feat[i].d[j] = desc[i * 272 + j];
    }
}

static void array_to_feat_desc(vector<feat_desc_t>& feat, float* x, float* y,
                               float* score, float* ori, float* size,
                               vector<vector<float>>& desc, unsigned nfeat) {
    feat.resize(nfeat);
    for (size_t i = 0; i < feat.size(); i++) {
        feat[i].f[0] = x[i];
        feat[i].f[1] = y[i];
        feat[i].f[2] = score[i];
        feat[i].f[3] = ori[i];
        feat[i].f[4] = size[i];
        for (unsigned j = 0; j < 272; j++) feat[i].d[j] = desc[i][j];
    }
}

static void split_feat_desc(vector<feat_desc_t>& fd, vector<feat_t>& f,
                            vector<desc_t>& d) {
    f.resize(fd.size());
    d.resize(fd.size());
    for (size_t i = 0; i < fd.size(); i++) {
        f[i].f[0] = fd[i].f[0];
        f[i].f[1] = fd[i].f[1];
        f[i].f[2] = fd[i].f[2];
        f[i].f[3] = fd[i].f[3];
        f[i].f[4] = fd[i].f[4];
        for (unsigned j = 0; j < 272; j++) d[i].d[j] = fd[i].d[j];
    }
}

static bool compareEuclidean(dim_t desc_len, dim_t ndesc, float* cpu,
                             float* gpu, float unit_thr = 1.f,
                             float euc_thr = 1.f) {
    bool ret  = true;
    float sum = 0.0f;

    for (dim_t i = 0; i < ndesc; i++) {
        sum = 0.0f;
        for (dim_t l = 0; l < desc_len; l++) {
            dim_t idx = i * desc_len + l;
            float x   = (cpu[idx] - gpu[idx]);
            sum += x * x;
            if (abs(x) > (float)unit_thr) {
                ret = false;
                cout << endl << "@compareEuclidean: unit mismatch." << endl;
                cout << "(cpu,gpu,cpu-gpu)[" << i << "," << l << "] : {"
                     << cpu[idx] << "," << gpu[idx] << ","
                     << cpu[idx] - gpu[idx] << "}" << endl;
                cout << endl;
                break;
            }
        }
        if (sqrt(sum) > euc_thr) {
            ret = false;
            cout << endl << "@compareEuclidean: distance mismatch." << endl;
            cout << "Euclidean distance: " << sqrt(sum) << endl;
        }
        if (ret == false) return ret;
    }

    return ret;
}

template<typename T>
class GLOH : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

typedef ::testing::Types<float, double> TestTypes;

TYPED_TEST_SUITE(GLOH, TestTypes);

template<typename T>
void glohTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<vector<float>> goldFeat;
    vector<vector<float>> goldDesc;

    readImageFeaturesDescriptors<float>(pTestFile, inDims, inFiles, goldFeat,
                                        goldDesc);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        af_array inArray_f32 = 0;
        af_array inArray     = 0;
        af_array desc        = 0;
        af_features feat;

        inFiles[testId].insert(0, string(TEST_DIR "/gloh/"));

        ASSERT_SUCCESS(
            af_load_image(&inArray_f32, inFiles[testId].c_str(), false));
        ASSERT_SUCCESS(conv_image<T>(&inArray, inArray_f32));

        ASSERT_SUCCESS_CHECK_SUPRT(af_gloh(&feat, &desc, inArray, 3,
                                           0.04f, 10.0f, 1.6f,
                                           true, 1.f / 256.f, 0.05f));

        dim_t n = 0;
        af_array x, y, score, orientation, size;

        ASSERT_SUCCESS(af_get_features_num(&n, feat));
        ASSERT_SUCCESS(af_get_features_xpos(&x, feat));
        ASSERT_SUCCESS(af_get_features_ypos(&y, feat));
        ASSERT_SUCCESS(af_get_features_score(&score, feat));
        ASSERT_SUCCESS(af_get_features_orientation(&orientation, feat));
        ASSERT_SUCCESS(af_get_features_size(&size, feat));

        float* outX           = new float[n];
        float* outY           = new float[n];
        float* outScore       = new float[n];
        float* outOrientation = new float[n];
        float* outSize        = new float[n];
        dim_t descSize;
        dim_t descDims[4];
        ASSERT_SUCCESS(af_get_elements(&descSize, desc));
        ASSERT_SUCCESS(af_get_dims(&descDims[0], &descDims[1], &descDims[2],
                                   &descDims[3], desc));
        float* outDesc = new float[descSize];
        ASSERT_SUCCESS(af_get_data_ptr((void*)outX, x));
        ASSERT_SUCCESS(af_get_data_ptr((void*)outY, y));
        ASSERT_SUCCESS(af_get_data_ptr((void*)outScore, score));
        ASSERT_SUCCESS(af_get_data_ptr((void*)outOrientation, orientation));
        ASSERT_SUCCESS(af_get_data_ptr((void*)outSize, size));
        ASSERT_SUCCESS(af_get_data_ptr((void*)outDesc, desc));

        vector<feat_desc_t> out_feat_desc;
        array_to_feat_desc(out_feat_desc, outX, outY, outScore, outOrientation,
                           outSize, outDesc, n);

        vector<feat_desc_t> gold_feat_desc;
        array_to_feat_desc(gold_feat_desc, &goldFeat[0].front(),
                           &goldFeat[1].front(), &goldFeat[2].front(),
                           &goldFeat[3].front(), &goldFeat[4].front(), goldDesc,
                           goldFeat[0].size());

        std::stable_sort(out_feat_desc.begin(), out_feat_desc.end(), feat_cmp);
        std::stable_sort(gold_feat_desc.begin(), gold_feat_desc.end(),
                         feat_cmp);

        vector<feat_t> out_feat;
        vector<desc_t> v_out_desc;
        vector<feat_t> gold_feat;
        vector<desc_t> v_gold_desc;

        split_feat_desc(out_feat_desc, out_feat, v_out_desc);
        split_feat_desc(gold_feat_desc, gold_feat, v_gold_desc);

        for (int elIter = 0; elIter < (int)n; elIter++) {
            ASSERT_LE(fabs(out_feat[elIter].f[0] - gold_feat[elIter].f[0]),
                      1e-3)
                << "at: " << elIter << endl;
            ASSERT_LE(fabs(out_feat[elIter].f[1] - gold_feat[elIter].f[1]),
                      1e-3)
                << "at: " << elIter << endl;
            ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]),
                      1e-3)
                << "at: " << elIter << endl;
            ASSERT_LE(fabs(out_feat[elIter].f[3] - gold_feat[elIter].f[3]),
                      0.5f)
                << "at: " << elIter << endl;
            ASSERT_LE(fabs(out_feat[elIter].f[4] - gold_feat[elIter].f[4]),
                      1e-3)
                << "at: " << elIter << endl;
        }

        EXPECT_TRUE(compareEuclidean(descDims[0], descDims[1],
                                     (float*)&v_out_desc[0],
                                     (float*)&v_gold_desc[0], 2.f, 5.5f));

        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(inArray_f32));

        ASSERT_SUCCESS(af_release_array(desc));
        ASSERT_SUCCESS(af_release_features(feat));

        delete[] outX;
        delete[] outY;
        delete[] outScore;
        delete[] outOrientation;
        delete[] outSize;
        delete[] outDesc;
    }
}

#define GLOH_INIT(desc, image)                                         \
    TYPED_TEST(GLOH, desc) {                                           \
        glohTest<TypeParam>(string(TEST_DIR "/gloh/" #image ".test")); \
    }

GLOH_INIT(man, man);

///////////////////////////////////// CPP ////////////////////////////////
//
TEST(GLOH, CPP) {
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<vector<float>> goldFeat;
    vector<vector<float>> goldDesc;

    readImageFeaturesDescriptors<float>(string(TEST_DIR "/gloh/man.test"),
                                        inDims, inFiles, goldFeat, goldDesc);
    inFiles[0].insert(0, string(TEST_DIR "/gloh/"));

    array in = loadImage(inFiles[0].c_str(), false);

    features feat;
    array desc;
    try { gloh(feat, desc, in, 3, 0.04f, 10.0f, 1.6f, true, 1.f / 256.f, 0.05f);
    } catch FUNCTION_UNSUPPORTED

    float* outX           = new float[feat.getNumFeatures()];
    float* outY           = new float[feat.getNumFeatures()];
    float* outScore       = new float[feat.getNumFeatures()];
    float* outOrientation = new float[feat.getNumFeatures()];
    float* outSize        = new float[feat.getNumFeatures()];
    float* outDesc        = new float[desc.elements()];
    dim4 descDims         = desc.dims();
    feat.getX().host(outX);
    feat.getY().host(outY);
    feat.getScore().host(outScore);
    feat.getOrientation().host(outOrientation);
    feat.getSize().host(outSize);
    desc.host(outDesc);

    vector<feat_desc_t> out_feat_desc;
    array_to_feat_desc(out_feat_desc, outX, outY, outScore, outOrientation,
                       outSize, outDesc, feat.getNumFeatures());

    vector<feat_desc_t> gold_feat_desc;
    array_to_feat_desc(gold_feat_desc, &goldFeat[0].front(),
                       &goldFeat[1].front(), &goldFeat[2].front(),
                       &goldFeat[3].front(), &goldFeat[4].front(), goldDesc,
                       goldFeat[0].size());

    std::stable_sort(out_feat_desc.begin(), out_feat_desc.end(), feat_cmp);
    std::stable_sort(gold_feat_desc.begin(), gold_feat_desc.end(), feat_cmp);

    vector<feat_t> out_feat;
    vector<desc_t> v_out_desc;
    vector<feat_t> gold_feat;
    vector<desc_t> v_gold_desc;

    split_feat_desc(out_feat_desc, out_feat, v_out_desc);
    split_feat_desc(gold_feat_desc, gold_feat, v_gold_desc);

    for (int elIter = 0; elIter < (int)feat.getNumFeatures(); elIter++) {
        ASSERT_LE(fabs(out_feat[elIter].f[0] - gold_feat[elIter].f[0]), 1e-3)
            << "at: " << elIter << endl;
        ASSERT_LE(fabs(out_feat[elIter].f[1] - gold_feat[elIter].f[1]), 1e-3)
            << "at: " << elIter << endl;
        ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]), 1e-3)
            << "at: " << elIter << endl;
        ASSERT_LE(fabs(out_feat[elIter].f[3] - gold_feat[elIter].f[3]), 0.5f)
            << "at: " << elIter << endl;
        ASSERT_LE(fabs(out_feat[elIter].f[4] - gold_feat[elIter].f[4]), 1e-3)
            << "at: " << elIter << endl;
    }

    EXPECT_TRUE(compareEuclidean(descDims[0], descDims[1],
                                 (float*)&v_out_desc[0],
                                 (float*)&v_gold_desc[0], 2.f, 5.5f));

    delete[] outX;
    delete[] outY;
    delete[] outScore;
    delete[] outOrientation;
    delete[] outSize;
    delete[] outDesc;
}
