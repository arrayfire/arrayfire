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

using af::array;
using af::dim4;
using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Homography : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

typedef ::testing::Types<float, double> TestTypes;

TYPED_TEST_SUITE(Homography, TestTypes);

template<typename T>
array perspectiveTransform(dim4 inDims, array H) {
    T d0 = (T)inDims[0];
    T d1 = (T)inDims[1];
    return transformCoordinates(H, d0, d1);
}

template<typename T>
void homographyTest(string pTestFile, const af_homography_type htype,
                    const bool rotate, const float size_ratio) {
    using af::dtype_traits;
    using af::Pi;

    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<vector<float>> gold;

    readImageTests(pTestFile, inDims, inFiles, gold);

    inFiles[0].insert(0, string(TEST_DIR "/homography/"));

    af_array trainArray_f32 = 0;
    af_array trainArray     = 0;
    af_array train_desc     = 0;
    af_array train_feat_x   = 0;
    af_array train_feat_y   = 0;
    af_features train_feat;

    ASSERT_SUCCESS(af_load_image(&trainArray_f32, inFiles[0].c_str(), false));
    ASSERT_SUCCESS(conv_image<T>(&trainArray, trainArray_f32));

    ASSERT_SUCCESS_CHECK_SUPRT(af_orb(&train_feat, &train_desc, trainArray,
                               20.0f, 2000, 1.2f, 8, true));

    ASSERT_SUCCESS(af_get_features_xpos(&train_feat_x, train_feat));
    ASSERT_SUCCESS(af_get_features_ypos(&train_feat_y, train_feat));

    af_array queryArray       = 0;
    af_array query_desc       = 0;
    af_array idx              = 0;
    af_array dist             = 0;
    af_array const_50         = 0;
    af_array dist_thr         = 0;
    af_array train_idx        = 0;
    af_array query_idx        = 0;
    af_array query_feat_x     = 0;
    af_array query_feat_y     = 0;
    af_array H                = 0;
    af_array train_feat_x_idx = 0;
    af_array train_feat_y_idx = 0;
    af_array query_feat_x_idx = 0;
    af_array query_feat_y_idx = 0;
    af_features query_feat;

    const float theta   = Pi * 0.5f;
    const dim_t test_d0 = inDims[0][0] * size_ratio;
    const dim_t test_d1 = inDims[0][1] * size_ratio;
    const dim_t tDims[] = {test_d0, test_d1};
    if (rotate)
        ASSERT_SUCCESS(af_rotate(&queryArray, trainArray, theta, false,
                                 AF_INTERP_NEAREST));
    else
        ASSERT_SUCCESS(af_resize(&queryArray, trainArray, test_d0, test_d1,
                                 AF_INTERP_BILINEAR));

    ASSERT_SUCCESS_CHECK_SUPRT(af_orb(&query_feat, &query_desc, queryArray,
                                      20.0f, 2000, 1.2f, 8, true));

    ASSERT_SUCCESS(
        af_hamming_matcher(&idx, &dist, train_desc, query_desc, 0, 1));

    dim_t distDims[4];
    ASSERT_SUCCESS(af_get_dims(&distDims[0], &distDims[1], &distDims[2],
                               &distDims[3], dist));

    ASSERT_SUCCESS(af_constant(&const_50, 50, 2, distDims, u32));
    ASSERT_SUCCESS(af_lt(&dist_thr, dist, const_50, false));
    ASSERT_SUCCESS(af_where(&train_idx, dist_thr));

    dim_t tidxDims[4];
    ASSERT_SUCCESS(af_get_dims(&tidxDims[0], &tidxDims[1], &tidxDims[2],
                               &tidxDims[3], train_idx));
    af_index_t tindexs;
    tindexs.isSeq   = false;
    tindexs.idx.seq = af_make_seq(0, tidxDims[0] - 1, 1);
    tindexs.idx.arr = train_idx;
    ASSERT_SUCCESS(af_index_gen(&query_idx, idx, 1, &tindexs));

    ASSERT_SUCCESS(af_get_features_xpos(&query_feat_x, query_feat));
    ASSERT_SUCCESS(af_get_features_ypos(&query_feat_y, query_feat));

    dim_t qidxDims[4];
    ASSERT_SUCCESS(af_get_dims(&qidxDims[0], &qidxDims[1], &qidxDims[2],
                               &qidxDims[3], query_idx));
    af_index_t qindexs;
    qindexs.isSeq   = false;
    qindexs.idx.seq = af_make_seq(0, qidxDims[0] - 1, 1);
    qindexs.idx.arr = query_idx;

    ASSERT_SUCCESS(af_index_gen(&train_feat_x_idx, train_feat_x, 1, &tindexs));
    ASSERT_SUCCESS(af_index_gen(&train_feat_y_idx, train_feat_y, 1, &tindexs));
    ASSERT_SUCCESS(af_index_gen(&query_feat_x_idx, query_feat_x, 1, &qindexs));
    ASSERT_SUCCESS(af_index_gen(&query_feat_y_idx, query_feat_y, 1, &qindexs));

    int inliers = 0;
    ASSERT_SUCCESS_CHECK_SUPRT(af_homography(&H, &inliers, train_feat_x_idx,
                                             train_feat_y_idx, query_feat_x_idx,
                                             query_feat_y_idx, htype, 3.0f, 1000,
                                             (af_dtype)dtype_traits<T>::af_type));

    array HH(H);

    array t = perspectiveTransform<T>(inDims[0], HH);

    T* gold_t = new T[8];
    for (int i = 0; i < 8; i++) gold_t[i] = (T)0;
    if (rotate) {
        gold_t[1] = test_d0;
        gold_t[2] = test_d0;
        gold_t[4] = test_d1;
        gold_t[5] = test_d1;
    } else {
        gold_t[2] = test_d1;
        gold_t[3] = test_d1;
        gold_t[5] = test_d0;
        gold_t[6] = test_d0;
    }

    T* out_t = new T[8];
    t.host(out_t);

    for (int elIter = 0; elIter < 8; elIter++) {
        ASSERT_LE(fabs(out_t[elIter] - gold_t[elIter]) / tDims[elIter & 1],
                  0.25f)
            << "at: " << elIter << endl;
    }

    delete[] gold_t;
    delete[] out_t;

    ASSERT_SUCCESS(af_release_array(queryArray));

    ASSERT_SUCCESS(af_release_array(query_desc));
    ASSERT_SUCCESS(af_release_array(idx));
    ASSERT_SUCCESS(af_release_array(dist));
    ASSERT_SUCCESS(af_release_array(const_50));
    ASSERT_SUCCESS(af_release_array(dist_thr));
    ASSERT_SUCCESS(af_release_array(train_idx));
    ASSERT_SUCCESS(af_release_array(query_idx));
    ASSERT_SUCCESS(af_release_features(query_feat));
    ASSERT_SUCCESS(af_release_features(train_feat));
    ASSERT_SUCCESS(af_release_array(train_feat_x_idx));
    ASSERT_SUCCESS(af_release_array(train_feat_y_idx));
    ASSERT_SUCCESS(af_release_array(query_feat_x_idx));
    ASSERT_SUCCESS(af_release_array(query_feat_y_idx));

    ASSERT_SUCCESS(af_release_array(trainArray));
    ASSERT_SUCCESS(af_release_array(trainArray_f32));
    ASSERT_SUCCESS(af_release_array(train_desc));
}

#define HOMOGRAPHY_INIT(desc, image, htype, rotate, size_ratio)            \
    TYPED_TEST(Homography, desc) {                                         \
        homographyTest<TypeParam>(                                         \
            string(TEST_DIR "/homography/" #image ".test"), htype, rotate, \
            size_ratio);                                                   \
    }

HOMOGRAPHY_INIT(Tux_RANSAC, tux, AF_HOMOGRAPHY_RANSAC, false, 1.0f);
HOMOGRAPHY_INIT(Tux_RANSAC_90degrees, tux, AF_HOMOGRAPHY_RANSAC, true, 1.0f);
HOMOGRAPHY_INIT(Tux_RANSAC_resize, tux, AF_HOMOGRAPHY_RANSAC, false, 1.5f);
// HOMOGRAPHY_INIT(Tux_LMedS, tux, AF_HOMOGRAPHY_LMEDS, false, 1.0f);
// HOMOGRAPHY_INIT(Tux_LMedS_90degrees, tux, AF_HOMOGRAPHY_LMEDS, true, 1.0f);
// HOMOGRAPHY_INIT(Tux_LMedS_resize, tux, AF_HOMOGRAPHY_LMEDS, false, 1.5f);

///////////////////////////////////// CPP ////////////////////////////////
//

using af::features;
using af::loadImage;

TEST(Homography, CPP) {
    IMAGEIO_ENABLED_CHECK();

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<vector<float>> gold;

    readImageTests(string(TEST_DIR "/homography/tux.test"), inDims, inFiles,
                   gold);

    inFiles[0].insert(0, string(TEST_DIR "/homography/"));

    const float size_ratio = 0.5f;

    array train_img = loadImage(inFiles[0].c_str(), false);
    array query_img = resize(size_ratio, train_img);
    dim4 tDims      = train_img.dims();

    features feat_train, feat_query;
    array desc_train, desc_query;
    try { orb(feat_train, desc_train, train_img, 20, 2000, 1.2, 8, true);
          orb(feat_query, desc_query, query_img, 20, 2000, 1.2, 8, true);
    } catch FUNCTION_UNSUPPORTED

    array idx, dist;
    hammingMatcher(idx, dist, desc_train, desc_query, 0, 1);

    array train_idx = where(dist < 30);
    array query_idx = idx(train_idx);

    array feat_train_x           = feat_train.getX()(train_idx);
    array feat_train_y           = feat_train.getY()(train_idx);
    array feat_train_score       = feat_train.getScore()(train_idx);
    array feat_train_orientation = feat_train.getOrientation()(train_idx);
    array feat_train_size        = feat_train.getSize()(train_idx);
    array feat_query_x           = feat_query.getX()(query_idx);
    array feat_query_y           = feat_query.getY()(query_idx);
    array feat_query_score       = feat_query.getScore()(query_idx);
    array feat_query_orientation = feat_query.getOrientation()(query_idx);
    array feat_query_size        = feat_query.getSize()(query_idx);

    array H;
    int inliers = 0;
    try { homography(H, inliers, feat_train_x, feat_train_y, feat_query_x,
                     feat_query_y, AF_HOMOGRAPHY_RANSAC, 3.0f, 1000, f32);
    } catch FUNCTION_UNSUPPORTED

    float* gold_t = new float[8];
    for (int i = 0; i < 8; i++) gold_t[i] = 0.f;
    gold_t[2] = tDims[1] * size_ratio;
    gold_t[3] = tDims[1] * size_ratio;
    gold_t[5] = tDims[0] * size_ratio;
    gold_t[6] = tDims[0] * size_ratio;

    array t = perspectiveTransform<float>(train_img.dims(), H);

    float* out_t = new float[4 * 2];
    t.host(out_t);

    for (int elIter = 0; elIter < 8; elIter++) {
        ASSERT_LE(fabs(out_t[elIter] - gold_t[elIter]) / tDims[elIter & 1],
                  0.1f)
            << "at: " << elIter << endl;
    }

    delete[] gold_t;
    delete[] out_t;
}
