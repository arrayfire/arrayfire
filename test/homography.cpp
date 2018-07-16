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

using af::array;
using af::constant;
using af::dim4;
using af::span;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
int print(vector<T> arr) {
    std::cout << "\n";
    copy(arr.begin(), arr.end(), std::ostream_iterator<float>(cout, ", "));
    return 0;
}

template<typename T>
class Homography : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

typedef ::testing::Types<float, double> TestTypes;

TYPED_TEST_CASE(Homography, TestTypes);

template<typename T>
array perspectiveTransform(dim4 inDims, array H_matrix)
{
    T d0 = (T)inDims[0];
    T d1 = (T)inDims[1];
    return transformCoordinates(H_matrix, d0, d1);
}

template<typename T>
void homographyTest(string pTestFile, const af_homography_type htype,
                    const bool rotate, const float size_ratio)
{
    using af::Pi;
    using af::dtype_traits;

    if (noDoubleTests<T>()) return;
    if (noImageIOTests()) return;

    vector<dim4>           inDims;
    vector<string>         inFiles;
    vector<vector<float> > gold;

    readImageTests(pTestFile, inDims, inFiles, gold);

    inFiles[0].insert(0,string(TEST_DIR"/homography/"));

    af_array trainArray_f32   = 0;
    af_array trainArray       = 0;
    af_array train_desc       = 0;
    af_array train_feat_x     = 0;
    af_array train_feat_y     = 0;
    af_features train_feat;

    ASSERT_EQ(AF_SUCCESS, af_load_image(&trainArray_f32, inFiles[0].c_str(), false));
    ASSERT_EQ(AF_SUCCESS, conv_image<T>(&trainArray, trainArray_f32));

    ASSERT_EQ(AF_SUCCESS, af_orb(&train_feat, &train_desc, trainArray, 20.0f, 2000, 1.2f, 8, true));

    ASSERT_EQ(AF_SUCCESS, af_get_features_xpos(&train_feat_x, train_feat));
    ASSERT_EQ(AF_SUCCESS, af_get_features_ypos(&train_feat_y, train_feat));

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
    af_array H_matrix                = 0;
    af_array train_feat_x_idx = 0;
    af_array train_feat_y_idx = 0;
    af_array query_feat_x_idx = 0;
    af_array query_feat_y_idx = 0;
    af_features query_feat;

    const float theta = Pi * 0.5f;
    const dim_t test_d0 = inDims[0][0] * size_ratio;
    const dim_t test_d1 = inDims[0][1] * size_ratio;
    const dim_t tDims[] = {test_d0, test_d1};
    if (rotate)
        ASSERT_EQ(AF_SUCCESS, af_rotate(&queryArray, trainArray, theta, false, AF_INTERP_NEAREST));
    else
        ASSERT_EQ(AF_SUCCESS, af_resize(&queryArray, trainArray, test_d0, test_d1, AF_INTERP_BILINEAR));

    ASSERT_EQ(AF_SUCCESS, af_orb(&query_feat, &query_desc, queryArray, 20.0f, 2000, 1.2f, 8, true));

    ASSERT_EQ(AF_SUCCESS, af_hamming_matcher(&idx, &dist, train_desc, query_desc, 0, 1));

    dim_t distDims[4];
    ASSERT_EQ(AF_SUCCESS, af_get_dims(&distDims[0], &distDims[1], &distDims[2], &distDims[3], dist));

    ASSERT_EQ(AF_SUCCESS, af_constant(&const_50, 50, 2, distDims, u32));
    ASSERT_EQ(AF_SUCCESS, af_lt(&dist_thr, dist, const_50, false));
    ASSERT_EQ(AF_SUCCESS, af_where(&train_idx, dist_thr));

    dim_t tidxDims[4];
    ASSERT_EQ(AF_SUCCESS, af_get_dims(&tidxDims[0], &tidxDims[1], &tidxDims[2], &tidxDims[3], train_idx));
    af_index_t tindexs;
    tindexs.isSeq = false;
    tindexs.idx.seq = af_make_seq(0, tidxDims[0]-1, 1);
    tindexs.idx.arr = train_idx;
    ASSERT_EQ(AF_SUCCESS, af_index_gen(&query_idx, idx, 1, &tindexs));

    ASSERT_EQ(AF_SUCCESS, af_get_features_xpos(&query_feat_x, query_feat));
    ASSERT_EQ(AF_SUCCESS, af_get_features_ypos(&query_feat_y, query_feat));

    dim_t qidxDims[4];
    ASSERT_EQ(AF_SUCCESS, af_get_dims(&qidxDims[0], &qidxDims[1], &qidxDims[2], &qidxDims[3], query_idx));
    af_index_t qindexs;
    qindexs.isSeq = false;
    qindexs.idx.seq = af_make_seq(0, qidxDims[0]-1, 1);
    qindexs.idx.arr = query_idx;

    ASSERT_EQ(AF_SUCCESS, af_index_gen(&train_feat_x_idx, train_feat_x, 1, &tindexs));
    ASSERT_EQ(AF_SUCCESS, af_index_gen(&train_feat_y_idx, train_feat_y, 1, &tindexs));
    ASSERT_EQ(AF_SUCCESS, af_index_gen(&query_feat_x_idx, query_feat_x, 1, &qindexs));
    ASSERT_EQ(AF_SUCCESS, af_index_gen(&query_feat_y_idx, query_feat_y, 1, &qindexs));

    int inliers = 0;
    ASSERT_EQ(AF_SUCCESS, af_homography(&H_matrix, &inliers, train_feat_x_idx, train_feat_y_idx,
                                        query_feat_x_idx, query_feat_y_idx, htype,
                                        3.0f, 1000, (af_dtype) dtype_traits<T>::af_type));

    array HH(H_matrix);

    array t = perspectiveTransform<T>(inDims[0], HH);

    T* gold_t = new T[8];
    for (int i = 0; i < 8; i++)
        gold_t[i] = (T)0;
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
        ASSERT_LE(fabs(out_t[elIter] - gold_t[elIter]) / tDims[elIter & 1], 0.25f)
            << "at: " << elIter << endl;
    }

    delete[] gold_t;
    delete[] out_t;

    ASSERT_EQ(AF_SUCCESS, af_release_array(queryArray));

    ASSERT_EQ(AF_SUCCESS, af_release_array(query_desc));
    ASSERT_EQ(AF_SUCCESS, af_release_array(idx));
    ASSERT_EQ(AF_SUCCESS, af_release_array(dist));
    ASSERT_EQ(AF_SUCCESS, af_release_array(const_50));
    ASSERT_EQ(AF_SUCCESS, af_release_array(dist_thr));
    ASSERT_EQ(AF_SUCCESS, af_release_array(train_idx));
    ASSERT_EQ(AF_SUCCESS, af_release_array(query_idx));
    ASSERT_EQ(AF_SUCCESS, af_release_features(query_feat));
    ASSERT_EQ(AF_SUCCESS, af_release_features(train_feat));
    ASSERT_EQ(AF_SUCCESS, af_release_array(train_feat_x_idx));
    ASSERT_EQ(AF_SUCCESS, af_release_array(train_feat_y_idx));
    ASSERT_EQ(AF_SUCCESS, af_release_array(query_feat_x_idx));
    ASSERT_EQ(AF_SUCCESS, af_release_array(query_feat_y_idx));

    ASSERT_EQ(AF_SUCCESS, af_release_array(trainArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(trainArray_f32));
    ASSERT_EQ(AF_SUCCESS, af_release_array(train_desc));
}

#define HOMOGRAPHY_INIT(desc, image, htype, rotate, size_ratio)                 \
    TYPED_TEST(Homography, desc)                                                \
    {                                                                           \
        homographyTest<TypeParam>(string(TEST_DIR"/homography/"#image".test"),  \
                                  htype, rotate, size_ratio);                   \
    }

    HOMOGRAPHY_INIT(Tux_RANSAC, tux, AF_HOMOGRAPHY_RANSAC, false, 1.0f);
    HOMOGRAPHY_INIT(DISABLED_Tux_RANSAC_90degrees, tux, AF_HOMOGRAPHY_RANSAC, true, 1.0f);
    HOMOGRAPHY_INIT(Tux_RANSAC_resize, tux, AF_HOMOGRAPHY_RANSAC, false, 1.5f);
    //HOMOGRAPHY_INIT(Tux_LMedS, tux, AF_HOMOGRAPHY_LMEDS, false, 1.0f);
    //HOMOGRAPHY_INIT(Tux_LMedS_90degrees, tux, AF_HOMOGRAPHY_LMEDS, true, 1.0f);
    //HOMOGRAPHY_INIT(Tux_LMedS_resize, tux, AF_HOMOGRAPHY_LMEDS, false, 1.5f);

///////////////////////////////////// CPP ////////////////////////////////
//

using af::features;
using af::loadImage;

TEST(Homography, CPP)
{
    if (noImageIOTests()) return;

    vector<dim4>           inDims;
    vector<string>         inFiles;
    vector<vector<float> > gold;

    readImageTests(string(TEST_DIR"/homography/tux.test"), inDims, inFiles, gold);

    inFiles[0].insert(0,string(TEST_DIR"/homography/"));

    const float size_ratio = 0.5f;

    array train_img = loadImage(inFiles[0].c_str(), false);
    array query_img = resize(size_ratio, train_img);
    dim4 tDims = train_img.dims();

    features feat_train, feat_query;
    array desc_train, desc_query;
    orb(feat_train, desc_train, train_img, 20, 2000, 1.2, 8, true);
    orb(feat_query, desc_query, query_img, 20, 2000, 1.2, 8, true);

    array idx, dist;
    hammingMatcher(idx, dist, desc_train, desc_query, 0, 1);

    array train_idx = where(dist < 30);
    array query_idx = idx(train_idx);

    array feat_train_x = feat_train.getX()(train_idx);
    array feat_train_y = feat_train.getY()(train_idx);
    array feat_train_score = feat_train.getScore()(train_idx);
    array feat_train_orientation = feat_train.getOrientation()(train_idx);
    array feat_train_size = feat_train.getSize()(train_idx);
    array feat_query_x = feat_query.getX()(query_idx);
    array feat_query_y = feat_query.getY()(query_idx);
    array feat_query_score = feat_query.getScore()(query_idx);
    array feat_query_orientation = feat_query.getOrientation()(query_idx);
    array feat_query_size = feat_query.getSize()(query_idx);

    array H_matrix;
    int inliers = 0;
    homography(H_matrix, inliers, feat_train_x, feat_train_y, feat_query_x, feat_query_y, AF_HOMOGRAPHY_RANSAC, 3.0f, 1000, f32);

    float* gold_t = new float[8];
    for (int i = 0; i < 8; i++)
        gold_t[i] = 0.f;
    gold_t[2] = tDims[1] * size_ratio;
    gold_t[3] = tDims[1] * size_ratio;
    gold_t[5] = tDims[0] * size_ratio;
    gold_t[6] = tDims[0] * size_ratio;

    array t = perspectiveTransform<float>(train_img.dims(), H_matrix);

    float* out_t = new float[4*2];
    t.host(out_t);

    for (int elIter = 0; elIter < 8; elIter++) {
        ASSERT_LE(fabs(out_t[elIter] - gold_t[elIter]) / tDims[elIter & 1], 0.1f)
            << "at: " << elIter << endl;
    }

    delete[] gold_t;
    delete[] out_t;
}

TEST(Homography, Rotation_90degCCW) {
    int nfeats = 200;

    // Generate points to rotate
    array in_x = af::randu(nfeats);
    array in_y = af::randu(nfeats);
    array ones = constant(1.f, in_x.elements());
    array in_xy = join(0, in_x.T(), in_y.T(), ones.T());

    // Build transformation matrix for 90deg CCW rotation
    float hh_rot_matrix[] = {0.f, -1.f, 0.f,
                             1.f,  0.f, 0.f,
                             0.f,  0.f, 1.f};
    vector<float> h_rot_matrix(hh_rot_matrix, hh_rot_matrix + 9);
    array rot_matrix(3, 3, h_rot_matrix.data());
    rot_matrix = rot_matrix.T();

    // Do the rotation using the hardcoded
    //  transformation matrix
    array in_rot = matmul(rot_matrix, in_xy);
    array in_rot_x = in_rot(0, span).T();
    array in_rot_y = in_rot(1, span).T();
    in_rot = in_rot.T();

    // Use homography to extract the transformation
    //  between generated points and their rotated versions
    array H_matrix;
    int inliers = 0;
    homography(H_matrix, inliers, in_x, in_y, in_rot_x, in_rot_y,
               AF_HOMOGRAPHY_RANSAC, 3.0f, 1000, f32);

    // Do the rotation using the transformation
    //  matrix returned by homography()
    H_matrix = H_matrix.T();
    array in_rot_H = matmul(H_matrix.as(f32), in_xy);
    in_rot_H /= tile(in_rot_H(2, span), 3);

    // Get the hardcoded transformation's results
    //  back to the host
    vector<float> h_in_rot(in_rot.elements());
    vector<float> h_in_rot_x(in_rot_x.elements());
    vector<float> h_in_rot_y(in_rot_y.elements());
    in_rot.host(h_in_rot.data());
    in_rot_x.host(h_in_rot_x.data());
    in_rot_y.host(h_in_rot_y.data());

    // Get the results due to homography back to
    //  the host
    vector<float> h_in_rot_H_x(in_rot_H.dims()[1]);
    vector<float> h_in_rot_H_y(in_rot_H.dims()[1]);
    in_rot_H(0, span).host(h_in_rot_H_x.data());
    in_rot_H(1, span).host(h_in_rot_H_y.data());

    // Compare results - output points using homography's
    //  transformation matrix must match those of the
    //  hardcoded transformation matrix
    ASSERT_EQ(in_rot_H.dims()[1], nfeats);
    for (int i = 0; i < nfeats; ++i) {
        ASSERT_NEAR(h_in_rot_H_x[i] - h_in_rot_x[i], 0.f, 1.f) << "at: " << i << " " << print<float>(h_in_rot_H_x) << print(h_in_rot_x) << endl;
        ASSERT_NEAR(h_in_rot_H_y[i] - h_in_rot_y[i], 0.f, 1.f) << "at: " << i << " " << print<float>(h_in_rot_H_y) << print(h_in_rot_y) << endl;
    }
}
