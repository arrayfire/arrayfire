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
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::dim4;
using af::dtype_traits;
using af::loadImage;
using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Transform : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

template<typename T>
class TransformInt : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

typedef ::testing::Types<float, double> TestTypes;
typedef ::testing::Types<int, intl, uint, uintl, short, ushort, uchar>
    TestTypesInt;

TYPED_TEST_SUITE(Transform, TestTypes);
TYPED_TEST_SUITE(TransformInt, TestTypesInt);

template<typename T>
void genTestData(af_array *gold, af_array *in, af_array *transform,
                 dim_t *odim0, dim_t *odim1, string pTestFile,
                 string pHomographyFile) {
    vector<dim4> inNumDims;
    vector<string> inFiles;
    vector<dim_t> goldNumDims;
    vector<string> goldFiles;

    readImageTests(pTestFile, inNumDims, inFiles, goldNumDims, goldFiles);

    inFiles[0].insert(0, string(TEST_DIR "/transform/"));
    inFiles[1].insert(0, string(TEST_DIR "/transform/"));
    goldFiles[0].insert(0, string(TEST_DIR "/transform/"));

    dim4 objDims = inNumDims[0];

    vector<dim4> HNumDims;
    vector<vector<float>> HIn;
    vector<vector<float>> HTests;
    readTests<float, float, float>(pHomographyFile, HNumDims, HIn, HTests);

    dim4 HDims = HNumDims[0];

    af_array sceneArray_f32 = 0;
    af_array goldArray_f32  = 0;
    af_array sceneArray     = 0;
    af_array goldArray      = 0;
    af_array HArray         = 0;

    ASSERT_SUCCESS(af_load_image(&sceneArray_f32, inFiles[1].c_str(), false));
    ASSERT_SUCCESS(af_load_image(&goldArray_f32, goldFiles[0].c_str(), false));

    ASSERT_SUCCESS(conv_image<T>(&sceneArray, sceneArray_f32));
    ASSERT_SUCCESS(conv_image<T>(&goldArray, goldArray_f32));

    ASSERT_SUCCESS(af_create_array(&HArray, &(HIn[0].front()), HDims.ndims(),
                                   HDims.get(), f32));

    *gold      = goldArray;
    *in        = sceneArray;
    *transform = HArray;
    *odim0     = objDims[0];
    *odim1     = objDims[1];

    if (goldArray_f32 != 0) af_release_array(goldArray_f32);
    if (sceneArray_f32 != 0) af_release_array(sceneArray_f32);
}

template<typename T>
void transformTest(string pTestFile, string pHomographyFile,
                   const af_interp_type method, const bool invert) {
    SUPPORTED_TYPE_CHECK(T);
    if (noImageIOTests()) return;

    af_array sceneArray = 0;
    af_array goldArray  = 0;
    af_array outArray   = 0;
    af_array HArray     = 0;

    dim_t odim0 = 0;
    dim_t odim1 = 0;

    genTestData<T>(&goldArray, &sceneArray, &HArray, &odim0, &odim1, pTestFile,
                   pHomographyFile);

    ASSERT_SUCCESS(af_transform(&outArray, sceneArray, HArray, odim0, odim1,
                                method, invert));

    // Get gold data
    dim_t goldEl = 0;
    ASSERT_SUCCESS(af_get_elements(&goldEl, goldArray));
    vector<T> goldData(goldEl);
    ASSERT_SUCCESS(af_get_data_ptr((void *)&goldData.front(), goldArray));

    // Get result
    dim_t outEl = 0;
    ASSERT_SUCCESS(af_get_elements(&outEl, outArray));
    vector<T> outData(outEl);
    ASSERT_SUCCESS(af_get_data_ptr((void *)&outData.front(), outArray));

    const float thr = 1.1f;

    // Maximum number of wrong pixels must be <= 0.01% of number of elements,
    // this metric is necessary due to rounding errors between different
    // backends for AF_INTERP_NEAREST and AF_INTERP_LOWER
    const size_t maxErr = goldEl * 0.0001f;
    size_t err          = 0;

    for (dim_t elIter = 0; elIter < goldEl; elIter++) {
        err += fabs((float)floor(outData[elIter]) -
                    (float)floor(goldData[elIter])) > thr;
        if (err > maxErr) {
            ASSERT_LE(err, maxErr) << "at: " << elIter << endl;
        }
    }

    if (HArray != 0) { af_release_array(HArray); }
    if (outArray != 0) { af_release_array(outArray); }
    if (goldArray != 0) { af_release_array(goldArray); }
    if (sceneArray != 0) { af_release_array(sceneArray); }
}

TYPED_TEST(Transform, PerspectiveNearest) {
    transformTest<TypeParam>(string(TEST_DIR "/transform/tux_nearest.test"),
                             string(TEST_DIR "/transform/tux_tmat.test"),
                             AF_INTERP_NEAREST, false);
}

TYPED_TEST(Transform, PerspectiveBilinear) {
    transformTest<TypeParam>(string(TEST_DIR "/transform/tux_bilinear.test"),
                             string(TEST_DIR "/transform/tux_tmat.test"),
                             AF_INTERP_BILINEAR, false);
}

TYPED_TEST(Transform, PerspectiveLower) {
    transformTest<TypeParam>(string(TEST_DIR "/transform/tux_lower.test"),
                             string(TEST_DIR "/transform/tux_tmat.test"),
                             AF_INTERP_LOWER, false);
}

TYPED_TEST(Transform, PerspectiveNearestInvert) {
    transformTest<TypeParam>(
        string(TEST_DIR "/transform/tux_nearest.test"),
        string(TEST_DIR "/transform/tux_tmat_inverse.test"), AF_INTERP_NEAREST,
        true);
}

TYPED_TEST(Transform, PerspectiveBilinearInvert) {
    transformTest<TypeParam>(
        string(TEST_DIR "/transform/tux_bilinear.test"),
        string(TEST_DIR "/transform/tux_tmat_inverse.test"), AF_INTERP_BILINEAR,
        true);
}

TYPED_TEST(Transform, PerspectiveLowerInvert) {
    transformTest<TypeParam>(
        string(TEST_DIR "/transform/tux_lower.test"),
        string(TEST_DIR "/transform/tux_tmat_inverse.test"), AF_INTERP_LOWER,
        true);
}

TYPED_TEST(TransformInt, PerspectiveNearest) {
    transformTest<TypeParam>(string(TEST_DIR "/transform/tux_nearest.test"),
                             string(TEST_DIR "/transform/tux_tmat.test"),
                             AF_INTERP_NEAREST, false);
}

TYPED_TEST(TransformInt, PerspectiveBilinear) {
    transformTest<TypeParam>(string(TEST_DIR "/transform/tux_bilinear.test"),
                             string(TEST_DIR "/transform/tux_tmat.test"),
                             AF_INTERP_BILINEAR, false);
}

TYPED_TEST(TransformInt, PerspectiveLower) {
    transformTest<TypeParam>(string(TEST_DIR "/transform/tux_lower.test"),
                             string(TEST_DIR "/transform/tux_tmat.test"),
                             AF_INTERP_LOWER, false);
}

TYPED_TEST(TransformInt, PerspectiveNearestInvert) {
    transformTest<TypeParam>(
        string(TEST_DIR "/transform/tux_nearest.test"),
        string(TEST_DIR "/transform/tux_tmat_inverse.test"), AF_INTERP_NEAREST,
        true);
}

TYPED_TEST(TransformInt, PerspectiveBilinearInvert) {
    transformTest<TypeParam>(
        string(TEST_DIR "/transform/tux_bilinear.test"),
        string(TEST_DIR "/transform/tux_tmat_inverse.test"), AF_INTERP_BILINEAR,
        true);
}

TYPED_TEST(TransformInt, PerspectiveLowerInvert) {
    transformTest<TypeParam>(
        string(TEST_DIR "/transform/tux_lower.test"),
        string(TEST_DIR "/transform/tux_tmat_inverse.test"), AF_INTERP_LOWER,
        true);
}

template<typename T>
class TransformV2 : public Transform<T> {
   protected:
    typedef typename dtype_traits<T>::base_type BT;

    af_array gold;
    af_array in;
    af_array transform;

    dim4 gold_dims;
    dim4 in_dims;
    dim4 transform_dims;

    dim_t odim0;
    dim_t odim1;

    af_interp_type method;
    bool invert;

    TransformV2()
        : gold(0)
        , in(0)
        , transform(0)
        , odim0(0)
        , odim1(0)
        , method(AF_INTERP_NEAREST)
        , invert(false) {}

    void setInterpType(af_interp_type m) { method = m; }
    void setInvertFlag(bool i) { invert = i; }

    void SetUp() {}

    void releaseArrays() {
        if (transform != 0) { ASSERT_SUCCESS(af_release_array(transform)); }
        if (in != 0) { ASSERT_SUCCESS(af_release_array(in)); }
        if (gold != 0) { ASSERT_SUCCESS(af_release_array(gold)); }

        gold      = 0;
        in        = 0;
        transform = 0;
    }

    void TearDown() { releaseArrays(); }

    void setTestData(float *h_gold, dim4 gold_dims, float *h_in, dim4 in_dims,
                     float *h_transform, dim4 transform_dims) {
        releaseArrays();

        this->gold_dims      = gold_dims;
        this->in_dims        = in_dims;
        this->transform_dims = transform_dims;

        vector<T> h_gold_cast;
        vector<T> h_in_cast;
        vector<BT> h_transform_cast;

        for (int i = 0; i < gold_dims.elements(); ++i) {
            h_gold_cast.push_back(static_cast<T>(h_gold[i]));
        }
        for (int i = 0; i < in_dims.elements(); ++i) {
            h_in_cast.push_back(static_cast<T>(h_in[i]));
        }
        for (int i = 0; i < transform_dims.elements(); ++i) {
            h_transform_cast.push_back(static_cast<BT>(h_transform[i]));
        }

        ASSERT_SUCCESS(af_create_array(&gold, &h_gold_cast.front(),
                                       gold_dims.ndims(), gold_dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
        ASSERT_SUCCESS(af_create_array(&in, &h_in_cast.front(), in_dims.ndims(),
                                       in_dims.get(),
                                       (af_dtype)dtype_traits<T>::af_type));
        ASSERT_SUCCESS(af_create_array(
            &transform, &h_transform_cast.front(), transform_dims.ndims(),
            transform_dims.get(), (af_dtype)dtype_traits<BT>::af_type));
    }

    void setTestData(string pTestFile, string pHomographyFile) {
        if (noImageIOTests()) return;
        releaseArrays();

        genTestData<T>(&gold, &in, &transform, &odim0, &odim1, pTestFile,
                       pHomographyFile);

        ASSERT_SUCCESS(af_get_dims(&gold_dims[0], &gold_dims[1], &gold_dims[2],
                                   &gold_dims[3], gold));
        ASSERT_SUCCESS(af_get_dims(&in_dims[0], &in_dims[1], &in_dims[2],
                                   &in_dims[3], in));
        ASSERT_SUCCESS(af_get_dims(&transform_dims[0], &transform_dims[1],
                                   &transform_dims[2], &transform_dims[3],
                                   transform));
    }

    void assertSpclArraysTransform(const af_array gold, const af_array out,
                                   TestOutputArrayInfo *metadata) {
        // In the case of NULL_ARRAY, the output array starts out as null.
        // After the af_* function is called, it shouldn't be null anymore
        if (metadata->getOutputArrayType() == NULL_ARRAY) {
            if (out == 0) {
                ASSERT_TRUE(out != 0) << "Output af_array is null";
            }
            metadata->setOutput(out);
        }
        // For every other case, must check if the af_array generated by
        // genTestOutputArray was used by the af_* function as its output array
        else {
            if (metadata->getOutput() != out) {
                ASSERT_TRUE(metadata->getOutput() != out)
                    << "af_array POINTER MISMATCH:\n"
                    << "  Actual: " << out << "\n"
                    << "Expected: " << metadata->getOutput();
            }
        }

        af_array out_  = 0;
        af_array gold_ = 0;

        if (metadata->getOutputArrayType() == SUB_ARRAY) {
            // There are two full arrays. One will be injected with the gold
            // subarray, the other should have already been injected with the
            // af_* function's output. Then we compare the two full arrays
            af_array gold_full_array = metadata->getFullOutputCopy();
            af_assign_seq(&gold_full_array, gold_full_array,
                          metadata->getSubArrayNumDims(),
                          metadata->getSubArrayIdxs(), gold);

            gold_ = metadata->getFullOutputCopy();
            out_  = metadata->getFullOutput();
        } else {
            gold_ = gold;
            out_  = out;
        }

        // Get gold data
        dim_t goldEl = 0;
        af_get_elements(&goldEl, gold_);
        vector<T> goldData(goldEl);
        af_get_data_ptr((void *)&goldData.front(), gold_);

        // Get result
        dim_t outEl = 0;
        af_get_elements(&outEl, out_);
        vector<T> outData(outEl);
        af_get_data_ptr((void *)&outData.front(), out_);

        const float thr = 1.1f;

        // Maximum number of wrong pixels must be <= 0.01% of number of
        // elements, this metric is necessary due to rounding errors between
        // different backends for AF_INTERP_NEAREST and AF_INTERP_LOWER
        const size_t maxErr = goldEl * 0.0001f;
        size_t err          = 0;

        for (dim_t elIter = 0; elIter < goldEl; elIter++) {
            err += fabs((float)floor(outData[elIter]) -
                        (float)floor(goldData[elIter])) > thr;
            if (err > maxErr) {
                ASSERT_LE(err, maxErr) << "at: " << elIter << endl;
            }
        }
    }

    void testSpclOutArray(TestOutputArrayType out_array_type) {
        SUPPORTED_TYPE_CHECK(T);
        if (noImageIOTests()) return;

        af_array out = 0;
        TestOutputArrayInfo metadata(out_array_type);
        genTestOutputArray(&out, gold_dims.ndims(), gold_dims.get(),
                           (af_dtype)dtype_traits<T>::af_type, &metadata);
        ASSERT_SUCCESS(
            af_transform_v2(&out, in, transform, odim0, odim1, method, invert));

        assertSpclArraysTransform(gold, out, &metadata);
    }
};

TYPED_TEST_SUITE(TransformV2, TestTypes);

template<typename T>
class TransformV2TuxNearest : public TransformV2<T> {
   protected:
    void SetUp() {
        this->setTestData(string(TEST_DIR "/transform/tux_nearest.test"),
                          string(TEST_DIR "/transform/tux_tmat.test"));
        this->setInterpType(AF_INTERP_NEAREST);
        this->setInvertFlag(false);
    }
};

TYPED_TEST_SUITE(TransformV2TuxNearest, TestTypes);

TYPED_TEST(TransformV2TuxNearest, UseNullOutputArray) {
    this->testSpclOutArray(NULL_ARRAY);
}

TYPED_TEST(TransformV2TuxNearest, UseFullExistingOutputArray) {
    this->testSpclOutArray(FULL_ARRAY);
}

TYPED_TEST(TransformV2TuxNearest, UseExistingOutputSubArray) {
    this->testSpclOutArray(SUB_ARRAY);
}

TYPED_TEST(TransformV2TuxNearest, UseReorderedOutputArray) {
    this->testSpclOutArray(REORDERED_ARRAY);
}

class TransformNullArgs : public TransformV2TuxNearest<float> {
   protected:
    af_array out;
    TransformNullArgs() : out(0) {}
};

TEST_F(TransformNullArgs, NullOutputPtr) {
    af_array *out_ptr = 0;
    ASSERT_EQ(AF_ERR_ARG,
              af_transform(out_ptr, this->in, this->transform, this->odim0,
                           this->odim1, this->method, this->invert));
}

TEST_F(TransformNullArgs, NullInputArray) {
    ASSERT_EQ(AF_ERR_ARG,
              af_transform(&this->out, 0, this->transform, this->odim0,
                           this->odim1, this->method, this->invert));
}

TEST_F(TransformNullArgs, NullTransformArray) {
    ASSERT_EQ(AF_ERR_ARG,
              af_transform(&this->out, this->in, 0, this->odim0, this->odim1,
                           this->method, this->invert));
}

TEST_F(TransformNullArgs, V2NullOutputPtr) {
    af_array *out_ptr = 0;
    ASSERT_EQ(AF_ERR_ARG,
              af_transform_v2(out_ptr, this->in, this->transform, this->odim0,
                              this->odim1, this->method, this->invert));
}

TEST_F(TransformNullArgs, V2NullInputArray) {
    ASSERT_EQ(AF_ERR_ARG,
              af_transform_v2(&this->out, 0, this->transform, this->odim0,
                              this->odim1, this->method, this->invert));
}

TEST_F(TransformNullArgs, V2NullTransformArray) {
    ASSERT_EQ(AF_ERR_ARG,
              af_transform_v2(&this->out, this->in, 0, this->odim0, this->odim1,
                              this->method, this->invert));
}

///////////////////////////////////// CPP ////////////////////////////////
//
TEST(Transform, CPP) {
    if (noImageIOTests()) return;

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> goldDim;
    vector<string> goldFiles;

    vector<dim4> HDims;
    vector<vector<float>> HIn;
    vector<vector<float>> HTests;
    readTests<float, float, float>(TEST_DIR "/transform/tux_tmat.test", HDims,
                                   HIn, HTests);

    readImageTests(string(TEST_DIR "/transform/tux_nearest.test"), inDims,
                   inFiles, goldDim, goldFiles);

    inFiles[0].insert(0, string(TEST_DIR "/transform/"));
    inFiles[1].insert(0, string(TEST_DIR "/transform/"));

    goldFiles[0].insert(0, string(TEST_DIR "/transform/"));

    array H  = array(HDims[0][0], HDims[0][1], &(HIn[0].front()));
    array IH = array(HDims[0][0], HDims[0][1], &(HIn[0].front()));

    array scene_img = loadImage(inFiles[1].c_str(), false);

    array gold_img = loadImage(goldFiles[0].c_str(), false);

    array out_img = transform(scene_img, IH, inDims[0][0], inDims[0][1],
                              AF_INTERP_NEAREST, false);

    dim4 outDims  = out_img.dims();
    dim4 goldDims = gold_img.dims();

    vector<float> h_out_img(outDims[0] * outDims[1]);
    out_img.host(&h_out_img.front());
    vector<float> h_gold_img(goldDims[0] * goldDims[1]);
    gold_img.host(&h_gold_img.front());

    const dim_t n   = gold_img.elements();
    const float thr = 1.0f;

    // Maximum number of wrong pixels must be <= 0.01% of number of elements,
    // this metric is necessary due to rounding errors between different
    // backends for AF_INTERP_NEAREST and AF_INTERP_LOWER
    const size_t maxErr = n * 0.0001f;
    size_t err          = 0;

    for (dim_t elIter = 0; elIter < n; elIter++) {
        err += fabs((int)h_out_img[elIter] - h_gold_img[elIter]) > thr;
        if (err > maxErr) {
            ASSERT_LE(err, maxErr) << "at: " << elIter << endl;
        }
    }
}

// This tests batching of different forms
// tf0 rotates by 90 clockwise
// tf1 rotates by 90 counter clockwise
// This test simply makes sure the batching is working correctly
TEST(TransformBatching, CPP) {
    vector<dim4> vDims;
    vector<vector<float>> in;
    vector<vector<float>> gold;

    readTests<float, float, int>(
        string(TEST_DIR "/transform/transform_batching.test"), vDims, in, gold);

    array img0(vDims[0], &(in[0].front()));
    array img1(vDims[1], &(in[1].front()));
    array ip_tile(vDims[2], &(in[2].front()));
    array ip_quad(vDims[3], &(in[3].front()));
    array ip_mult(vDims[4], &(in[4].front()));
    array ip_tile3(vDims[5], &(in[5].front()));
    array ip_quad3(vDims[6], &(in[6].front()));

    array tf0(vDims[7 + 0], &(in[7 + 0].front()));
    array tf1(vDims[7 + 1], &(in[7 + 1].front()));
    array tf_tile(vDims[7 + 2], &(in[7 + 2].front()));
    array tf_quad(vDims[7 + 3], &(in[7 + 3].front()));
    array tf_mult(vDims[7 + 4], &(in[7 + 4].front()));
    array tf_mult3(vDims[7 + 5], &(in[7 + 5].front()));
    array tf_mult3x(vDims[7 + 6], &(in[7 + 6].front()));

    const int X = img0.dims(0);
    const int Y = img0.dims(1);

    ASSERT_EQ(gold.size(), 21u);
    vector<array> out(gold.size());
    out[0] = transform(img0, tf0, Y, X, AF_INTERP_NEAREST);  // 1,1 x 1,1
    out[1] = transform(img0, tf1, Y, X, AF_INTERP_NEAREST);  // 1,1 x 1,1
    out[2] = transform(img1, tf0, Y, X, AF_INTERP_NEAREST);  // 1,1 x 1,1
    out[3] = transform(img1, tf1, Y, X, AF_INTERP_NEAREST);  // 1,1 x 1,1

    out[4] = transform(img0, tf_tile, Y, X, AF_INTERP_NEAREST);  // 1,1 x N,1
    out[5] = transform(img0, tf_mult, Y, X, AF_INTERP_NEAREST);  // 1,1 x N,N
    out[6] = transform(img0, tf_quad, Y, X, AF_INTERP_NEAREST);  // 1,1 x 1,N

    out[7] = transform(ip_tile, tf0, Y, X, AF_INTERP_NEAREST);      // N,1 x 1,1
    out[8] = transform(ip_tile, tf_tile, Y, X, AF_INTERP_NEAREST);  // N,1 x N,1
    out[9] = transform(ip_tile, tf_mult, Y, X, AF_INTERP_NEAREST);  // N,N x N,N
    out[10] =
        transform(ip_tile, tf_quad, Y, X, AF_INTERP_NEAREST);  // N,1 x 1,N

    out[11] = transform(ip_quad, tf0, Y, X, AF_INTERP_NEAREST);  // 1,N x 1,1
    out[12] =
        transform(ip_quad, tf_quad, Y, X, AF_INTERP_NEAREST);  // 1,N x 1,N
    out[13] =
        transform(ip_quad, tf_mult, Y, X, AF_INTERP_NEAREST);  // 1,N x N,N
    out[14] =
        transform(ip_quad, tf_tile, Y, X, AF_INTERP_NEAREST);  // 1,N x N,1

    out[15] = transform(ip_mult, tf0, Y, X, AF_INTERP_NEAREST);  // N,N x 1,1
    out[16] =
        transform(ip_mult, tf_tile, Y, X, AF_INTERP_NEAREST);  // N,N x N,1
    out[17] =
        transform(ip_mult, tf_mult, Y, X, AF_INTERP_NEAREST);  // N,N x N,N
    out[18] =
        transform(ip_mult, tf_quad, Y, X, AF_INTERP_NEAREST);  // N,N x 1,N

    out[19] =
        transform(ip_tile3, tf_mult3, Y, X, AF_INTERP_NEAREST);  // N,1 x N,N
    out[20] =
        transform(ip_quad3, tf_mult3x, Y, X, AF_INTERP_NEAREST);  // 1,N x N,N

    array x_(dim4(35, 40, 1, 1), &(gold[1].front()));

    for (int i = 0; i < (int)gold.size(); i++) {
        // Get result
        vector<float> outData(out[i].elements());
        out[i].host((void *)&outData.front());

        for (int iter = 0; iter < (int)gold[i].size(); iter++) {
            ASSERT_EQ(gold[i][iter], outData[iter])
                << "at: " << iter << endl
                << "for " << i << "-th operation" << endl;
        }
    }
}
