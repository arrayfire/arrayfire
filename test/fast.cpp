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
#include <type_traits>
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
class FloatFAST : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

template<typename T>
class FixedFAST : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

typedef ::testing::Types<float, double> FloatTestTypes;
typedef ::testing::Types<int, unsigned> FixedTestTypes;

TYPED_TEST_CASE(FloatFAST, FloatTestTypes);
TYPED_TEST_CASE(FixedFAST, FixedTestTypes);

// TODO: perform conversion on device for CUDA and OpenCL
template<typename T>
af_err conv_image(af_array *out, af_array in)
{
    af_array outArray;

    dim_type d0, d1, d2, d3;
    af_get_dims(&d0, &d1, &d2, &d3, in);
    dim4 idims(d0, d1, d2, d3);

    dim_type nElems = 0;
    af_get_elements(&nElems, in);

    float *in_data = new float[nElems];
    af_get_data_ptr(in_data, in);

    T *out_data = new T[nElems];

    for (int i = 0; i < nElems; i++)
        out_data[i] = (T)in_data[i];

    af_create_array(&outArray, out_data, idims.ndims(), idims.get(), (af_dtype) af::dtype_traits<T>::af_type);

    std::swap(*out, outArray);

    delete in_data;

    return AF_SUCCESS;
}

template<typename T>
void fastTest(string pTestFile, bool nonmax)
{
    vector<dim4>        inDims;
    vector<string>     inFiles;
    vector<vector<float>> gold;

    readImageTests(pTestFile, inDims, inFiles, gold);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {
        dim_type nElems       = 0;
        af_array inArray_f32  = 0;
        af_array inArray      = 0;
        af_features outFeat;

        inFiles[testId].insert(0,string(TEST_DIR"/fast/"));

        ASSERT_EQ(AF_SUCCESS, af_load_image(&inArray_f32, inFiles[testId].c_str(), false));
        ASSERT_EQ(AF_SUCCESS, conv_image<T>(&inArray, inArray_f32));

        ASSERT_EQ(AF_SUCCESS, af_fast(&outFeat, inArray, 20.0f, 9, nonmax, 0.05f));
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&nElems, outFeat.x));

        float * outX           = new float[gold[0].size()];
        float * outY           = new float[gold[1].size()];
        float * outScore       = new float[gold[2].size()];
        float * outOrientation = new float[gold[3].size()];
        float * outSize        = new float[gold[4].size()];
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outX, outFeat.x));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outY, outFeat.y));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outScore, outFeat.score));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outOrientation, outFeat.orientation));
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outSize, outFeat.size));

        vector<feat_t> out_feat;
        array_to_feat(out_feat, outX, outY, outScore, outOrientation, outSize, outFeat.n);

        vector<feat_t> gold_feat;
        array_to_feat(gold_feat, &gold[0].front(), &gold[1].front(), &gold[2].front(), &gold[3].front(), &gold[4].front(), gold[0].size());

        std::sort(out_feat.begin(), out_feat.end(), feat_cmp);
        std::sort(gold_feat.begin(), gold_feat.end(), feat_cmp);

        for (int elIter = 0; elIter < nElems; elIter++) {
            ASSERT_EQ(out_feat[elIter].f[0], gold_feat[elIter].f[0]) << "at: " << elIter << std::endl;
            ASSERT_EQ(out_feat[elIter].f[1], gold_feat[elIter].f[1]) << "at: " << elIter << std::endl;
            ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]), 1e-3) << "at: " << elIter << std::endl;
            ASSERT_EQ(out_feat[elIter].f[3], gold_feat[elIter].f[3]) << "at: " << elIter << std::endl;
            ASSERT_EQ(out_feat[elIter].f[4], gold_feat[elIter].f[4]) << "at: " << elIter << std::endl;
        }

        ASSERT_EQ(AF_SUCCESS, af_destroy_array(inArray));

        delete outX;
        delete outY;
        delete outScore;
        delete outOrientation;
        delete outSize;
    }
}

#define FLOAT_FAST_INIT(desc, image, nonmax) \
    TYPED_TEST(FloatFAST, desc) \
    {   \
        fastTest<TypeParam>(string(TEST_DIR"/fast/"#image"_float.test"), nonmax); \
    }

#define FIXED_FAST_INIT(desc, image, nonmax) \
    TYPED_TEST(FixedFAST, desc) \
    {   \
        fastTest<TypeParam>(string(TEST_DIR"/fast/"#image"_fixed.test"), nonmax); \
    }

    FLOAT_FAST_INIT(square, square, false);
    FLOAT_FAST_INIT(square_nonmax, square_nonmax, true);
    FIXED_FAST_INIT(square, square, false);
    FIXED_FAST_INIT(square_nonmax, square_nonmax, true);

///////////////////////////////////// CPP ////////////////////////////////
//
TEST(FloatFAST, CPP)
{
    vector<dim4>        inDims;
    vector<string>     inFiles;
    vector<vector<float>> gold;

    readImageTests(string(TEST_DIR"/fast/square_nonmax_float.test"), inDims, inFiles, gold);
    inFiles[0].insert(0,string(TEST_DIR"/fast/"));

    af::array in = af::loadimage(inFiles[0].c_str(), false);

    af::features out = fast(in, 20.0f, 9, true, 0.05f);

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
        ASSERT_LE(fabs(out_feat[elIter].f[2] - gold_feat[elIter].f[2]), 1e-3) << "at: " << elIter << std::endl;
        ASSERT_EQ(out_feat[elIter].f[3], gold_feat[elIter].f[3]) << "at: " << elIter << std::endl;
        ASSERT_EQ(out_feat[elIter].f[4], gold_feat[elIter].f[4]) << "at: " << elIter << std::endl;
    }
}
