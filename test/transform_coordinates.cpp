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
using std::endl;
using std::string;
using std::vector;

template<typename T>
class TransformCoordinates : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

typedef ::testing::Types<float, double> TestTypes;

TYPED_TEST_SUITE(TransformCoordinates, TestTypes);

template<typename T>
void transformCoordinatesTest(string pTestFile) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> inDims;
    vector<vector<T>> in;
    vector<vector<float>> gold;

    readTests<T, float, float>(pTestFile, inDims, in, gold);

    af_array tfArray  = 0;
    af_array outArray = 0;
    ASSERT_SUCCESS(af_create_array(&tfArray, &(in[0].front()),
                                   inDims[0].ndims(), inDims[0].get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    int nTests = in.size();

    for (int test = 1; test < nTests; test++) {
        dim_t d0 = (dim_t)in[test][0];
        dim_t d1 = (dim_t)in[test][1];

        ASSERT_SUCCESS(af_transform_coordinates(&outArray, tfArray, d0, d1));

        // Get result
        dim_t outEl = 0;
        ASSERT_SUCCESS(af_get_elements(&outEl, outArray));
        vector<T> outData(outEl);
        ASSERT_SUCCESS(af_get_data_ptr((void*)&outData.front(), outArray));

        ASSERT_SUCCESS(af_release_array(outArray));
        const float thr = 1.f;

        for (dim_t elIter = 0; elIter < outEl; elIter++) {
            ASSERT_LE(fabs(outData[elIter] - gold[test - 1][elIter]), thr)
                << "at: " << elIter << endl;
        }
    }

    if (tfArray != 0) af_release_array(tfArray);
}

TYPED_TEST(TransformCoordinates, RotateMatrix) {
    transformCoordinatesTest<TypeParam>(
        string(TEST_DIR "/transformCoordinates/rotate_matrix.test"));
}

TYPED_TEST(TransformCoordinates, 3DMatrix) {
    transformCoordinatesTest<TypeParam>(
        string(TEST_DIR "/transformCoordinates/3d_matrix.test"));
}

///////////////////////////////////// CPP ////////////////////////////////
//
TEST(TransformCoordinates, CPP) {
    vector<dim4> inDims;
    vector<vector<float>> in;
    vector<vector<float>> gold;

    readTests<float, float, float>(
        TEST_DIR "/transformCoordinates/3d_matrix.test", inDims, in, gold);

    array tf = array(inDims[0][0], inDims[0][1], &(in[0].front()));

    float d0 = in[1][0];
    float d1 = in[1][1];

    array out    = transformCoordinates(tf, d0, d1);
    dim4 outDims = out.dims();

    vector<float> h_out(outDims[0] * outDims[1]);
    out.host(&h_out.front());

    const size_t n  = gold[0].size();
    const float thr = 1.f;

    for (size_t elIter = 0; elIter < n; elIter++) {
        ASSERT_LE(fabs(h_out[elIter] - gold[0][elIter]), thr)
            << "at: " << elIter << endl;
    }
}
