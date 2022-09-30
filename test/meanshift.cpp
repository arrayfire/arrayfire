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
#include <cmath>
#include <string>
#include <vector>

using af::dim4;
using af::dtype_traits;
using std::abs;
using std::string;
using std::vector;

template<typename T>
class Meanshift : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

typedef ::testing::Types<float, double, int, uint, char, uchar, short, ushort,
                         intl, uintl>
    TestTypes;

TYPED_TEST_SUITE(Meanshift, TestTypes);

TYPED_TEST(Meanshift, InvalidArgs) {
    SUPPORTED_TYPE_CHECK(TypeParam);

    vector<TypeParam> in(100, 1);

    af_array inArray  = 0;
    af_array outArray = 0;

    dim4 dims = dim4(100, 1, 1, 1);
    ASSERT_SUCCESS(af_create_array(&inArray, &in.front(), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)dtype_traits<TypeParam>::af_type));
    ASSERT_EQ(AF_ERR_SIZE,
              af_mean_shift(&outArray, inArray, 0.12f, 0.34f, 5, true));
    ASSERT_SUCCESS(af_release_array(inArray));
}

template<typename T, bool isColor>
void meanshiftTest(string pTestFile, const float ss) {
    SUPPORTED_TYPE_CHECK(T);
    if (noImageIOTests()) return;

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        af_array inArray       = 0;
        af_array inArray_f32   = 0;
        af_array outArray      = 0;
        af_array goldArray     = 0;
        af_array goldArray_f32 = 0;
        dim_t nElems           = 0;

        inFiles[testId].insert(0, string(TEST_DIR "/meanshift/"));
        outFiles[testId].insert(0, string(TEST_DIR "/meanshift/"));

        ASSERT_SUCCESS(
            af_load_image(&inArray_f32, inFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(conv_image<T>(&inArray, inArray_f32));

        ASSERT_SUCCESS(
            af_load_image(&goldArray_f32, outFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(conv_image<T>(
            &goldArray,
            goldArray_f32));  // af_load_image always returns float array
        ASSERT_SUCCESS(af_get_elements(&nElems, goldArray));

        ASSERT_SUCCESS(af_mean_shift(&outArray, inArray, ss, 30.f, 5, isColor));

        ASSERT_IMAGES_NEAR(goldArray, outArray, 0.02f);

        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(inArray_f32));
        ASSERT_SUCCESS(af_release_array(outArray));
        ASSERT_SUCCESS(af_release_array(goldArray));
        ASSERT_SUCCESS(af_release_array(goldArray_f32));
    }
}

// create a list of types to be tested
// FIXME: since af_load_image returns only f32 type arrays
//       only float, double data types test are enabled & passing
//       Note: compareArraysRMSD is handling upcasting while working
//       with two different type of types
//
#define IMAGE_TESTS(T)                                                   \
    TEST(Meanshift, Grayscale_##T) {                                     \
        meanshiftTest<T, false>(string(TEST_DIR "/meanshift/gray.test"), \
                                6.67f);                                  \
    }                                                                    \
    TEST(Meanshift, Color_##T) {                                         \
        meanshiftTest<T, true>(string(TEST_DIR "/meanshift/color.test"), \
                               3.5f);                                    \
    }

IMAGE_TESTS(float)
IMAGE_TESTS(double)

//////////////////////////////////////// CPP ///////////////////////////////
//

using af::array;
using af::constant;
using af::iota;
using af::loadImage;
using af::max;
using af::meanShift;
using af::seq;
using af::span;

TEST(Meanshift, Color_CPP) {
    if (noImageIOTests()) return;

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(string(TEST_DIR "/meanshift/color.test"), inDims, inFiles,
                   outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        inFiles[testId].insert(0, string(TEST_DIR "/meanshift/"));
        outFiles[testId].insert(0, string(TEST_DIR "/meanshift/"));

        array img    = loadImage(inFiles[testId].c_str(), true);
        array gold   = loadImage(outFiles[testId].c_str(), true);
        dim_t nElems = gold.elements();
        array output = meanShift(img, 3.5f, 30.f, 5, true);

        ASSERT_IMAGES_NEAR(gold, output, 0.02f);
    }
}

TEST(Meanshift, GFOR) {
    dim4 dims = dim4(10, 10, 3);
    array A   = iota(dims);
    array B   = constant(0, dims);

    gfor(seq ii, 3) {
        B(span, span, ii) = meanShift(A(span, span, ii), 3, 5, 3);
    }

    for (int ii = 0; ii < 3; ii++) {
        array c_ii = meanShift(A(span, span, ii), 3, 5, 3);
        array b_ii = B(span, span, ii);

        ASSERT_LT(max<double>(abs(c_ii - b_ii)), 1E-5);
    }
}
