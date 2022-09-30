/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/data.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>

using std::abs;
using std::string;
using std::vector;
using namespace af;

template<typename T>
class InverseDeconvolution : public ::testing::Test {};

// create a list of types to be tested
typedef ::testing::Types<float, uchar, short, ushort> TestTypes;

// register the type list
TYPED_TEST_SUITE(InverseDeconvolution, TestTypes);

template<typename T, bool isColor>
void invDeconvImageTest(string pTestFile, const float gamma,
                        const af_inverse_deconv_algo algo) {
    typedef
        typename cond_type<is_same_type<T, double>::value, double, float>::type
            OutType;

    SUPPORTED_TYPE_CHECK(T);
    if (noImageIOTests()) return;

    using af::dim4;

    vector<dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        inFiles[testId].insert(0, string(TEST_DIR "/inverse_deconv/"));
        outFiles[testId].insert(0, string(TEST_DIR "/inverse_deconv/"));

        af_array _inArray   = 0;
        af_array inArray    = 0;
        af_array kerArray   = 0;
        af_array _outArray  = 0;
        af_array cstArray   = 0;
        af_array minArray   = 0;
        af_array numArray   = 0;
        af_array denArray   = 0;
        af_array divArray   = 0;
        af_array outArray   = 0;
        af_array goldArray  = 0;
        af_array _goldArray = 0;
        dim_t nElems        = 0;

        ASSERT_SUCCESS(af_gaussian_kernel(&kerArray, 13, 13, 2.25, 2.25));

        af_dtype otype = (af_dtype)af::dtype_traits<OutType>::af_type;

        ASSERT_SUCCESS(
            af_load_image(&_inArray, inFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(conv_image<T>(&inArray, _inArray));

        ASSERT_SUCCESS(
            af_load_image(&_goldArray, outFiles[testId].c_str(), isColor));
        ASSERT_SUCCESS(conv_image<OutType>(&goldArray, _goldArray));
        ASSERT_SUCCESS(af_get_elements(&nElems, goldArray));

        unsigned ndims;
        dim_t dims[4];
        ASSERT_SUCCESS(af_get_numdims(&ndims, goldArray));
        ASSERT_SUCCESS(
            af_get_dims(dims, dims + 1, dims + 2, dims + 3, goldArray));

        ASSERT_SUCCESS(
            af_inverse_deconv(&_outArray, inArray, kerArray, gamma, algo));

        double maxima, minima, imag;
        ASSERT_SUCCESS(af_min_all(&minima, &imag, _outArray));
        ASSERT_SUCCESS(af_max_all(&maxima, &imag, _outArray));
        ASSERT_SUCCESS(af_constant(&cstArray, 255.0, ndims, dims, otype));
        ASSERT_SUCCESS(
            af_constant(&denArray, (maxima - minima), ndims, dims, otype));
        ASSERT_SUCCESS(af_constant(&minArray, minima, ndims, dims, otype));
        ASSERT_SUCCESS(af_sub(&numArray, _outArray, minArray, false));
        ASSERT_SUCCESS(af_div(&divArray, numArray, denArray, false));
        ASSERT_SUCCESS(af_mul(&outArray, divArray, cstArray, false));

        ASSERT_IMAGES_NEAR(goldArray, outArray, 0.03);

        ASSERT_SUCCESS(af_release_array(_inArray));
        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(kerArray));
        ASSERT_SUCCESS(af_release_array(cstArray));
        ASSERT_SUCCESS(af_release_array(minArray));
        ASSERT_SUCCESS(af_release_array(denArray));
        ASSERT_SUCCESS(af_release_array(numArray));
        ASSERT_SUCCESS(af_release_array(divArray));
        ASSERT_SUCCESS(af_release_array(_outArray));
        ASSERT_SUCCESS(af_release_array(outArray));
        ASSERT_SUCCESS(af_release_array(_goldArray));
        ASSERT_SUCCESS(af_release_array(goldArray));
    }
}

TYPED_TEST(InverseDeconvolution, TikhonovOnGrayscale) {
    // Test file name format: <colorspace>_<gamma with dots replaced by
    // "_">_<inverse deconv algo>.test
    invDeconvImageTest<TypeParam, false>(
        string(TEST_DIR "/inverse_deconv/gray_00_1_tikhonov.test"), 00.1f,
        AF_INVERSE_DECONV_TIKHONOV);
}

TYPED_TEST(InverseDeconvolution, DISABLED_WienerOnGrayscale) {
    // Test file name format: <colorspace>_<gamma with dots replaced by
    // "_">_<inverse deconv algo>.test
    invDeconvImageTest<TypeParam, false>(
        string(TEST_DIR "/inverse_deconv/gray_1_wiener.test"), 1.0,
        AF_INVERSE_DECONV_DEFAULT);
    // TODO(pradeep) change to wiener enum value
}
