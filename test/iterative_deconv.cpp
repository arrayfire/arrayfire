/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/data.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <testHelpers.hpp>

using std::string;
using std::vector;
using std::abs;
using namespace af;

template<typename T>
class IterativeDeconvolution : public ::testing::Test
{
};

// create a list of types to be tested
typedef ::testing::Types<float, uchar, short, ushort> TestTypes;

// register the type list
TYPED_TEST_CASE(IterativeDeconvolution, TestTypes);

template<typename T, bool isColor>
void iterDeconvImageTest(string pTestFile, const unsigned iters, const float rf,
                         const af::iterativeDeconvAlgo algo)
{
    typedef typename cond_type<is_same_type<T, double>::value, double, float>::type OutType;

    if (noDoubleTests<T>()) return;
    if (noImageIOTests()) return;

    using af::dim4;

    vector<dim4>       inDims;
    vector<string>    inFiles;
    vector<dim_t> outSizes;
    vector<string>   outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId)
    {
        inFiles[testId].insert(0, string(TEST_DIR "/iterative_deconv/"));
        outFiles[testId].insert(0, string(TEST_DIR "/iterative_deconv/"));

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

        ASSERT_EQ(AF_SUCCESS, af_gaussian_kernel(&kerArray, 13, 13, 2.25, 2.25));

        af_dtype itype = (af_dtype)af::dtype_traits<T>::af_type;
        af_dtype otype = (af_dtype)af::dtype_traits<OutType>::af_type;

        ASSERT_EQ(AF_SUCCESS, af_load_image(&_inArray, inFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, conv_image<T>(&inArray, _inArray));

        ASSERT_EQ(AF_SUCCESS, af_load_image(&_goldArray, outFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, conv_image<OutType>(&goldArray, _goldArray));
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&nElems, goldArray));

        unsigned ndims;
        dim_t dims[4];
        ASSERT_EQ(AF_SUCCESS, af_get_numdims(&ndims, goldArray));
        ASSERT_EQ(AF_SUCCESS, af_get_dims(dims, dims+1, dims+2, dims+3, goldArray));

        ASSERT_EQ(AF_SUCCESS, af_iterative_deconv(&_outArray, inArray, kerArray, iters, rf, algo));

        double maxima, minima, imag;
        ASSERT_EQ(AF_SUCCESS, af_min_all(&minima, &imag, _outArray));
        ASSERT_EQ(AF_SUCCESS, af_max_all(&maxima, &imag, _outArray));
        ASSERT_EQ(AF_SUCCESS, af_constant(&cstArray, 255.0, ndims, dims, otype));
        ASSERT_EQ(AF_SUCCESS, af_constant(&denArray, (maxima-minima), ndims, dims, otype));
        ASSERT_EQ(AF_SUCCESS, af_constant(&minArray, minima, ndims, dims, otype));
        ASSERT_EQ(AF_SUCCESS, af_sub(&numArray, _outArray, minArray, false));
        ASSERT_EQ(AF_SUCCESS, af_div(&divArray, numArray, denArray, false));
        ASSERT_EQ(AF_SUCCESS, af_mul(&outArray, divArray, cstArray, false));

        std::vector<OutType> outData(nElems);
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData.data(), outArray));

        std::vector<OutType> goldData(nElems);
        ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)goldData.data(), goldArray));

        ASSERT_EQ(AF_SUCCESS, af_release_array(_inArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(kerArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(cstArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(minArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(denArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(numArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(divArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(_outArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(_goldArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(goldArray));

        ASSERT_EQ(true, compareArraysRMSD(nElems, goldData.data(), outData.data(), 0.03));
    }
}

TYPED_TEST(IterativeDeconvolution, LandweberOnGrayscale)
{
    // Test file name format: <colorspace>_<iterations>_<number/1000:relaxation factor>_<algo>.test
    iterDeconvImageTest<TypeParam, false>(string(TEST_DIR "/iterative_deconv/gray_100_50_landweber.test"),
            100, 0.05, AF_ITERATIVE_DECONV_LANDWEBER);
}

TYPED_TEST(IterativeDeconvolution, RichardsonLucyOnGrayscale)
{
    // Test file name format: <colorspace>_<iterations>_<number/1000:relaxation factor>_<algo>.test
    // For RichardsonLucy algorithm, relaxation factor is not used.
    iterDeconvImageTest<TypeParam, false>(string(TEST_DIR "/iterative_deconv/gray_100_50_lucy.test"),
            100, 0.05, AF_ITERATIVE_DECONV_RICHARDSONLUCY);
}
