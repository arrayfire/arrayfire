/*******************************************************
 * Copyright (c) 2017, ArrayFire
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

template<typename T>
class AnisotropicDiffusion : public ::testing::Test
{
};

typedef ::testing::Types<float, double, int, uint, uchar, short, ushort> TestTypes;

TYPED_TEST_CASE(AnisotropicDiffusion, TestTypes);

template<typename T>
af::array normalize(const af::array &p_in)
{
    T mx = af::max<T>(p_in);
    T mn = af::min<T>(p_in);
    return (p_in-mn)/(mx-mn);
}

template<typename T, bool isColor>
void imageTest(string pTestFile, const float dt, const float K, const uint iters,
               af::fluxFunction fluxKind, bool isCurvatureDiffusion=false)
{
    typedef typename cond_type<is_same_type<T, double>::value, double, float>::type OutType;

    if (noDoubleTests<T>()) return;
    if (noImageIOTests()) return;

    using af::dim4;

    vector<dim4>       inDims;
    vector<string>    inFiles;
    vector<dim_t>    outSizes;
    vector<string>   outFiles;

    readImageTests(pTestFile, inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId=0; testId<testCount; ++testId) {

        if (isCurvatureDiffusion) {
            inFiles[testId].insert(0, string(TEST_DIR "/curvature_diffusion/"));
            outFiles[testId].insert(0, string(TEST_DIR "/curvature_diffusion/"));
        } else {
            inFiles[testId].insert(0, string(TEST_DIR "/gradient_diffusion/"));
            outFiles[testId].insert(0, string(TEST_DIR "/gradient_diffusion/"));
        }

        af_array _inArray    = 0;
        af_array inArray     = 0;
        af_array _outArray   = 0;
        af_array cstArray    = 0;
        af_array minArray    = 0;
        af_array numArray    = 0;
        af_array denArray    = 0;
        af_array divArray    = 0;
        af_array outArray    = 0;
        af_array goldArray   = 0;
        af_array _goldArray  = 0;
        dim_t nElems         = 0;

        ASSERT_EQ(AF_SUCCESS, af_load_image(&_inArray, inFiles[testId].c_str(), isColor));
        ASSERT_EQ(AF_SUCCESS, conv_image<T>(&inArray, _inArray));

        ASSERT_EQ(AF_SUCCESS, af_load_image(&_goldArray, outFiles[testId].c_str(), isColor));
        // af_load_image always returns float array, so convert to output type
        ASSERT_EQ(AF_SUCCESS, conv_image<OutType>(&goldArray, _goldArray));
        ASSERT_EQ(AF_SUCCESS, af_get_elements(&nElems, goldArray));

        if (isCurvatureDiffusion) {
            ASSERT_EQ(AF_SUCCESS, af_anisotropic_diffusion(&_outArray, inArray, dt, K, iters,
                                                           fluxKind, AF_DIFFUSION_MCDE));
        } else {
            ASSERT_EQ(AF_SUCCESS, af_anisotropic_diffusion(&_outArray, inArray, dt, K, iters,
                                                           fluxKind, AF_DIFFUSION_GRAD));
        }

        double maxima, minima, imag;
        ASSERT_EQ(AF_SUCCESS, af_min_all(&minima, &imag, _outArray));
        ASSERT_EQ(AF_SUCCESS, af_max_all(&maxima, &imag, _outArray));

        unsigned ndims;
        dim_t dims[4];
        ASSERT_EQ(AF_SUCCESS, af_get_numdims(&ndims, _outArray));
        ASSERT_EQ(AF_SUCCESS, af_get_dims(dims, dims+1, dims+2, dims+3, _outArray));

        af_dtype otype = (af_dtype)af::dtype_traits<OutType>::af_type;
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

        ASSERT_EQ(true, compareArraysRMSD(nElems, goldData.data(), outData.data(), 0.025f));

        ASSERT_EQ(AF_SUCCESS, af_release_array(_inArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(_outArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(cstArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(minArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(denArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(numArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(divArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(_goldArray));
        ASSERT_EQ(AF_SUCCESS, af_release_array(goldArray));
    }
}

TYPED_TEST(AnisotropicDiffusion, GradientGrayscale)
{
    // Numeric values separated by underscore are arguments to fn being tested.
    // Divide first value by 1000 to get time step `dt`
    // Divide second value by 100 to get time step `K`
    // Divide third value stays as it is since it is iteration count
    // Fourth value is a 4-character string indicating the flux kind
    imageTest<TypeParam, false>(string(TEST_DIR "/gradient_diffusion/gray_00125_100_64_exp.test"),
                                0.125f, 1.0, 64, AF_FLUX_EXPONENTIAL);
}

TYPED_TEST(AnisotropicDiffusion, GradientColorImage)
{
    imageTest<TypeParam, true>(string(TEST_DIR "/gradient_diffusion/color_00125_100_64_exp.test"),
                               0.125f, 1.0, 64, AF_FLUX_EXPONENTIAL);
}

TEST(AnisotropicDiffusion, GradientInvalidInputArray)
{
    try {
        af::array out = af::anisotropicDiffusion(af::randu(100), 0.125f, 0.2f, 10, AF_FLUX_QUADRATIC);
    } catch (af::exception &exp) {
        ASSERT_EQ(AF_ERR_SIZE, exp.err());
    }
}

TYPED_TEST(AnisotropicDiffusion, CurvatureGrayscale)
{
    // Numeric values separated by underscore are arguments to fn being tested.
    // Divide first value by 1000 to get time step `dt`
    // Divide second value by 100 to get time step `K`
    // Divide third value stays as it is since it is iteration count
    // Fourth value is a 4-character string indicating the flux kind
    imageTest<TypeParam, false>(string(TEST_DIR "/curvature_diffusion/gray_00125_100_64_mcde.test"),
                                0.125f, 1.0, 64, AF_FLUX_EXPONENTIAL, true);
}

TYPED_TEST(AnisotropicDiffusion, CurvatureColorImage)
{
    imageTest<TypeParam, true>(string(TEST_DIR "/curvature_diffusion/color_00125_100_64_mcde.test"),
                               0.125f, 1.0, 64, AF_FLUX_EXPONENTIAL, true);
}

TEST(AnisotropicDiffusion, CurvatureInvalidInputArray)
{
    try {
        af::array out = af::anisotropicDiffusion(af::randu(100), 0.125f, 0.2f, 10);
    } catch (af::exception &exp) {
        ASSERT_EQ(AF_ERR_SIZE, exp.err());
    }
}
