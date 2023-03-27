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
#include <af/traits.hpp>

#include <sstream>
#include <string>
#include <vector>

using af::dim4;
using std::abs;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;

template<typename T>
class ConfidenceConnectedImageTest : public testing::Test {
   public:
    virtual void SetUp() {}
};

typedef ::testing::Types<float, uint, ushort, uchar> TestTypes;

TYPED_TEST_SUITE(ConfidenceConnectedImageTest, TestTypes);

struct CCCTestParams {
    const char *prefix;
    unsigned int radius;
    unsigned int multiplier;
    unsigned int iterations;
    double replace;
};

void apiWrapper(af_array *out, const af_array in, const af_array seedx,
                const af_array seedy, const CCCTestParams params) {
    ASSERT_SUCCESS(af_confidence_cc(out, in, seedx, seedy, params.radius,
                                    params.multiplier, params.iterations,
                                    params.replace));

    int device = 0;
    ASSERT_SUCCESS(af_get_device(&device));
    ASSERT_SUCCESS(af_sync(device));
}

template<typename T>
void testImage(const std::string pTestFile, const size_t numSeeds,
               const unsigned *seedx, const unsigned *seedy,
               const int multiplier, const unsigned neighborhood_radius,
               const int iter) {
    SUPPORTED_TYPE_CHECK(T);
    IMAGEIO_ENABLED_CHECK();

    vector<af::dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(std::string(TEST_DIR) + "/confidence_cc/" + pTestFile,
                   inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    af_array seedxArr = 0, seedyArr = 0;
    dim4 seedDims(numSeeds);
    ASSERT_SUCCESS(af_create_array(&seedxArr, seedx, seedDims.ndims(),
                                   seedDims.get(), u32));
    ASSERT_SUCCESS(af_create_array(&seedyArr, seedy, seedDims.ndims(),
                                   seedDims.get(), u32));

    for (size_t testId = 0; testId < testCount; ++testId) {
        af_array _inArray   = 0;
        af_array inArray    = 0;
        af_array outArray   = 0;
        af_array _goldArray = 0;
        af_array goldArray  = 0;

        inFiles[testId].insert(0, string(TEST_DIR "/confidence_cc/"));
        outFiles[testId].insert(0, string(TEST_DIR "/confidence_cc/"));

        ASSERT_SUCCESS(
            af_load_image(&_inArray, inFiles[testId].c_str(), false));
        ASSERT_SUCCESS(
            af_load_image(&_goldArray, outFiles[testId].c_str(), false));

        // af_load_image always returns float array, so convert to output type
        ASSERT_SUCCESS(conv_image<T>(&inArray, _inArray));
        ASSERT_SUCCESS(conv_image<T>(&goldArray, _goldArray));

        CCCTestParams params;
        params.prefix     = "Image";
        params.radius     = neighborhood_radius;
        params.multiplier = multiplier;
        params.iterations = iter;
        params.replace    = 255.0;

        apiWrapper(&outArray, inArray, seedxArr, seedyArr, params);

        ASSERT_ARRAYS_EQ(outArray, goldArray);

        ASSERT_SUCCESS(af_release_array(_inArray));
        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(outArray));
        ASSERT_SUCCESS(af_release_array(_goldArray));
        ASSERT_SUCCESS(af_release_array(goldArray));
    }
    ASSERT_SUCCESS(af_release_array(seedxArr));
    ASSERT_SUCCESS(af_release_array(seedyArr));
}

template<typename T>
void testData(CCCTestParams params) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;

    string file = string(TEST_DIR) + "/confidence_cc/" + string(params.prefix) +
                  "_" + to_string(params.radius) + "_" +
                  to_string(params.multiplier) + ".test";
    readTests<T, T, int>(file, numDims, in, tests);

    dim4 dims         = numDims[0];
    af_array inArray  = 0;
    af_array seedxArr = 0, seedyArr = 0;

    vector<uint> seedCoords(in[1].begin(), in[1].end());
    const unsigned *seedxy = seedCoords.data();

    dim4 seedDims(1);
    ASSERT_SUCCESS(af_create_array(&seedxArr, seedxy + 0, seedDims.ndims(),
                                   seedDims.get(), u32));
    ASSERT_SUCCESS(af_create_array(&seedyArr, seedxy + 1, seedDims.ndims(),
                                   seedDims.get(), u32));
    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                                   dims.get(),
                                   (af_dtype)af::dtype_traits<T>::af_type));

    af_array outArray = 0;
    apiWrapper(&outArray, inArray, seedxArr, seedyArr, params);

    ASSERT_VEC_ARRAY_EQ(tests[0], dims, outArray);

    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
    ASSERT_SUCCESS(af_release_array(seedxArr));
    ASSERT_SUCCESS(af_release_array(seedyArr));
}

class ConfidenceConnectedDataTest
    : public testing::TestWithParam<CCCTestParams> {};

TYPED_TEST(ConfidenceConnectedImageTest, DonutBackgroundExtraction) {
    const unsigned seedx = 10;
    const unsigned seedy = 10;
    testImage<TypeParam>(std::string("donut_background.test"), 1, &seedx,
                         &seedy, 3, 3, 25);
}

TYPED_TEST(ConfidenceConnectedImageTest, DonutRingExtraction) {
    const unsigned seedx = 132;
    const unsigned seedy = 132;
    testImage<TypeParam>(std::string("donut_ring.test"), 1, &seedx, &seedy, 3,
                         3, 25);
}

TYPED_TEST(ConfidenceConnectedImageTest, DonutKernelExtraction) {
    const unsigned seedx = 150;
    const unsigned seedy = 150;
    testImage<TypeParam>(std::string("donut_core.test"), 1, &seedx, &seedy, 3,
                         3, 25);
}

TEST_P(ConfidenceConnectedDataTest, SegmentARegion) {
    testData<unsigned char>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    SingleSeed, ConfidenceConnectedDataTest,
    testing::Values(CCCTestParams{"core", 0u, 1u, 5u, 255.0},
                    CCCTestParams{"background", 0u, 1u, 5u, 255.0},
                    CCCTestParams{"ring", 0u, 1u, 5u, 255.0}),
    [](const ::testing::TestParamInfo<ConfidenceConnectedDataTest::ParamType>
           info) {
        stringstream ss;
        ss << "_prefix_" << info.param.prefix << "_radius_" << info.param.radius
           << "_multiplier_" << info.param.multiplier << "_iterations_"
           << info.param.iterations << "_replace_" << info.param.replace;
        return ss.str();
    });
