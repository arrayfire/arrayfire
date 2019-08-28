/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define GTEST_LINKED_AS_SHARED_LIBRARY 1
#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/traits.hpp>

#include <string>
#include <sstream>
#include <vector>

using af::dim4;
using std::abs;
using std::string;
using std::stringstream;
using std::vector;

template<typename T>
class ConfidenceConnectedImageTest : public testing::Test {
   public:
    virtual void SetUp() {}
};

typedef ::testing::Types<double, float, uint, ushort, uchar> TestTypes;

TYPED_TEST_CASE(ConfidenceConnectedImageTest, TestTypes);

struct SingleSeedTestParams {
    const char* prefix;
    unsigned int seedx;
    unsigned int seedy;
    unsigned int radius;
    unsigned int multiplier;
    unsigned int iterations;
    double replace;
};

void singleSeedTest(af_array* out, const af_array in,
        const SingleSeedTestParams params) {
    const af::dim4 seedDims(1);
    const unsigned int seedxArr[] = {params.seedx};
    const unsigned int seedyArr[] = {params.seedy};

    af_array seedxArray = 0;
    af_array seedyArray = 0;

    ASSERT_SUCCESS(
            af_create_array(&seedxArray, seedxArr, seedDims.ndims(),
                seedDims.get(), u32));
    ASSERT_SUCCESS(
            af_create_array(&seedyArray, seedyArr, seedDims.ndims(),
                seedDims.get(), u32));
    ASSERT_SUCCESS(
            af_confidence_cc(out, in, seedxArray, seedyArray,
                params.radius, params.multiplier,
                params.iterations, params.replace));

    int device = 0;
    ASSERT_SUCCESS(af_get_device(&device));
    ASSERT_SUCCESS(af_sync(device));
    ASSERT_SUCCESS(af_release_array(seedxArray));
    ASSERT_SUCCESS(af_release_array(seedyArray));
}

template<typename T>
void testImage(const std::string pTestFile,
        const unsigned seedx, const unsigned seedy, const int multiplier,
        const unsigned neighborhood_radius, const int iter) {
    SUPPORTED_TYPE_CHECK(T);
    if (noImageIOTests()) return;

    vector<af::dim4> inDims;
    vector<string> inFiles;
    vector<dim_t> outSizes;
    vector<string> outFiles;

    readImageTests(std::string(TEST_DIR)+"/confidence_cc/"+pTestFile,
            inDims, inFiles, outSizes, outFiles);

    size_t testCount = inDims.size();

    for (size_t testId = 0; testId < testCount; ++testId) {
        af_array _inArray   = 0;
        af_array inArray    = 0;
        af_array outArray   = 0;
        af_array _goldArray = 0;
        af_array goldArray  = 0;
        dim_t nElems        = 0;

        inFiles[testId].insert(0, string(TEST_DIR "/confidence_cc/"));
        outFiles[testId].insert(0, string(TEST_DIR "/confidence_cc/"));

        ASSERT_SUCCESS(
                af_load_image(&_inArray, inFiles[testId].c_str(), false));
        ASSERT_SUCCESS(
                af_load_image(&_goldArray, outFiles[testId].c_str(), false));

        // af_load_image always returns float array, so convert to output type
        ASSERT_SUCCESS(conv_image<T>(&inArray, _inArray));
        ASSERT_SUCCESS(conv_image<T>(&goldArray, _goldArray));

        SingleSeedTestParams args;
        args.prefix = "Image";
        args.seedx  = seedx;
        args.seedy  = seedy;
        args.radius = neighborhood_radius;
        args.multiplier = multiplier;
        args.iterations = iter;
        args.replace = 255.0;

        singleSeedTest(&outArray, inArray, args);

        ASSERT_ARRAYS_EQ(outArray, goldArray);

        ASSERT_SUCCESS(af_release_array(_inArray));
        ASSERT_SUCCESS(af_release_array(inArray));
        ASSERT_SUCCESS(af_release_array(outArray));
        ASSERT_SUCCESS(af_release_array(_goldArray));
        ASSERT_SUCCESS(af_release_array(goldArray));
    }
}

#define CONF_CC_INIT(desc, file, seedx, seedy, iter)          \
    TYPED_TEST(ConfidenceConnectedImageTest, desc) {          \
        testImage<TypeParam>(                                 \
            std::string(#file "_" #seedx "_" #seedy ".test"), \
            seedx, seedy, 3, 3, iter);                        \
    }

CONF_CC_INIT(background, donut, 10, 10, 25)
CONF_CC_INIT(ring, donut, 132, 132, 25)
CONF_CC_INIT(core, donut, 150, 150, 25)

template<typename T>
void testData(SingleSeedTestParams params) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > tests;

    auto file = std::string(TEST_DIR) + "/confidence_cc/" +
                std::string(params.prefix) + "_" +
                std::to_string(params.seedx) + "_" +
                std::to_string(params.seedy) + "_" +
                std::to_string(params.radius) + "_" +
                std::to_string(params.multiplier) + ".test";
    readTests<T, T, int>(file, numDims, in, tests);

    dim4 dims          = numDims[0];
    af_array inArray   = 0;

    ASSERT_SUCCESS(af_create_array(&inArray, &(in[0].front()), dims.ndims(),
                dims.get(), (af_dtype)af::dtype_traits<T>::af_type));

    af_array outArray = 0;
    singleSeedTest(&outArray, inArray, params);

    ASSERT_VEC_ARRAY_EQ(tests[0], dims, outArray);

    ASSERT_SUCCESS(af_release_array(inArray));
    ASSERT_SUCCESS(af_release_array(outArray));
}

class ConfidenceConnectedDataTest
    : public testing::TestWithParam<SingleSeedTestParams> {
};

TEST_P(ConfidenceConnectedDataTest, SegmentARegion) {
    testData<unsigned char>(GetParam());
}

INSTANTIATE_TEST_CASE_P(SingleSeed, ConfidenceConnectedDataTest,
        testing::Values(SingleSeedTestParams{"core", 4u, 4u, 0u, 1u, 5u, 255.0},
            SingleSeedTestParams{"background", 1u, 1u, 0u, 1u, 5u, 255.0},
            SingleSeedTestParams{"ring", 3u, 3u, 0u, 1u, 5u, 255.0}),
        [](const ::testing::TestParamInfo<ConfidenceConnectedDataTest::ParamType> info) {
            stringstream ss;
            ss << "seedx_" << info.param.seedx
               << "_seedy_" << info.param.seedy
               << "_radius_" << info.param.radius
               << "_multiplier_" << info.param.multiplier
               << "_replace_" << info.param.replace;
            return ss.str();
        });
