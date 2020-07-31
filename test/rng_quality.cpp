

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

using af::allTrue;
using af::array;
using af::constant;
using af::dtype;
using af::dtype_traits;
using af::randomEngine;
using af::randomEngineType;
using af::sum;

template<typename T>
class RandomEngine : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

template<typename T>
class RandomEngineSeed : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double> TestTypesEngine;
// register the type list
TYPED_TEST_CASE(RandomEngine, TestTypesEngine);

typedef ::testing::Types<unsigned> TestTypesEngineSeed;
// register the type list
TYPED_TEST_CASE(RandomEngineSeed, TestTypesEngineSeed);

template<typename T>
void testRandomEnginePeriod(randomEngineType type) {
    SUPPORTED_TYPE_CHECK(T);
    dtype ty = (dtype)dtype_traits<T>::af_type;

    int elem  = 1024 * 1024;
    int steps = 4 * 1024;
    randomEngine r(type, 0);

    array first = randu(elem, ty, r);

    for (int i = 0; i < steps; ++i) {
        array step     = randu(elem, ty, r);
        bool different = !allTrue<bool>(first == step);
        ASSERT_TRUE(different);
    }
}

TYPED_TEST(RandomEngine, philoxRandomEnginePeriod) {
    testRandomEnginePeriod<TypeParam>(AF_RANDOM_ENGINE_PHILOX_4X32_10);
}

TYPED_TEST(RandomEngine, threefryRandomEnginePeriod) {
    testRandomEnginePeriod<TypeParam>(AF_RANDOM_ENGINE_THREEFRY_2X32_16);
}

TYPED_TEST(RandomEngine, mersenneRandomEnginePeriod) {
    testRandomEnginePeriod<TypeParam>(AF_RANDOM_ENGINE_MERSENNE_GP11213);
}

template<typename T>
double chi2_statistic(array input, array expected) {
    expected *=
        convert<float>(sum<T>(input)) / convert<float>(sum<T>(expected));
    array diff = input - expected;
    return convert<float>(sum<T>((diff * diff) / expected));
}

template<>
double chi2_statistic<half_float::half>(array input, array expected) {
    expected *= convert<float>(sum<float>(input)) /
                convert<float>(sum<float>(expected));
    array diff = input - expected;
    return convert<float>(sum<float>((diff * diff) / expected));
}

template<typename T>
void testRandomEngineUniformChi2(randomEngineType type) {
    SUPPORTED_TYPE_CHECK(T);
    dtype ty = (dtype)dtype_traits<T>::af_type;

    int elem  = 256 * 1024 * 1024;
    int steps = 32;
    int bins  = 100;

    array total_hist = constant(0.0, bins, ty);
    array expected   = constant(1.0 / bins, bins, ty);

    randomEngine r(type, 0);

    // R> qchisq(c(5e-6, 1 - 5e-6), 99)
    // [1]  48.68125 173.87456
    double lower(48.68125);
    double upper(173.87456);

    bool prev_step  = true;
    bool prev_total = true;
    for (int i = 0; i < steps; ++i) {
        array step_hist  = histogram(randu(elem, ty, r), bins, 0.0, 1.0);
        double step_chi2 = chi2_statistic<T>(step_hist, expected);
        if (!prev_step) {
            EXPECT_GT(step_chi2, lower) << "at step: " << i;
            EXPECT_LT(step_chi2, upper) << "at step: " << i;
        }
        prev_step = step_chi2 > lower && step_chi2 < upper;

        total_hist += step_hist;
        double total_chi2 = chi2_statistic<T>(total_hist, expected);
        if (!prev_total) {
            EXPECT_GT(total_chi2, lower) << "at step: " << i;
            EXPECT_LT(total_chi2, upper) << "at step: " << i;
        }
        prev_total = total_chi2 > lower && total_chi2 < upper;
    }
}

#ifndef AF_CPU
TYPED_TEST(RandomEngine, philoxRandomEngineUniformChi2) {
    testRandomEngineUniformChi2<TypeParam>(AF_RANDOM_ENGINE_PHILOX_4X32_10);
}

TYPED_TEST(RandomEngine, threefryRandomEngineUniformChi2) {
    testRandomEngineUniformChi2<TypeParam>(AF_RANDOM_ENGINE_THREEFRY_2X32_16);
}

TYPED_TEST(RandomEngine, mersenneRandomEngineUniformChi2) {
    testRandomEngineUniformChi2<TypeParam>(AF_RANDOM_ENGINE_MERSENNE_GP11213);
}
#endif
