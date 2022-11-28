

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

using af::allTrue;
using af::array;
using af::constant;
using af::deviceGC;
using af::dtype;
using af::dtype_traits;
using af::randomEngine;
using af::randomEngineType;
using af::sum;

template<typename T>
class RandomEngine : public ::testing::Test {
   public:
    virtual void SetUp() {
        // Ensure all unlocked buffers are freed
        deviceGC();
        SUPPORTED_TYPE_CHECK(T);
    }
};

// create a list of types to be tested
typedef ::testing::Types<float, double> TestTypesEngine;
// register the type list
TYPED_TEST_SUITE(RandomEngine, TestTypesEngine);

template<typename T>
void testRandomEnginePeriod(randomEngineType type) {
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
double chi2_statistic(array input, array expected, bool print = false) {
    expected *= sum<T>(input) / sum<T>(expected);
    array diff = input - expected;

    double chi2 = sum<T>((diff * diff) / expected);
    if (print) {
        array legend = af::seq(input.elements());
        legend -= (input.elements() / 2.);
        legend *= (14. / input.elements());

        af_print(
            join(1, legend, expected.as(f32), input.as(f32), diff.as(f32)));
    }

    return chi2;
}

template<>
double chi2_statistic<half_float::half>(array input, array expected,
                                        bool print) {
    expected *= convert<float>(sum<float>(input)) /
                convert<float>(sum<float>(expected));
    array diff  = input - expected;
    double chi2 = convert<float>(sum<float>((diff * diff) / expected));
    return chi2;
}

template<typename T>
void testRandomEngineUniformChi2(randomEngineType type) {
    dtype ty = (dtype)dtype_traits<T>::af_type;

    int elem  = 256 * 1024 * 1024;
    int steps = 256;
    int bins  = 100;

    array total_hist = constant(0.0, bins, f32);
    array expected   = constant(1.0 / bins, bins, f32);

    randomEngine r(type, 0);

    // R> qchisq(c(5e-6, 1 - 5e-6), 99)
    // [1]  48.68125 173.87456
    float lower(48.68125);
    float upper(173.87456);

    bool prev_step  = true;
    bool prev_total = true;
    for (int i = 0; i < steps; ++i) {
        array rn_numbers = randu(elem, ty, r);
        array step_hist  = histogram(rn_numbers, bins, 0.0, 1.0);
        step_hist        = step_hist.as(f32);
        float step_chi2  = chi2_statistic<float>(step_hist, expected);
        if (!prev_step) {
            EXPECT_GT(step_chi2, lower) << "at step: " << i;
            EXPECT_LT(step_chi2, upper) << "at step: " << i;
        }
        prev_step = step_chi2 > lower && step_chi2 < upper;

        total_hist += step_hist;
        float total_chi2 = chi2_statistic<float>(total_hist, expected);
        if (!prev_total) {
            EXPECT_GT(total_chi2, lower) << "at step: " << i;
            EXPECT_LT(total_chi2, upper) << "at step: " << i;
        }
        prev_total = total_chi2 > lower && total_chi2 < upper;
    }
}

TYPED_TEST(RandomEngine, philoxRandomEngineUniformChi2) {
    testRandomEngineUniformChi2<TypeParam>(AF_RANDOM_ENGINE_PHILOX_4X32_10);
}

TYPED_TEST(RandomEngine, threefryRandomEngineUniformChi2) {
    testRandomEngineUniformChi2<TypeParam>(AF_RANDOM_ENGINE_THREEFRY_2X32_16);
}

TYPED_TEST(RandomEngine, mersenneRandomEngineUniformChi2) {
    testRandomEngineUniformChi2<TypeParam>(AF_RANDOM_ENGINE_MERSENNE_GP11213);
}

// should be used only for x <= 5 (roughly)
array cnd(array x) { return 0.5 * erfc(-x * sqrt(0.5)); }

template<typename T>
bool testRandomEngineNormalChi2(randomEngineType type)

{
    af::dtype ty = (af::dtype)af::dtype_traits<T>::af_type;

    int elem  = 256 * 1024 * 1024;
    int steps = 64;  // 256 * 32;
    int bins  = 100;

    T lower_edge(-7.0);
    T upper_edge(7.0);

    array total_hist = af::constant(0.0, 2 * bins, f32);
    array edges      = af::seq(bins + 1) / bins * lower_edge;
    array expected   = -af::diff1(cnd(edges));

    expected =
        af::join(0, expected(af::seq(bins - 1, 0, -1)), expected).as(f32);

    af::randomEngine r(type, 0);

    // NOTE(@rstub): In the chi^2 test one computes the test statistic and
    // compares the value with the chi^2 distribution with appropriate number of
    // degrees of freedom. For the uniform distribution one has "number of bins
    // minus 1" degrees of freedom. For the normal distribution it is "number of
    // bins minus 3", since there are two parameters mu and sigma. Here I used
    // the qchisq() function from R to compute "suitable" values from the chi^2
    // distribution.
    //
    // R> qchisq(c(5e-6, 1 - 5e-6), 197)
    // [1] 121.3197 297.2989
    float lower(121.3197);
    float upper(297.2989);

    bool prev_step  = true;
    bool prev_total = true;

    af::setSeed(0x76fa214467690e3c);

    // std::cout << std::setw(4) << "step" << std::setw(7) << "chi2_i"
    //           << std::setw(7) << "chi2_t" << std::setprecision(2) <<
    //           std::fixed
    //           << std::endl;

    for (int i = 0; i < steps; ++i) {
        array rn_numbers = randn(elem, ty, r);
        array step_hist =
            af::histogram(rn_numbers, 2 * bins, lower_edge, upper_edge);
        step_hist = step_hist.as(f32);

        float step_chi2 = chi2_statistic<float>(step_hist, expected);

        // if (step_chi2 > 10000) af_print(rn_numbers);
        // std::cout << std::setprecision(2) << std::fixed << std::setw(4) << i
        //           << std::setw(9) << step_chi2;

        bool step = step_chi2 > lower && step_chi2 < upper;

        if (!prev_step) {
            EXPECT_GT(step_chi2, lower) << "at step " << i;
            EXPECT_LT(step_chi2, upper) << "at step: " << i;
            if (step_chi2 < lower || step_chi2 > upper) {
                bool print = true;
                chi2_statistic<float>(step_hist, expected, print);
            }
        }

        // if (!(step || prev_step)) break;

        prev_step = step;
        total_hist += step_hist;

        float total_chi2 = chi2_statistic<float>(total_hist, expected);

        // std::cout << std::setw(9) << total_chi2 << std::endl;

        bool total = total_chi2 > lower && total_chi2 < upper;
        if (!prev_total) {
            EXPECT_GT(total_chi2, lower) << "at step " << i;
            EXPECT_LT(total_chi2, upper) << "at step " << i;
            if (total_chi2 < lower || total_chi2 > upper) {
                bool print = true;
                chi2_statistic<float>(total_hist, expected, print);
            }
        }

        prev_total = total;
    }

    return true;
}

TYPED_TEST(RandomEngine, philoxRandomEngineNormalChi2) {
    testRandomEngineNormalChi2<TypeParam>(AF_RANDOM_ENGINE_PHILOX_4X32_10);
}

TYPED_TEST(RandomEngine, threefryRandomEngineNormalChi2) {
    testRandomEngineNormalChi2<TypeParam>(AF_RANDOM_ENGINE_THREEFRY_2X32_16);
}

TYPED_TEST(RandomEngine, DISABLED_mersenneRandomEngineNormalChi2) {
    testRandomEngineNormalChi2<TypeParam>(AF_RANDOM_ENGINE_MERSENNE_GP11213);
}
