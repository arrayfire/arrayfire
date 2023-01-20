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

#include <iterator>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using std::back_inserter;
using std::move;
using std::string;
using std::vector;

af_err init_err = af_init();

template<typename T>
struct elseType {
    typedef typename cond_type<is_same_type<T, uintl>::value ||
                                   is_same_type<T, intl>::value,
                               double, T>::type type;
};

template<typename T>
struct varOutType {
    typedef typename cond_type<
        is_same_type<T, float>::value || is_same_type<T, int>::value ||
            is_same_type<T, uint>::value || is_same_type<T, short>::value ||
            is_same_type<T, ushort>::value || is_same_type<T, uchar>::value ||
            is_same_type<T, char>::value,
        float, typename elseType<T>::type>::type type;
};

template<typename T>
using outType = typename varOutType<T>::type;

template<typename T>
struct meanvar_test {
    static af_dtype af_type;
    string test_description_;
    af_array in_;
    af_array weights_;
    af_var_bias bias_;
    int dim_;
    vector<outType<T>> mean_;
    vector<outType<T>> variance_;

    meanvar_test(string description, af_array in, af_array weights,
                 af_var_bias bias, int dim,
                 vector<typename varOutType<T>::type> &&mean,
                 vector<typename varOutType<T>::type> &&variance)
        : test_description_(description)
        , in_(0)
        , weights_(0)
        , bias_(bias)
        , dim_(dim) {
        af_retain_array(&in_, in);
        if (weights) { af_retain_array(&weights_, weights); }
        mean_.reserve(mean.size());
        variance_.reserve(variance.size());
        for (auto &v : mean) mean_.push_back((outType<T>)v);
        for (auto &v : variance) variance_.push_back((outType<T>)v);
    }

    meanvar_test(std::string name)
        : test_description_(name), in_(0), weights_(0) {}

    meanvar_test(meanvar_test<T> &&other)
        : test_description_(other.test_description_)
        , in_(other.in_)
        , weights_(other.weights_)
        , bias_(other.bias_)
        , dim_(other.dim_)
        , mean_(other.mean_)
        , variance_(other.variance_) {
        other.in_      = 0;
        other.weights_ = 0;
    }
    meanvar_test &operator=(meanvar_test<T> &&other) = default;
    meanvar_test &operator=(meanvar_test<T> &other)  = delete;

    meanvar_test(const meanvar_test<T> &other)
        : test_description_(other.test_description_)
        , in_(0)
        , weights_(0)
        , bias_(other.bias_)
        , dim_(other.dim_)
        , mean_(other.mean_)
        , variance_(other.variance_) {
        if (other.in_) af_retain_array(&in_, other.in_);
        if (other.weights_) { af_retain_array(&weights_, other.weights_); }
    }

    ~meanvar_test() {
#ifndef _WIN32
        if (in_) af_release_array(in_);
        if (weights_) {
            af_release_array(weights_);
            weights_ = 0;
        }
#endif
    }
};

template<typename T>
af_dtype meanvar_test<T>::af_type = dtype_traits<T>::af_type;

template<typename T>
class MeanVarTyped : public ::testing::TestWithParam<meanvar_test<T>> {
   public:
    void meanvar_test_function(const meanvar_test<T> &test) {
        SUPPORTED_TYPE_CHECK(T);
        SUPPORTED_TYPE_CHECK(outType<T>);
        af_array mean, var;

        // Cast to the expected type
        af_array in = 0;
        ASSERT_SUCCESS(
            af_cast(&in, test.in_, (af_dtype)dtype_traits<T>::af_type));

        EXPECT_EQ(AF_SUCCESS, af_meanvar(&mean, &var, in, test.weights_,
                                         test.bias_, test.dim_));

        vector<outType<T>> h_mean(test.mean_.size()),
            h_var(test.variance_.size());

        dim4 outDim(1);
        af_get_dims(&outDim[0], &outDim[1], &outDim[2], &outDim[3], in);
        outDim[test.dim_] = 1;

        if (is_same_type<half_float::half, outType<T>>::value) {
            ASSERT_VEC_ARRAY_NEAR(test.mean_, outDim, mean, 1.f);
            ASSERT_VEC_ARRAY_NEAR(test.variance_, outDim, var, 0.5f);
        } else if (is_same_type<float, outType<T>>::value ||
                   is_same_type<cfloat, outType<T>>::value) {
            ASSERT_VEC_ARRAY_NEAR(test.mean_, outDim, mean, 0.001f);
            ASSERT_VEC_ARRAY_NEAR(test.variance_, outDim, var, 0.2f);
        } else {
            ASSERT_VEC_ARRAY_NEAR(test.mean_, outDim, mean, 0.00001f);
            ASSERT_VEC_ARRAY_NEAR(test.variance_, outDim, var, 0.0001f);
        }

        ASSERT_SUCCESS(af_release_array(in));
        ASSERT_SUCCESS(af_release_array(mean));
        ASSERT_SUCCESS(af_release_array(var));
    }

    void meanvar_cpp_test_function(const meanvar_test<T> &test) {
        SUPPORTED_TYPE_CHECK(T);
        SUPPORTED_TYPE_CHECK(outType<T>);
        array mean, var;

        // Cast to the expected type
        af_array in_tmp = 0;
        ASSERT_SUCCESS(af_retain_array(&in_tmp, test.in_));
        array in(in_tmp);
        in = in.as((af_dtype)dtype_traits<T>::af_type);

        af_array weights_tmp = test.weights_;
        if (weights_tmp) {
            ASSERT_SUCCESS(af_retain_array(&weights_tmp, weights_tmp));
        }
        array weights(weights_tmp);
        meanvar(mean, var, in, weights, test.bias_, test.dim_);

        vector<outType<T>> h_mean(test.mean_.size()),
            h_var(test.variance_.size());

        dim4 outDim       = in.dims();
        outDim[test.dim_] = 1;

        if (is_same_type<half_float::half, outType<T>>::value) {
            ASSERT_VEC_ARRAY_NEAR(test.mean_, outDim, mean, 1.f);
            ASSERT_VEC_ARRAY_NEAR(test.variance_, outDim, var, 0.5f);
        } else if (is_same_type<float, outType<T>>::value ||
                   is_same_type<cfloat, outType<T>>::value) {
            ASSERT_VEC_ARRAY_NEAR(test.mean_, outDim, mean, 0.001f);
            ASSERT_VEC_ARRAY_NEAR(test.variance_, outDim, var, 0.2f);
        } else {
            ASSERT_VEC_ARRAY_NEAR(test.mean_, outDim, mean, 0.00001f);
            ASSERT_VEC_ARRAY_NEAR(test.variance_, outDim, var, 0.0001f);
        }
    }
};

af_array empty = 0;

enum test_size { MEANVAR_SMALL, MEANVAR_LARGE };

template<typename T>
meanvar_test<T> meanvar_test_gen(string name, int in_index, int weight_index,
                                 af_var_bias bias, int dim, int mean_index,
                                 int var_index, test_size size) {
    if (noDoubleTests((af_dtype)af::dtype_traits<T>::af_type) ||
        noDoubleTests((
            af_dtype)af::dtype_traits<typename varOutType<T>::type>::af_type) ||
        noHalfTests((af_dtype)af::dtype_traits<T>::af_type)) {
        meanvar_test<T> out(name);
        return out;
    }

    vector<af_array> inputs;
    vector<vector<typename varOutType<T>::type>> outputs;
    if (size == MEANVAR_SMALL) {
        vector<af::dim4> numDims_;
        vector<vector<T>> in_;
        vector<vector<typename varOutType<T>::type>> tests_;
        readTests<T, typename varOutType<T>::type, double>(
            TEST_DIR "/meanvar/meanvar.data", numDims_, in_, tests_);

        inputs.resize(in_.size());
        for (size_t i = 0; i < in_.size(); i++) {
            af_create_array(&inputs[i], &in_[i].front(), numDims_[i].ndims(),
                            numDims_[i].get(),
                            (af_dtype)af::dtype_traits<T>::af_type);
        }

        outputs.resize(tests_.size());
        for (size_t i = 0; i < tests_.size(); i++) {
            copy(tests_[i].begin(), tests_[i].end(), back_inserter(outputs[i]));
        }
    } else {
        dim_t full_array_size            = 2000;
        vector<vector<dim_t>> dimensions = {
            {2000, 1, 1, 1},  // 0
            {1, 2000, 1, 1},  // 1
            {1, 1, 2000, 1},  // 2

            {500, 4, 1, 1},  // 3
            {4, 500, 1, 1},  // 4
            {50, 40, 1, 1}   // 5
        };

        vector<T> large_(full_array_size);
        for (size_t i = 0; i < large_.size(); i++) {
            large_[i] = static_cast<T>(i);
        }

        inputs.resize(dimensions.size());
        for (size_t i = 0; i < dimensions.size(); i++) {
            af_create_array(&inputs[i], &large_.front(), 4,
                            dimensions[i].data(),
                            (af_dtype)af::dtype_traits<T>::af_type);
        }

        outputs.push_back(
            vector<typename varOutType<T>::type>(1, outType<T>(999.5)));
        outputs.push_back(
            vector<typename varOutType<T>::type>(1, outType<T>(333500)));
        outputs.push_back({outType<T>(249.50), outType<T>(749.50),
                           outType<T>(1249.50), outType<T>(1749.50)});
        outputs.push_back(
            vector<typename varOutType<T>::type>(4, outType<T>(20875)));
    }
    meanvar_test<T> out(name, inputs[in_index],
                        (weight_index == -1) ? empty : inputs[weight_index],
                        bias, dim, move(outputs[mean_index]),
                        move(outputs[var_index]));

    for (auto input : inputs) { af_release_array(input); }
    return out;
}

template<typename T>
vector<meanvar_test<T>> small_test_values() {
    // clang-format off
    return {
        //                  |           Name |   in_index | weight_index |                  bias |  dim | mean_index | var_index |
        meanvar_test_gen<T>(   "Sample1Ddim0",           0,            -1,     AF_VARIANCE_SAMPLE,     0,           0,          1, MEANVAR_SMALL),
        meanvar_test_gen<T>(   "Sample1Ddim1",           1,            -1,     AF_VARIANCE_SAMPLE,     1,           0,          1, MEANVAR_SMALL),
        meanvar_test_gen<T>(   "Sample2Ddim0",           2,            -1,     AF_VARIANCE_SAMPLE,     0,           3,          4, MEANVAR_SMALL),
        meanvar_test_gen<T>(   "Sample2Ddim1",           2,            -1,     AF_VARIANCE_SAMPLE,     1,           6,          7, MEANVAR_SMALL),

        meanvar_test_gen<T>("Population1Ddim0",          0,            -1, AF_VARIANCE_POPULATION,     0,           0,          2, MEANVAR_SMALL),
        meanvar_test_gen<T>("Population1Ddim1",          1,            -1, AF_VARIANCE_POPULATION,     1,           0,          2, MEANVAR_SMALL),
        meanvar_test_gen<T>("Population2Ddim0",          2,            -1, AF_VARIANCE_POPULATION,     0,           3,          5, MEANVAR_SMALL),
        meanvar_test_gen<T>("Population2Ddim1",          2,            -1, AF_VARIANCE_POPULATION,     1,           6,          8, MEANVAR_SMALL)};
    // clang-format on
}

template<typename T>
vector<meanvar_test<T>> large_test_values() {
    return {
        // clang-format off
        //                  |       Name |      in_index | weight_index |                  bias |  dim | mean_index | var_index |
        meanvar_test_gen<T>("Sample1Ddim0",             0,            -1,     AF_VARIANCE_SAMPLE,     0,           0,          1, MEANVAR_LARGE),
        meanvar_test_gen<T>("Sample1Ddim1",             1,            -1,     AF_VARIANCE_SAMPLE,     1,           0,          1, MEANVAR_LARGE),
        meanvar_test_gen<T>("Sample1Ddim2",             2,            -1,     AF_VARIANCE_SAMPLE,     2,           0,          1, MEANVAR_LARGE),
        meanvar_test_gen<T>("Sample2Ddim0",             3,            -1,     AF_VARIANCE_SAMPLE,     0,           2,          3, MEANVAR_LARGE),
        // TODO(umar) Add additional large tests
        // meanvar_test_gen<T>(    "Sample2Ddim1",           3,            -1, AF_VARIANCE_SAMPLE,     1,           2,          3, MEANVAR_LARGE),
        // meanvar_test_gen<T>(    "Sample2Ddim1",           2,            -1, AF_VARIANCE_SAMPLE,     1,           6,          7, MEANVAR_LARGE),
        // clang-format on
    };
}

#define MEANVAR_TEST(NAME, TYPE)                                              \
    using MeanVar##NAME = MeanVarTyped<TYPE>;                                 \
    INSTANTIATE_TEST_SUITE_P(                                                 \
        Small, MeanVar##NAME, ::testing::ValuesIn(small_test_values<TYPE>()), \
        [](const ::testing::TestParamInfo<MeanVar##NAME::ParamType> info) {   \
            return info.param.test_description_;                              \
        });                                                                   \
    INSTANTIATE_TEST_SUITE_P(                                                 \
        Large, MeanVar##NAME, ::testing::ValuesIn(large_test_values<TYPE>()), \
        [](const ::testing::TestParamInfo<MeanVar##NAME::ParamType> info) {   \
            return info.param.test_description_;                              \
        });                                                                   \
                                                                              \
    TEST_P(MeanVar##NAME, Testing) {                                          \
        const meanvar_test<TYPE> &test = GetParam();                          \
        meanvar_test_function(test);                                          \
    }                                                                         \
    TEST_P(MeanVar##NAME, TestingCPP) {                                       \
        const meanvar_test<TYPE> &test = GetParam();                          \
        meanvar_cpp_test_function(test);                                      \
    }

MEANVAR_TEST(Float, float)
MEANVAR_TEST(Double, double)
MEANVAR_TEST(Int, int)
MEANVAR_TEST(UnsignedInt, unsigned int)
MEANVAR_TEST(Short, short)
MEANVAR_TEST(UnsignedShort, unsigned short)
MEANVAR_TEST(Long, long long)
MEANVAR_TEST(UnsignedLong, unsigned long long)
MEANVAR_TEST(ComplexFloat, af::af_cfloat)
MEANVAR_TEST(ComplexDouble, af::af_cdouble)

#undef MEANVAR_TEST

using MeanVarHalf = MeanVarTyped<half_float::half>;
INSTANTIATE_TEST_SUITE_P(
    Small, MeanVarHalf,
    ::testing::ValuesIn(small_test_values<half_float::half>()),
    [](const ::testing::TestParamInfo<MeanVarHalf::ParamType> info) {
        return info.param.test_description_;
    });
TEST_P(MeanVarHalf, Testing) {
    const meanvar_test<half_float::half> &test = GetParam();
    meanvar_test_function(test);
}
TEST_P(MeanVarHalf, TestingCPP) {
    const meanvar_test<half_float::half> &test = GetParam();
    meanvar_cpp_test_function(test);
}

#define MEANVAR_TEST(NAME, TYPE)                                              \
    using MeanVar##NAME = MeanVarTyped<TYPE>;                                 \
    INSTANTIATE_TEST_SUITE_P(                                                 \
        Small, MeanVar##NAME, ::testing::ValuesIn(small_test_values<TYPE>()), \
        [](const ::testing::TestParamInfo<MeanVar##NAME::ParamType> &info) {  \
            return info.param.test_description_;                              \
        });                                                                   \
                                                                              \
    TEST_P(MeanVar##NAME, Testing) {                                          \
        const meanvar_test<TYPE> &test = GetParam();                          \
        meanvar_test_function(test);                                          \
    }                                                                         \
    TEST_P(MeanVar##NAME, TestingCPP) {                                       \
        const meanvar_test<TYPE> &test = GetParam();                          \
        meanvar_cpp_test_function(test);                                      \
    }

// Only test small sizes because the range of the large arrays go out of bounds
MEANVAR_TEST(UnsignedChar, unsigned char)
// MEANVAR_TEST(Bool, unsigned char) // TODO(umar): test this type
