/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#define GTEST_LINKED_AS_SHARED_LIBRARY 1

#include <gtest/gtest.h>
#include <arrayfire.h>

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

template<typename T>
struct elseType {
  typedef typename cond_type< is_same_type<T, uintl>::value ||
                              is_same_type<T, intl> ::value,
                              double,
                              T>::type type;
};

template<typename T>
struct varOutType {
  typedef typename cond_type< is_same_type<T, float >::value ||
                              is_same_type<T, int   >::value ||
                              is_same_type<T, uint  >::value ||
                              is_same_type<T, short >::value ||
                              is_same_type<T, ushort>::value ||
                              is_same_type<T, uchar >::value ||
                              is_same_type<T, char  >::value,
                              float,
                              typename elseType<T>::type>::type type;
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
                vector<double> &&mean, vector<double> &&variance)
      : test_description_(description)
      , in_(0)
      , weights_(0)
      , bias_(bias)
      , dim_(dim) {
        af_retain_array(&in_, in);
        if (weights) {
            af_retain_array(&weights_, weights);
        }
        mean_.reserve(mean.size());
        variance_.reserve(variance.size());
        std::copy(begin(mean), end(mean), back_inserter(mean_));
        std::copy(begin(variance), end(variance), back_inserter(variance_));
    }

    meanvar_test(const meanvar_test<T> &other)
      : test_description_(other.test_description_)
      , in_(0)
      , weights_(0)
      , bias_(other.bias_)
      , dim_(other.dim_)
      , mean_(other.mean_)
      , variance_(other.variance_) {
        af_retain_array(&in_, other.in_);
        if (other.weights_) {
            af_retain_array(&weights_, other.weights_);
        }
    }

    ~meanvar_test() {
        af_release_array(in_);
        if (weights_) {
            af_release_array(weights_);
            weights_ = 0;
        }
    }

    meanvar_test() = default;
    meanvar_test(meanvar_test<T> &&other) = default;
    meanvar_test& operator=(meanvar_test<T> &&other) = default;
};

template<typename T>
af_dtype meanvar_test<T>::af_type = dtype_traits<T>::af_type;

template<typename T>
class MeanVarTyped : public ::testing::TestWithParam<meanvar_test<T> > {
public:
  void meanvar_test_function(meanvar_test<T>& test) {
      af_array mean, var;

      // Cast to the expected type
      af_array in = 0;
      af_cast(&in, test.in_, (af_dtype) dtype_traits<T>::af_type);

      EXPECT_EQ(AF_SUCCESS, af_meanvar(&mean, &var, in, test.weights_, test.bias_, test.dim_));

      vector<outType<T>> h_mean(test.mean_.size()), h_var(test.variance_.size());

      dim4 outDim(1);
      af_get_dims(&outDim[0], &outDim[1], &outDim[2], &outDim[3], in);
      outDim[test.dim_] = 1;

      if (is_same_type<float, outType<T>>::value ||
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
};

af_array empty = 0;

enum test_size {
  MEANVAR_SMALL,
  MEANVAR_LARGE
};

template<typename T>
meanvar_test<T>
meanvar_test_gen(string name, int in_index, int weight_index, af_var_bias bias, int dim, int mean_index, int var_index, test_size size) {

    vector<af_array> inputs;
    vector<vector<double>> outputs;
    if(size == MEANVAR_SMALL) {
        vector<af::dim4> numDims_;
        vector<vector<double> > in_;
        vector<vector<double> > tests_;
        readTests<double, typename varOutType<double>::type, double> (TEST_DIR"/meanvar/meanvar.data", numDims_, in_, tests_);

        inputs.resize(in_.size());
        for(size_t i = 0; i < in_.size(); i++) {
            af_create_array(&inputs[i], &in_[i].front(),
                            numDims_[i].ndims(), numDims_[i].get(), f64);
        }

        outputs.resize(tests_.size());
        for(size_t i = 0; i < tests_.size(); i++) {
            copy(tests_[i].begin(), tests_[i].end(), back_inserter(outputs[i]));
        }
    } else {

      dim_t full_array_size = 2000;
      vector<vector<dim_t> > dimensions = {
                                          {2000, 1, 1, 1},  // 0
                                          {1, 2000, 1, 1},  // 1
                                          {1, 1, 2000, 1},  // 2

                                          {500, 4, 1, 1},  // 3
                                          {4, 500, 1, 1},  // 4
                                          {50, 40, 1, 1}   // 5
      };

      vector<double> large_(full_array_size);
      for(size_t i = 0; i < large_.size(); i++) {
          large_[i] = static_cast<double>(i);
      }

      inputs.resize(dimensions.size());
      for(size_t i = 0; i < dimensions.size(); i++) {
          af_create_array(&inputs[i], &large_.front(), 4, dimensions[i].data(), f64);
      }

      outputs.push_back(vector<double>(1, 999.5));
      outputs.push_back(vector<double>(1, 333500));
      outputs.push_back({249.50, 749.50, 1249.50, 1749.50});
      outputs.push_back(vector<double>(4, 20875));
    }
    meanvar_test<T> out = meanvar_test<T> (name,
                                           inputs[in_index],
                                           (weight_index == -1) ? empty : inputs[weight_index],
                                           bias,
                                           dim,
                                           move(outputs[mean_index]),
                                           move(outputs[var_index]));

    for(auto input : inputs) {
      af_release_array(input);
    }
    return out;
}


template<typename T>
vector<meanvar_test<T> >
small_test_values() {
  return {
                    // |           Name |   in_index | weight_index |                  bias |  dim | mean_index | var_index |
    meanvar_test_gen<T>(    "Sample1Ddim0",           0,            -1,     AF_VARIANCE_SAMPLE,     0,           0,          1, MEANVAR_SMALL),
    meanvar_test_gen<T>(    "Sample1Ddim1",           1,            -1,     AF_VARIANCE_SAMPLE,     1,           0,          1, MEANVAR_SMALL),
    meanvar_test_gen<T>(    "Sample2Ddim0",           2,            -1,     AF_VARIANCE_SAMPLE,     0,           3,          4, MEANVAR_SMALL),
    meanvar_test_gen<T>(    "Sample2Ddim1",           2,            -1,     AF_VARIANCE_SAMPLE,     1,           6,          7, MEANVAR_SMALL),

    meanvar_test_gen<T>("Population1Ddim0",           0,            -1, AF_VARIANCE_POPULATION,     0,           0,          2, MEANVAR_SMALL),
    meanvar_test_gen<T>("Population1Ddim1",           1,            -1, AF_VARIANCE_POPULATION,     1,           0,          2, MEANVAR_SMALL),
    meanvar_test_gen<T>("Population2Ddim0",           2,            -1, AF_VARIANCE_POPULATION,     0,           3,          5, MEANVAR_SMALL),
    meanvar_test_gen<T>("Population2Ddim1",           2,            -1, AF_VARIANCE_POPULATION,     1,           6,          8, MEANVAR_SMALL)
    };
}

template<typename T>
vector<meanvar_test<T> >
large_test_values() {
  return {
                      // |           Name |   in_index | weight_index |                  bias |  dim | mean_index | var_index |
    meanvar_test_gen<T>(    "Sample1Ddim0",           0,            -1,     AF_VARIANCE_SAMPLE,     0,           0,          1, MEANVAR_LARGE),
    meanvar_test_gen<T>(    "Sample1Ddim1",           1,            -1,     AF_VARIANCE_SAMPLE,     1,           0,          1, MEANVAR_LARGE),
    meanvar_test_gen<T>(    "Sample1Ddim2",           2,            -1,     AF_VARIANCE_SAMPLE,     2,           0,          1, MEANVAR_LARGE),
    meanvar_test_gen<T>(    "Sample2Ddim0",           3,            -1,     AF_VARIANCE_SAMPLE,     0,           2,          3, MEANVAR_LARGE),
    // TODO(umar) Add additional large tests
    //meanvar_test_gen<T>(    "Sample2Ddim1",           3,            -1,     AF_VARIANCE_SAMPLE,     1,           2,          3, MEANVAR_LARGE),
    //meanvar_test_gen<T>(    "Sample2Ddim1",           2,            -1,     AF_VARIANCE_SAMPLE,     1,           6,          7, MEANVAR_LARGE),
      };
}

#define MEANVAR_TEST(NAME, TYPE)                                        \
  using MeanVar##NAME = MeanVarTyped<TYPE>;                             \
  INSTANTIATE_TEST_CASE_P(Small,                                        \
                          MeanVar##NAME,                                \
                          ::testing::ValuesIn(small_test_values<TYPE>()), \
                          []( const ::testing::TestParamInfo<MeanVar##NAME::ParamType> info) { \
                            return info.param.test_description_;        \
                          });                                           \
  INSTANTIATE_TEST_CASE_P(Large,                                        \
                          MeanVar##NAME,                                \
                          ::testing::ValuesIn(large_test_values<TYPE>()), \
                          []( const ::testing::TestParamInfo<MeanVar##NAME::ParamType> info) { \
                            return info.param.test_description_;        \
                          });                                           \
                                                                        \
  TEST_P(MeanVar##NAME, Testing) {                                      \
    meanvar_test<TYPE> test = GetParam();                               \
    meanvar_test_function(test);                                        \
  }                                                                     \

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

#define MEANVAR_TEST(NAME, TYPE)                                        \
  using MeanVar##NAME = MeanVarTyped<TYPE>;                             \
  INSTANTIATE_TEST_CASE_P(Small,                                        \
                          MeanVar##NAME,                                \
                          ::testing::ValuesIn(small_test_values<TYPE>()), \
                          []( const ::testing::TestParamInfo<MeanVar##NAME::ParamType> info) { \
                            return info.param.test_description_;        \
                          });                                           \
                                                                        \
  TEST_P(MeanVar##NAME, Testing) {                                      \
    meanvar_test<TYPE> test = GetParam();                               \
    meanvar_test_function(test);                                        \
  }                                                                     \

// Only test small sizes because the range of the large arrays go out of bounds
MEANVAR_TEST(UnsignedChar, unsigned char)
//MEANVAR_TEST(Bool, unsigned char) // TODO(umar): test this type
