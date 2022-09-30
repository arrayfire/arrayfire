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
#include <half.hpp>
#include <testHelpers.hpp>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::constant;
using af::dim4;
using af::randu;
using half_float::half;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Mean : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
// This list does not allow to cleanly add the af_half/half_float type : at the
// moment half tested in some special unittests
typedef ::testing::Types<cdouble, cfloat, float, double, int, uint, intl, uintl,
                         char, uchar, short, ushort, half_float::half>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Mean, TestTypes);

template<typename T>
struct f32HelperType {
    typedef
        typename cond_type<is_same_type<T, double>::value, double, float>::type
            type;
};

template<typename T>
struct c32HelperType {
    typedef typename cond_type<is_same_type<T, cfloat>::value, cfloat,
                               typename f32HelperType<T>::type>::type type;
};

template<typename T>
struct elseType {
    typedef typename cond_type<is_same_type<T, uintl>::value ||
                                   is_same_type<T, intl>::value,
                               double, T>::type type;
};

template<typename T>
struct meanOutType {
    typedef typename cond_type<
        is_same_type<T, float>::value || is_same_type<T, int>::value ||
            is_same_type<T, uint>::value || is_same_type<T, uchar>::value ||
            is_same_type<T, short>::value || is_same_type<T, ushort>::value ||
            is_same_type<T, char>::value,
        float, typename elseType<T>::type>::type type;
};

template<typename T>
void meanDimTest(string pFileName, dim_t dim, bool isWeighted = false) {
    typedef typename meanOutType<T>::type outType;
    SUPPORTED_TYPE_CHECK(T);
    SUPPORTED_TYPE_CHECK(outType);

    double tol = 1.0e-3;
    if ((af_dtype)af::dtype_traits<T>::af_type == f16) tol = 4.e-3;
    vector<dim4> numDims;
    vector<vector<int>> in;
    vector<vector<float>> tests;

    readTestsFromFile<int, float>(pFileName, numDims, in, tests);

    dim4 goldDims = numDims[0];
    goldDims[dim] = 1;
    if (!isWeighted) {
        dim4 dims = numDims[0];
        vector<T> input(in[0].begin(), in[0].end());

        array inArray(dims, &(input.front()));

        array outArray = mean(inArray, dim);

        vector<outType> outData(dims.elements());

        outArray.host((void*)outData.data());

        vector<outType> currGoldBar(tests[0].begin(), tests[0].end());

        dim4 goldDims = dims;
        goldDims[dim] = 1;
        ASSERT_VEC_ARRAY_NEAR(currGoldBar, goldDims, outArray, tol);
    } else {
        dim4 dims  = numDims[0];
        dim4 wdims = numDims[1];
        vector<T> input(in[0].begin(), in[0].end());
        vector<float> weights(in[1].size());
        transform(in[1].begin(), in[1].end(), weights.begin(),
                  convert_to<float, int>);

        array inArray(dims, &(input.front()));
        array wtsArray(wdims, &(weights.front()));

        array outArray = mean(inArray, wtsArray, dim);

        vector<outType> outData(dims.elements());

        outArray.host((void*)outData.data());

        vector<outType> currGoldBar(tests[0].begin(), tests[0].end());

        ASSERT_VEC_ARRAY_NEAR(currGoldBar, goldDims, outArray, tol);
    }
}

TYPED_TEST(Mean, Dim0Matrix) {
    meanDimTest<TypeParam>(string(TEST_DIR "/mean/mean_dim0_matrix.test"), 0);
}

TYPED_TEST(Mean, Dim1Cube) {
    meanDimTest<TypeParam>(string(TEST_DIR "/mean/mean_dim1_cube.test"), 1);
}

TYPED_TEST(Mean, Dim0HyperCube) {
    meanDimTest<TypeParam>(string(TEST_DIR "/mean/mean_dim0_hypercube.test"),
                           0);
}

TYPED_TEST(Mean, Dim2Matrix) {
    meanDimTest<TypeParam>(string(TEST_DIR "/mean/mean_dim2_matrix.test"), 2);
}

TYPED_TEST(Mean, Dim2Cube) {
    meanDimTest<TypeParam>(string(TEST_DIR "/mean/mean_dim2_cube.test"), 2);
}

TYPED_TEST(Mean, Dim2HyperCube) {
    meanDimTest<TypeParam>(string(TEST_DIR "/mean/mean_dim2_hypercube.test"),
                           2);
}

TYPED_TEST(Mean, Wtd_Dim0Matrix) {
    meanDimTest<TypeParam>(string(TEST_DIR "/mean/wtd_mean_dim0_mat.test"), 0,
                           true);
}

TYPED_TEST(Mean, Wtd_Dim1Matrix) {
    meanDimTest<TypeParam>(string(TEST_DIR "/mean/wtd_mean_dim1_mat.test"), 1,
                           true);
}

template<typename T>
void meanAllTest(T const_value, dim4 dims) {
    typedef typename meanOutType<T>::type outType;

    SUPPORTED_TYPE_CHECK(T);
    SUPPORTED_TYPE_CHECK(outType);

    using af::array;
    using af::mean;

    vector<T> hundred(dims.elements(), const_value);

    outType gold = outType(0);
    // for(auto i:hundred) gold += i;
    for (int i = 0; i < (int)hundred.size(); i++) { gold = gold + hundred[i]; }
    gold = gold / dims.elements();

    array a(dims, &(hundred.front()));
    outType output = mean<outType>(a);

    ASSERT_NEAR(::real(output), ::real(gold), 1.0e-3);
    ASSERT_NEAR(::imag(output), ::imag(gold), 1.0e-3);
}

template<>
void meanAllTest(half_float::half const_value, dim4 dims) {
    SUPPORTED_TYPE_CHECK(half_float::half);

    using af::array;
    using af::mean;

    vector<float> hundred(dims.elements(), const_value);

    float gold = float(0);
    for (int i = 0; i < (int)hundred.size(); i++) { gold = gold + hundred[i]; }
    gold = gold / dims.elements();

    array a         = array(dims, &(hundred.front())).as(f16);
    half output     = mean<half>(a);
    af_half output2 = mean<af_half>(a);

    // make sure output2 and output are binary equals. This is necessary
    // because af_half is not a complete type
    half output2_copy;
    memcpy(static_cast<void*>(&output2_copy), &output2, sizeof(af_half));
    ASSERT_EQ(output, output2_copy);

    ASSERT_NEAR(output, gold, 1.0e-3);
}

TEST(MeanAll, f64) { meanAllTest<double>(2.1, dim4(10, 10, 1, 1)); }

TEST(MeanAll, f32) { meanAllTest<float>(2.1f, dim4(10, 5, 2, 1)); }

TEST(MeanAll, f16) { meanAllTest<half>((half)0.3f, dim4(10, 5, 2, 1)); }

TEST(MeanAll, s32) { meanAllTest<int>(2, dim4(5, 5, 2, 2)); }

TEST(MeanAll, u32) { meanAllTest<unsigned>(2, dim4(100, 1, 1, 1)); }

TEST(MeanAll, s8) { meanAllTest<char>(2, dim4(5, 5, 2, 2)); }

TEST(MeanAll, u8) { meanAllTest<uchar>(2, dim4(100, 1, 1, 1)); }

TEST(MeanAll, c32) { meanAllTest<cfloat>(cfloat(2.1f), dim4(10, 5, 2, 1)); }

TEST(MeanAll, s16) { meanAllTest<short>(2, dim4(5, 5, 2, 2)); }

TEST(MeanAll, u16) { meanAllTest<ushort>(2, dim4(100, 1, 1, 1)); }

TEST(MeanAll, c64) { meanAllTest<cdouble>(cdouble(2.1), dim4(10, 10, 1, 1)); }

template<typename T>
T random() {
    return T(std::rand() % 10);
}

template<>
half random<half>() {
    // create values from -0.5 to 0.5 to ensure sum does not deviate
    // too far out of half's useful range
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
    return half(r);
}

template<>
cfloat random<cfloat>() {
    return cfloat(float(std::rand() % 10), float(std::rand() % 10));
}

template<>
cdouble random<cdouble>() {
    return cdouble(double(std::rand() % 10), double(std::rand() % 10));
}

template<typename T>
class WeightedMean : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// register the type list
TYPED_TEST_SUITE(WeightedMean, TestTypes);

template<typename T, typename wtsType>
void weightedMeanAllTest(dim4 dims) {
    typedef typename meanOutType<T>::type outType;

    SUPPORTED_TYPE_CHECK(T);
    SUPPORTED_TYPE_CHECK(outType);
    SUPPORTED_TYPE_CHECK(wtsType);

    using af::array;
    using af::mean;

    std::srand(std::time(0));

    vector<T> data(dims.elements());
    vector<wtsType> wts(dims.elements());
    std::generate(data.begin(), data.end(), random<T>);
    std::generate(wts.begin(), wts.end(), random<wtsType>);

    outType wtdSum = outType(0);
    wtsType wtsSum = wtsType(0);

    for (int i = 0; i < (int)data.size(); i++) {
        wtdSum = wtdSum + data[i] * wts[i];
        wtsSum = wtsSum + wts[i];
    }

    outType gold = wtdSum / outType(wtsSum);

    array a(dims, &(data.front()));
    array w(dims, &(wts.front()));
    outType output = mean<outType>(a, w);

    ASSERT_NEAR(::real(output), ::real(gold), 1.0e-2);
    ASSERT_NEAR(::imag(output), ::imag(gold), 1.0e-2);
}

TYPED_TEST(WeightedMean, Basic) {
    weightedMeanAllTest<TypeParam, float>(dim4(32, 30, 33, 17));
}

TEST(WeightedMean, Broadacst) {
    float val = 0.5f;
    array a   = randu(4096, 32);
    array w   = constant(val, a.dims());
    array c   = mean(a);
    array d   = mean(a, w);

    vector<float> hc(c.elements());
    vector<float> hd(d.elements());

    c.host(hc.data());
    d.host(hd.data());

    for (size_t i = 0; i < hc.size(); i++) {
        // C and D are the same because they are normalized by the sum of the
        // weights.
        ASSERT_NEAR(hc[i], hd[i], 1E-5);
    }
}

TEST(Mean, Issue2093) {
    const int NELEMS = 512;

    array data = randu(1, NELEMS);
    array wts  = constant(1.0f, 1, NELEMS);
    vector<float> hdata(NELEMS);
    data.host(hdata.data());

    array out = mean(data, wts, 1);
    float outVal;
    out.host(&outVal);

    float expected = 0.0;
    for (size_t i = 0; i < NELEMS; ++i) expected += hdata[i];
    expected /= NELEMS;

    ASSERT_NEAR(outVal, expected, 0.001);
}

TEST(MeanAll, SubArray) {
    // Fixes Issue 2636
    using af::mean;
    using af::span;
    using af::sum;

    const dim4 inDims(10, 10, 10, 10);

    array in  = randu(inDims);
    array sub = in(0, span, span, span);

    size_t nElems = sub.elements();
    ASSERT_FLOAT_EQ(mean<float>(sub), sum<float>(sub) / nElems);
}

TEST(MeanHalf, dim0) {
    SUPPORTED_TYPE_CHECK(half_float::half);
    // Keeping N low to be able to run on 6GB GPUs
    int N = 1024;
    const dim4 inDims(N, N, 1, 1);
    array in  = randu(inDims, f16);
    array m16 = af::mean(in, 0);
    array m32 = af::mean(in.as(f32), 0);
    // Some diffs appears at 0.0001 max diff : example: float: 0.507014 vs half:
    // 0.506836
    ASSERT_ARRAYS_NEAR(m16.as(f32), m32, 0.001f);
}
