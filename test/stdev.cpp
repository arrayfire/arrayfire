/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
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
using af::dim4;
using af::exception;
using af::seq;
using af::stdev;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template <typename T>
class StandardDev : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, intl, uintl, char, uchar>
    TestTypes;

// register the type list
TYPED_TEST_CASE(StandardDev, TestTypes);

template <typename T>
struct f32HelperType {
    typedef
        typename cond_type<is_same_type<T, double>::value, double, float>::type
            type;
};

template <typename T>
struct c32HelperType {
    typedef typename cond_type<is_same_type<T, cfloat>::value, cfloat,
                               typename f32HelperType<T>::type>::type type;
};

template <typename T>
struct elseType {
    typedef typename cond_type<is_same_type<T, uintl>::value ||
                                   is_same_type<T, intl>::value,
                               double, T>::type type;
};

template <typename T>
struct sdOutType {
    typedef typename cond_type<
        is_same_type<T, float>::value || is_same_type<T, int>::value ||
            is_same_type<T, uint>::value || is_same_type<T, uchar>::value ||
            is_same_type<T, short>::value || is_same_type<T, ushort>::value ||
            is_same_type<T, char>::value,
        float, typename elseType<T>::type>::type type;
};

template <typename T>
void stdevDimTest(string pFileName, dim_t dim = -1) {
    typedef typename sdOutType<T>::type outType;
    if (noDoubleTests<T>()) return;
    if (noDoubleTests<outType>()) return;

    vector<dim4> numDims;
    vector<vector<int> > in;
    vector<vector<float> > tests;

    readTestsFromFile<int, float>(pFileName, numDims, in, tests);

    dim4 dims = numDims[0];
    vector<T> input(in[0].begin(), in[0].end());

    array a(dims, &(input.front()));

    array b = stdev(a, dim);

    vector<outType> currGoldBar(tests[0].begin(), tests[0].end());

    size_t nElems = currGoldBar.size();
    vector<outType> outData(nElems);

    b.host((void*)outData.data());

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_NEAR(::real(currGoldBar[elIter]), ::real(outData[elIter]),
                    1.0e-3)
            << "at: " << elIter << endl;
        ASSERT_NEAR(::imag(currGoldBar[elIter]), ::imag(outData[elIter]),
                    1.0e-3)
            << "at: " << elIter << endl;
    }
}

TYPED_TEST(StandardDev, Dim0) {
    stdevDimTest<TypeParam>(string(TEST_DIR "/stdev/mat_10x10_dim0.test"), 0);
}

TYPED_TEST(StandardDev, Dim1) {
    stdevDimTest<TypeParam>(string(TEST_DIR "/stdev/mat_10x10_dim1.test"), 1);
}

TYPED_TEST(StandardDev, Dim2) {
    stdevDimTest<TypeParam>(
        string(TEST_DIR "/stdev/hypercube_10x10x5x5_dim2.test"), 2);
}

TYPED_TEST(StandardDev, Dim3) {
    stdevDimTest<TypeParam>(
        string(TEST_DIR "/stdev/hypercube_10x10x5x5_dim3.test"), 3);
}

TEST(StandardDev, InvalidDim) { ASSERT_THROW(stdev(array(), 5), exception); }

TEST(StandardDev, InvalidType) {
    ASSERT_THROW(stdev(constant(cdouble(1.0, -1.0), 10)), exception);
}

template <typename T>
void stdevDimIndexTest(string pFileName, dim_t dim = -1) {
    typedef typename sdOutType<T>::type outType;
    if (noDoubleTests<T>()) return;
    if (noDoubleTests<outType>()) return;

    vector<dim4> numDims;
    vector<vector<int> > in;
    vector<vector<float> > tests;

    readTestsFromFile<int, float>(pFileName, numDims, in, tests);

    dim4 dims = numDims[0];
    vector<T> input(in[0].begin(), in[0].end());

    array a(dims, &(input.front()));
    array b = a(seq(2, 6), seq(1, 7));

    array c = stdev(b, dim);

    vector<outType> currGoldBar(tests[0].begin(), tests[0].end());

    size_t nElems = currGoldBar.size();
    vector<outType> outData(nElems);

    c.host((void*)outData.data());

    for (size_t elIter = 0; elIter < nElems; ++elIter) {
        ASSERT_NEAR(::real(currGoldBar[elIter]), ::real(outData[elIter]),
                    1.0e-3)
            << "at: " << elIter << endl;
        ASSERT_NEAR(::imag(currGoldBar[elIter]), ::imag(outData[elIter]),
                    1.0e-3)
            << "at: " << elIter << endl;
    }
}

TYPED_TEST(StandardDev, IndexedArrayDim0) {
    stdevDimIndexTest<TypeParam>(
        string(TEST_DIR "/stdev/mat_10x10_seq2_6x1_7_dim0.test"), 0);
}

TYPED_TEST(StandardDev, IndexedArrayDim1) {
    stdevDimIndexTest<TypeParam>(
        string(TEST_DIR "/stdev/mat_10x10_seq2_6x1_7_dim1.test"), 1);
}

TYPED_TEST(StandardDev, All) {
    typedef typename sdOutType<TypeParam>::type outType;
    if (noDoubleTests<TypeParam>()) return;
    if (noDoubleTests<outType>()) return;

    vector<dim4> numDims;
    vector<vector<int> > in;
    vector<vector<float> > tests;

    readTestsFromFile<int, float>(
        string(TEST_DIR "/stdev/mat_10x10_scalar.test"), numDims, in, tests);

    dim4 dims = numDims[0];
    vector<TypeParam> input(in[0].begin(), in[0].end());

    array a(dims, &(input.front()));
    outType b = stdev<outType>(a);

    vector<outType> currGoldBar(tests[0].begin(), tests[0].end());
    ASSERT_NEAR(::real(currGoldBar[0]), ::real(b), 1.0e-3);
    ASSERT_NEAR(::imag(currGoldBar[0]), ::imag(b), 1.0e-3);
}
