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
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <algorithm>
#include <ctime>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::constant;
using af::dim4;
using af::exception;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Covariance : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, intl, uintl, uchar, short,
                         ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Covariance, TestTypes);

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
struct covOutType {
    typedef typename cond_type<
        is_same_type<T, float>::value || is_same_type<T, int>::value ||
            is_same_type<T, uint>::value || is_same_type<T, uchar>::value ||
            is_same_type<T, short>::value || is_same_type<T, ushort>::value ||
            is_same_type<T, char>::value,
        float, typename elseType<T>::type>::type type;
};

template<typename T>
void covTest(string pFileName, bool isbiased = true,
             const bool useDeprecatedAPI = false) {
    typedef typename covOutType<T>::type outType;
    SUPPORTED_TYPE_CHECK(T);
    SUPPORTED_TYPE_CHECK(outType);

    vector<dim4> numDims;
    vector<vector<int>> in;
    vector<vector<float>> tests;

    readTestsFromFile<int, float>(pFileName, numDims, in, tests);

    dim4 dims1 = numDims[0];
    dim4 dims2 = numDims[1];
    vector<T> input1(in[0].begin(), in[0].end());
    vector<T> input2(in[1].begin(), in[1].end());

    array a(dims1, &(input1.front()));
    array b(dims2, &(input2.front()));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    array c =
        (useDeprecatedAPI
             ? cov(a, b, isbiased)
             : cov(a, b,
                   (isbiased ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION)));
#pragma GCC diagnostic pop

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

TYPED_TEST(Covariance, Vector) {
    covTest<TypeParam>(string(TEST_DIR "/covariance/vec_size60.test"));
    covTest<TypeParam>(string(TEST_DIR "/covariance/vec_size60.test"), true);
}

TYPED_TEST(Covariance, Matrix) {
    covTest<TypeParam>(string(TEST_DIR "/covariance/matrix_65x121.test"));
    covTest<TypeParam>(string(TEST_DIR "/covariance/matrix_65x121.test"), true);
}

TEST(Covariance, c32) {
    array a = constant(cfloat(1.0f, -1.0f), 10, c32);
    array b = constant(cfloat(2.0f, -1.0f), 10, c32);
    ASSERT_THROW(cov(a, b, AF_VARIANCE_POPULATION), exception);
}

TEST(Covariance, c64) {
    SUPPORTED_TYPE_CHECK(double);
    array a = constant(cdouble(1.0, -1.0), 10, c64);
    array b = constant(cdouble(2.0, -1.0), 10, c64);
    ASSERT_THROW(cov(a, b, AF_VARIANCE_POPULATION), exception);
}
