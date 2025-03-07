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
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using std::string;
using std::vector;

template<typename T>
class Var : public ::testing::Test {};

typedef ::testing::Types<float, double, cfloat, cdouble, uint, int, uintl, intl,
                         char, uchar, short, ushort, half_float::half>
    TestTypes;
TYPED_TEST_SUITE(Var, TestTypes);

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

//////////////////////////////// CPP ////////////////////////////////////
// test var_all interface using cpp api

template<typename T>
void testCPPVar(T const_value, dim4 dims, const bool useDeprecatedAPI = false) {
    typedef typename varOutType<T>::type outType;
    SUPPORTED_TYPE_CHECK(T);
    SUPPORTED_TYPE_CHECK(outType);

    using af::array;
    using af::var;

    vector<T> hundred(dims.elements(), const_value);

    outType gold = outType(0);

    array a(dims, &(hundred.front()));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    outType output =
        (useDeprecatedAPI ? var<outType>(a, false)
                          : var<outType>(a, AF_VARIANCE_POPULATION));

    ASSERT_NEAR(::real(output), ::real(gold), 1.0e-3);
    ASSERT_NEAR(::imag(output), ::imag(gold), 1.0e-3);

    output = (useDeprecatedAPI ? var<outType>(a, true)
                               : var<outType>(a, AF_VARIANCE_SAMPLE));

    ASSERT_NEAR(::real(output), ::real(gold), 1.0e-3);
    ASSERT_NEAR(::imag(output), ::imag(gold), 1.0e-3);

    gold          = outType(2);
    outType tmp[] = {outType(0), outType(1), outType(2), outType(3),
                     outType(4)};
    array b(5, tmp);
    af_print(b);
    output = (useDeprecatedAPI ? var<outType>(b, false)
                               : var<outType>(b, AF_VARIANCE_POPULATION));

    ASSERT_NEAR(::real(output), ::real(gold), 1.0e-3);
    ASSERT_NEAR(::imag(output), ::imag(gold), 1.0e-3);

    gold   = outType(2.5);
    output = (useDeprecatedAPI ? var<outType>(b, true)
                               : var<outType>(b, AF_VARIANCE_SAMPLE));
#pragma GCC diagnostic pop

    ASSERT_NEAR(::real(output), ::real(gold), 1.0e-3);
    ASSERT_NEAR(::imag(output), ::imag(gold), 1.0e-3);
}

TYPED_TEST(Var, AllCPPSmall) {
    testCPPVar<TypeParam>(TypeParam(2), dim4(10, 10, 1, 1));
    testCPPVar<TypeParam>(TypeParam(2), dim4(10, 10, 1, 1), true);
}

TYPED_TEST(Var, AllCPPMedium) {
    testCPPVar<TypeParam>(TypeParam(2), dim4(100, 100, 1, 1));
    testCPPVar<TypeParam>(TypeParam(2), dim4(100, 100, 1, 1), true);
}

TYPED_TEST(Var, AllCPPLarge) {
    testCPPVar<TypeParam>(TypeParam(2), dim4(1000, 1000, 1, 1));
    testCPPVar<TypeParam>(TypeParam(2), dim4(1000, 1000, 1, 1), true);
}

template<typename T>
void dimCppSmallTest(const string pFileName,
                     const bool useDeprecatedAPI = false) {
    typedef typename varOutType<T>::type outType;
    float tol = 0.001f;
    if ((af_dtype)af::dtype_traits<T>::af_type == f16) { tol = 0.6f; }

    SUPPORTED_TYPE_CHECK(T);
    SUPPORTED_TYPE_CHECK(outType);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<outType>> tests;

    readTests<T, outType, float>(pFileName, numDims, in, tests);

    for (size_t i = 0; i < in.size(); i++) {
        array input(numDims[i], &in[i].front(), afHost);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        array bout  = (useDeprecatedAPI ? var(input, true)
                                        : var(input, AF_VARIANCE_SAMPLE));
        array nbout = (useDeprecatedAPI ? var(input, false)
                                        : var(input, AF_VARIANCE_POPULATION));

        array bout1 = (useDeprecatedAPI ? var(input, true, 1)
                                        : var(input, AF_VARIANCE_SAMPLE, 1));
        array nbout1 =
            (useDeprecatedAPI ? var(input, false, 1)
                              : var(input, AF_VARIANCE_POPULATION, 1));
#pragma GCC diagnostic pop

        vector<vector<outType>> h_out(4);

        h_out[0].resize(bout.elements());
        h_out[1].resize(nbout.elements());
        h_out[2].resize(bout1.elements());
        h_out[3].resize(nbout1.elements());

        bout.host(&h_out[0].front());
        nbout.host(&h_out[1].front());
        bout1.host(&h_out[2].front());
        nbout1.host(&h_out[3].front());

        ASSERT_VEC_ARRAY_NEAR(tests[0], bout.dims(), bout, tol);
        ASSERT_VEC_ARRAY_NEAR(tests[1], nbout.dims(), nbout, tol);
        ASSERT_VEC_ARRAY_NEAR(tests[2], bout1.dims(), bout1, tol);
        ASSERT_VEC_ARRAY_NEAR(tests[3], nbout1.dims(), nbout1, tol);
    }
}

TYPED_TEST(Var, DimCPPSmall) {
    dimCppSmallTest<TypeParam>(string(TEST_DIR "/var/var.data"));
    dimCppSmallTest<TypeParam>(string(TEST_DIR "/var/var.data"), true);
}

TEST(Var, ISSUE2117) {
    using af::constant;
    using af::sum;
    using af::var;

    array myArray = constant(1, 1000, 3000);
    myArray       = var(myArray, AF_VARIANCE_SAMPLE, 1);
    ASSERT_NEAR(0.0f, sum<float>(myArray), 0.000001);

    myArray = constant(1, 1000, 3000);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    myArray = var(myArray, true, 1);
#pragma GCC diagnostic pop
    ASSERT_NEAR(0.0f, sum<float>(myArray), 0.000001);
}
