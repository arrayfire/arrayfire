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
#include <complex>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dot;
using af::dtype_traits;
using std::abs;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class DotF : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

template<typename T>
class DotC : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

typedef ::testing::Types<float, double> TestTypesF;
typedef ::testing::Types<cfloat, cdouble> TestTypesC;

// register the type list
TYPED_TEST_SUITE(DotF, TestTypesF);
TYPED_TEST_SUITE(DotC, TestTypesC);

bool isinf(af::af_cfloat val) {
    using std::isinf;
    return isinf(val.real) || isinf(val.imag);
}
bool isinf(af::af_cdouble val) {
    using std::isinf;
    return isinf(val.real) || isinf(val.imag);
}

template<typename T>
void dotTest(string pTestFile, const int resultIdx,
             const af_mat_prop optLhs = AF_MAT_NONE,
             const af_mat_prop optRhs = AF_MAT_NONE) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;

    readTests<T, T, T>(pTestFile, numDims, in, tests);

    dim4 aDims = numDims[0];
    dim4 bDims = numDims[1];

    af_array a   = 0;
    af_array b   = 0;
    af_array out = 0;

    ASSERT_SUCCESS(af_create_array(&a, &(in[0].front()), aDims.ndims(),
                                   aDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));
    ASSERT_SUCCESS(af_create_array(&b, &(in[1].front()), bDims.ndims(),
                                   bDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SUCCESS(af_dot(&out, a, b, optLhs, optRhs));

    vector<T> goldData = tests[resultIdx];
    size_t nElems      = goldData.size();

    ASSERT_VEC_ARRAY_NEAR(goldData, dim4(nElems), out, 0.03);

    ASSERT_SUCCESS(af_release_array(a));
    ASSERT_SUCCESS(af_release_array(b));
    ASSERT_SUCCESS(af_release_array(out));
}

template<typename T>
void compare(double rval, double /*ival*/, T gold) {
    ASSERT_NEAR(gold, rval, 0.03);
}

template<>
void compare<cfloat>(double rval, double ival, cfloat gold) {
    ASSERT_NEAR(gold.real, rval, 0.03);
    ASSERT_NEAR(gold.imag, ival, 0.03);
}

template<>
void compare<cdouble>(double rval, double ival, cdouble gold) {
    ASSERT_NEAR(gold.real, rval, 0.03);
    ASSERT_NEAR(gold.imag, ival, 0.03);
}

template<typename T>
void dotAllTest(string pTestFile, const int resultIdx,
                const af_mat_prop optLhs = AF_MAT_NONE,
                const af_mat_prop optRhs = AF_MAT_NONE) {
    SUPPORTED_TYPE_CHECK(T);

    vector<dim4> numDims;
    vector<vector<T>> in;
    vector<vector<T>> tests;

    readTests<T, T, T>(pTestFile, numDims, in, tests);

    dim4 aDims = numDims[0];
    dim4 bDims = numDims[1];

    af_array a = 0;
    af_array b = 0;

    ASSERT_SUCCESS(af_create_array(&a, &(in[0].front()), aDims.ndims(),
                                   aDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));
    ASSERT_SUCCESS(af_create_array(&b, &(in[1].front()), bDims.ndims(),
                                   bDims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    double rval = 0, ival = 0;
    ASSERT_SUCCESS(af_dot_all(&rval, &ival, a, b, optLhs, optRhs));

    vector<T> goldData = tests[resultIdx];

    using ::isinf;
    using std::isinf;
    if (false == (isinf(rval) && isinf(goldData[0]))) {
        compare<T>(rval, ival, goldData[0]);
    }

    ASSERT_SUCCESS(af_release_array(a));
    ASSERT_SUCCESS(af_release_array(b));
}

#define INSTANTIATEF(SIZE, FILENAME)                                           \
    TYPED_TEST(DotF, DotF_##SIZE) {                                            \
        dotTest<TypeParam>(string(TEST_DIR "/blas/" #FILENAME ".test"), 0);    \
        dotAllTest<TypeParam>(string(TEST_DIR "/blas/" #FILENAME ".test"), 0); \
    }

#define INSTANTIATEC(SIZE, FILENAME)                                          \
    TYPED_TEST(DotC, DotC_CC_##SIZE) {                                        \
        dotTest<TypeParam>(string(TEST_DIR "/blas/" #FILENAME ".test"), 0,    \
                           AF_MAT_CONJ, AF_MAT_CONJ);                         \
        dotAllTest<TypeParam>(string(TEST_DIR "/blas/" #FILENAME ".test"), 0, \
                              AF_MAT_CONJ, AF_MAT_CONJ);                      \
    }                                                                         \
    TYPED_TEST(DotC, DotC_UU_##SIZE) {                                        \
        dotTest<TypeParam>(string(TEST_DIR "/blas/" #FILENAME ".test"), 1,    \
                           AF_MAT_NONE, AF_MAT_NONE);                         \
        dotAllTest<TypeParam>(string(TEST_DIR "/blas/" #FILENAME ".test"), 1, \
                              AF_MAT_NONE, AF_MAT_NONE);                      \
    }                                                                         \
    TYPED_TEST(DotC, DotC_CU_##SIZE) {                                        \
        dotTest<TypeParam>(string(TEST_DIR "/blas/" #FILENAME ".test"), 2,    \
                           AF_MAT_CONJ, AF_MAT_NONE);                         \
        dotAllTest<TypeParam>(string(TEST_DIR "/blas/" #FILENAME ".test"), 2, \
                              AF_MAT_CONJ, AF_MAT_NONE);                      \
    }                                                                         \
    TYPED_TEST(DotC, DotC_UC_##SIZE) {                                        \
        dotTest<TypeParam>(string(TEST_DIR "/blas/" #FILENAME ".test"), 3,    \
                           AF_MAT_NONE, AF_MAT_CONJ);                         \
        dotAllTest<TypeParam>(string(TEST_DIR "/blas/" #FILENAME ".test"), 3, \
                              AF_MAT_NONE, AF_MAT_CONJ);                      \
    }

INSTANTIATEF(1000, dot_f_1000);
INSTANTIATEF(10, dot_f_10);
INSTANTIATEF(25600, dot_f_25600);
INSTANTIATEC(1000, dot_c_1000);
INSTANTIATEC(10, dot_c_10);
INSTANTIATEC(25600, dot_c_25600);

///////////////////////////////////// CPP ////////////////////////////////
//
TEST(DotF, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTests<float, float, float>(TEST_DIR "/blas/dot_f_1000.test", numDims,
                                   in, tests);

    dim4 aDims = numDims[0];
    dim4 bDims = numDims[1];

    array a(aDims, &(in[0].front()));
    array b(bDims, &(in[1].front()));

    array out = dot(a, b, AF_MAT_CONJ, AF_MAT_NONE);

    vector<float> goldData = tests[0];
    dim4 goldDims(1);
    ASSERT_VEC_ARRAY_EQ(goldData, goldDims, out);
}

TEST(DotCCU, CPP) {
    vector<dim4> numDims;
    vector<vector<cfloat>> in;
    vector<vector<cfloat>> tests;

    readTests<cfloat, cfloat, cfloat>(TEST_DIR "/blas/dot_c_1000.test", numDims,
                                      in, tests);

    dim4 aDims = numDims[0];
    dim4 bDims = numDims[1];

    array a(aDims, &(in[0].front()));
    array b(bDims, &(in[1].front()));

    array out = dot(a, b, AF_MAT_CONJ, AF_MAT_NONE);

    vector<cfloat> goldData = tests[2];
    dim4 goldDims(1);
    ASSERT_VEC_ARRAY_EQ(goldData, goldDims, out);
}

TEST(DotAllF, CPP) {
    vector<dim4> numDims;
    vector<vector<float>> in;
    vector<vector<float>> tests;

    readTests<float, float, float>(TEST_DIR "/blas/dot_f_1000.test", numDims,
                                   in, tests);

    dim4 aDims = numDims[0];
    dim4 bDims = numDims[1];

    array a(aDims, &(in[0].front()));
    array b(bDims, &(in[1].front()));

    float out = dot<float>(a, b, AF_MAT_CONJ, AF_MAT_NONE);

    vector<float> goldData = tests[0];

    ASSERT_EQ(goldData[0], out);
}

TEST(DotAllCCU, CPP) {
    vector<dim4> numDims;
    vector<vector<cfloat>> in;
    vector<vector<cfloat>> tests;

    readTests<cfloat, cfloat, cfloat>(TEST_DIR "/blas/dot_c_1000.test", numDims,
                                      in, tests);

    dim4 aDims = numDims[0];
    dim4 bDims = numDims[1];

    array a(aDims, &(in[0].front()));
    array b(bDims, &(in[1].front()));

    cfloat out = dot<cfloat>(a, b, AF_MAT_CONJ, AF_MAT_NONE);

    vector<cfloat> goldData = tests[2];

    ASSERT_EQ(goldData[0], out);
}

class Dot : public ::testing::TestWithParam<int> {
   public:
    array ha, hb, gold;

    void SetUp() {
        SUPPORTED_TYPE_CHECK(half_float::half);
        int elems = GetParam();
        array fa  = af::randu(elems) - 0.5f;
        array fb  = af::randu(elems) - 0.5f;

        ha = fa.as(f16);
        hb = fb.as(f16);

        gold = dot(fa, fb);
    }
};

std::string print_dot(const ::testing::TestParamInfo<Dot::ParamType> info) {
    std::stringstream ss;

    ss << info.param;

    return ss.str();
}

INSTANTIATE_TEST_SUITE_P(Small, Dot,
                         ::testing::Values(2, 4, 5, 10, 31, 32, 33, 100, 127,
                                           128, 129, 200, 500, 511, 512, 513,
                                           1000),
                         print_dot);

TEST_P(Dot, Half) {
    SUPPORTED_TYPE_CHECK(half_float::half);
    array hc = dot(ha, hb);

    ASSERT_ARRAYS_NEAR(gold, hc.as(f32), 1e-2);
}
