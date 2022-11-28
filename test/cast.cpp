/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/algorithm.h>
#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>
#include <af/random.h>
#include <algorithm>
#include <cstdlib>
#include <vector>

using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;

const int num = 10;

template<typename Ti, typename To>
void cast_test() {
    SUPPORTED_TYPE_CHECK(Ti);
    SUPPORTED_TYPE_CHECK(To);

    af_dtype ta = (af_dtype)dtype_traits<Ti>::af_type;
    af_dtype tb = (af_dtype)dtype_traits<To>::af_type;
    dim4 dims(num, 1, 1, 1);
    af_array a, b;
    af_randu(&a, dims.ndims(), dims.get(), ta);
    af_err err = af_cast(&b, a, tb);
    af_release_array(a);
    af_release_array(b);
    ASSERT_SUCCESS(err);
}

#define REAL_TO_TESTS(Ti, To) \
    TEST(CAST_TEST, Test_Real_##Ti##_##To) { cast_test<Ti, To>(); }

#define REAL_TEST_INVOKE(Ti)     \
    REAL_TO_TESTS(Ti, float);    \
    REAL_TO_TESTS(Ti, cfloat);   \
    REAL_TO_TESTS(Ti, double);   \
    REAL_TO_TESTS(Ti, cdouble);  \
    REAL_TO_TESTS(Ti, char);     \
    REAL_TO_TESTS(Ti, int);      \
    REAL_TO_TESTS(Ti, unsigned); \
    REAL_TO_TESTS(Ti, uchar);    \
    REAL_TO_TESTS(Ti, intl);     \
    REAL_TO_TESTS(Ti, uintl);    \
    REAL_TO_TESTS(Ti, short);    \
    REAL_TO_TESTS(Ti, ushort);

#define CPLX_TEST_INVOKE(Ti)   \
    REAL_TO_TESTS(Ti, cfloat); \
    REAL_TO_TESTS(Ti, cdouble);

REAL_TEST_INVOKE(float)
REAL_TEST_INVOKE(double)
REAL_TEST_INVOKE(char)
REAL_TEST_INVOKE(int)
REAL_TEST_INVOKE(unsigned)
REAL_TEST_INVOKE(uchar)
REAL_TEST_INVOKE(intl)
REAL_TEST_INVOKE(uintl)
REAL_TEST_INVOKE(short)
REAL_TEST_INVOKE(ushort)
CPLX_TEST_INVOKE(cfloat)
CPLX_TEST_INVOKE(cdouble)

// Converting complex to real; expected to fail as this operation is
// not allowed. Use functions abs, real, image, arg, etc to make the
// conversion explicit.
template<typename Ti, typename To>
void cast_test_complex_real() {
    SUPPORTED_TYPE_CHECK(Ti);
    SUPPORTED_TYPE_CHECK(To);

    af_dtype ta = (af_dtype)dtype_traits<Ti>::af_type;
    af_dtype tb = (af_dtype)dtype_traits<To>::af_type;
    dim4 dims(num, 1, 1, 1);
    af_array a, b;
    af_randu(&a, dims.ndims(), dims.get(), ta);
    af_err err = af_cast(&b, a, tb);
    ASSERT_EQ(err, AF_ERR_TYPE);
    ASSERT_SUCCESS(af_release_array(a));
}

#define COMPLEX_REAL_TESTS(Ti, To)                      \
    TEST(CAST_TEST, Test_Complex_To_Real_##Ti##_##To) { \
        SUPPORTED_TYPE_CHECK(Ti);                       \
        SUPPORTED_TYPE_CHECK(To);                       \
        cast_test_complex_real<Ti, To>();               \
    }

COMPLEX_REAL_TESTS(cfloat, float)
COMPLEX_REAL_TESTS(cfloat, double)
COMPLEX_REAL_TESTS(cdouble, float)
COMPLEX_REAL_TESTS(cdouble, double)

TEST(CAST_TEST, Test_JIT_DuplicateCastNoop) {
    // Does a trivial cast - check JIT kernel trace to ensure a __noop is
    // generated since we don't have a way to test it directly
    SUPPORTED_TYPE_CHECK(double);
    af_dtype ta = (af_dtype)dtype_traits<float>::af_type;
    af_dtype tb = (af_dtype)dtype_traits<double>::af_type;
    dim4 dims(num, 1, 1, 1);
    af_array a, b, c;
    af_randu(&a, dims.ndims(), dims.get(), ta);

    af_cast(&b, a, tb);
    af_cast(&c, b, ta);

    std::vector<float> a_vals(num);
    std::vector<float> c_vals(num);
    ASSERT_SUCCESS(af_get_data_ptr((void **)&a_vals[0], a));
    ASSERT_SUCCESS(af_get_data_ptr((void **)&c_vals[0], c));

    for (size_t i = 0; i < num; ++i) { ASSERT_FLOAT_EQ(a_vals[i], c_vals[i]); }

    af_release_array(a);
    af_release_array(b);
    af_release_array(c);
}

TEST(Cast, ImplicitCast) {
    using namespace af;
    SUPPORTED_TYPE_CHECK(double);
    array a = randu(100, 100, f64);
    array b = a.as(f32);

    array c = max(abs(a - b));
    ASSERT_ARRAYS_NEAR(constant(0, 1, 100, f64), c, 1e-7);
}

TEST(Cast, ConstantCast) {
    using namespace af;
    SUPPORTED_TYPE_CHECK(double);
    array a = constant(1, 100, f64);
    array b = a.as(f32);

    array c = max(abs(a - b));
    ASSERT_ARRAYS_NEAR(c, constant(0, 1, f64), 1e-7);
}

TEST(Cast, OpCast) {
    using namespace af;
    SUPPORTED_TYPE_CHECK(double);
    array a = constant(1, 100, f64);
    a       = a + a;
    array b = a.as(f32);

    array c = max(abs(a - b));
    ASSERT_ARRAYS_NEAR(c, constant(0, 1, f64), 1e-7);
}
TEST(Cast, ImplicitCastIndexed) {
    using namespace af;
    SUPPORTED_TYPE_CHECK(double);
    array a = randu(100, 100, f64);
    array b = a(span, 1).as(f32);
    array c = max(abs(a(span, 1) - b));
    ASSERT_ARRAYS_NEAR(constant(0, 1, 1, f64), c, 1e-7);
}

TEST(Cast, ImplicitCastIndexedNonLinear) {
    using namespace af;
    SUPPORTED_TYPE_CHECK(double);
    array a = randu(100, 100, f64);
    array b = a(seq(10, 20, 2), 1).as(f32);
    array c = max(abs(a(seq(10, 20, 2), 1) - b));
    ASSERT_ARRAYS_NEAR(constant(0, 1, 1, f64), c, 1e-7);
}

TEST(Cast, ImplicitCastIndexedNonLinearArray) {
    using namespace af;
    SUPPORTED_TYPE_CHECK(double);
    array a   = randu(100, 100, f64);
    array idx = seq(10, 20, 2);
    array b   = a(idx, 1).as(f32);
    array c   = max(abs(a(idx, 1) - b));
    ASSERT_ARRAYS_NEAR(constant(0, 1, 1, f64), c, 1e-7);
}

TEST(Cast, ImplicitCastIndexedAndScoped) {
    using namespace af;
    SUPPORTED_TYPE_CHECK(double);
    array c;
    {
        array a = randu(100, 100, f64);
        array b = a(span, 1).as(f32);
        c       = abs(a(span, 1) - b);
    }
    c = max(c);
    ASSERT_ARRAYS_NEAR(constant(0, 1, 1, f64), c, 1e-7);
}
