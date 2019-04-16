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
#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>

using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;

const int num = 10;

template<typename Ti, typename To>
void cast_test() {
    if (noDoubleTests<Ti>()) return;
    if (noDoubleTests<To>()) return;

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
    if (noDoubleTests<Ti>()) return;
    if (noDoubleTests<To>()) return;

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
        cast_test_complex_real<Ti, To>();               \
    }

COMPLEX_REAL_TESTS(cfloat, float)
COMPLEX_REAL_TESTS(cfloat, double)
COMPLEX_REAL_TESTS(cdouble, float)
COMPLEX_REAL_TESTS(cdouble, double)
