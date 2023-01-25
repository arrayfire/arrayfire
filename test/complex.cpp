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
#include <af/device.h>
#include <af/random.h>

using std::endl;
using namespace af;

const int num = 10;

#define CPLX(TYPE) af_c##TYPE

#define COMPLEX_TESTS(Ta, Tb, Tc)                                     \
    TEST(ComplexTests, Test_##Ta##_##Tb) {                            \
        SUPPORTED_TYPE_CHECK(Ta);                                     \
        SUPPORTED_TYPE_CHECK(Tb);                                     \
        SUPPORTED_TYPE_CHECK(Tc);                                     \
                                                                      \
        af_dtype ta   = (af_dtype)dtype_traits<Ta>::af_type;          \
        af_dtype tb   = (af_dtype)dtype_traits<Tb>::af_type;          \
        array a       = randu(num, ta);                               \
        array b       = randu(num, tb);                               \
        array c       = complex(a, b);                                \
        Ta *h_a       = a.host<Ta>();                                 \
        Tb *h_b       = b.host<Tb>();                                 \
        CPLX(Tc) *h_c = c.host<CPLX(Tc)>();                           \
        for (int i = 0; i < num; i++)                                 \
            ASSERT_EQ(h_c[i], CPLX(Tc)(h_a[i], h_b[i]))               \
                << "for values: " << h_a[i] << "," << h_b[i] << endl; \
        freeHost(h_a);                                                \
        freeHost(h_b);                                                \
        freeHost(h_c);                                                \
    }                                                                 \
    TEST(ComplexTests, Test_cplx_##Ta##_##Tb##_left) {                \
        SUPPORTED_TYPE_CHECK(Ta);                                     \
        SUPPORTED_TYPE_CHECK(Tb);                                     \
                                                                      \
        af_dtype ta   = (af_dtype)dtype_traits<Ta>::af_type;          \
        array a       = randu(num, ta);                               \
        Tb h_b        = 0.3;                                          \
        array c       = complex(a, h_b);                              \
        Ta *h_a       = a.host<Ta>();                                 \
        CPLX(Ta) *h_c = c.host<CPLX(Ta)>();                           \
        for (int i = 0; i < num; i++)                                 \
            ASSERT_EQ(h_c[i], CPLX(Ta)(h_a[i], h_b))                  \
                << "for values: " << h_a[i] << "," << h_b << endl;    \
        freeHost(h_a);                                                \
        freeHost(h_c);                                                \
    }                                                                 \
                                                                      \
    TEST(ComplexTests, Test_cplx_##Ta##_##Tb##_right) {               \
        SUPPORTED_TYPE_CHECK(Ta);                                     \
        SUPPORTED_TYPE_CHECK(Tb);                                     \
                                                                      \
        af_dtype tb   = (af_dtype)dtype_traits<Tb>::af_type;          \
        Ta h_a        = 0.3;                                          \
        array b       = randu(num, tb);                               \
        array c       = complex(h_a, b);                              \
        Tb *h_b       = b.host<Tb>();                                 \
        CPLX(Tb) *h_c = c.host<CPLX(Tb)>();                           \
        for (int i = 0; i < num; i++)                                 \
            ASSERT_EQ(h_c[i], CPLX(Tb)(h_a, h_b[i]))                  \
                << "for values: " << h_a << "," << h_b[i] << endl;    \
        freeHost(h_b);                                                \
        freeHost(h_c);                                                \
    }                                                                 \
    TEST(ComplexTests, Test_##Ta##_##Tb##_Real) {                     \
        SUPPORTED_TYPE_CHECK(Ta);                                     \
        SUPPORTED_TYPE_CHECK(Tb);                                     \
        SUPPORTED_TYPE_CHECK(Tc);                                     \
                                                                      \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;            \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;            \
        array a     = randu(num, ta);                                 \
        array b     = randu(num, tb);                                 \
        array c     = complex(a, b);                                  \
        array d     = real(c);                                        \
        Ta *h_a     = a.host<Ta>();                                   \
        Tc *h_d     = d.host<Tc>();                                   \
        for (int i = 0; i < num; i++)                                 \
            ASSERT_EQ(h_d[i], h_a[i]) << "at: " << i << endl;         \
        freeHost(h_a);                                                \
        freeHost(h_d);                                                \
    }                                                                 \
    TEST(ComplexTests, Test_##Ta##_##Tb##_Imag) {                     \
        SUPPORTED_TYPE_CHECK(Ta);                                     \
        SUPPORTED_TYPE_CHECK(Tb);                                     \
        SUPPORTED_TYPE_CHECK(Tc);                                     \
                                                                      \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;            \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;            \
        array a     = randu(num, ta);                                 \
        array b     = randu(num, tb);                                 \
        array c     = complex(a, b);                                  \
        array d     = imag(c);                                        \
        Tb *h_b     = b.host<Tb>();                                   \
        Tc *h_d     = d.host<Tc>();                                   \
        for (int i = 0; i < num; i++)                                 \
            ASSERT_EQ(h_d[i], h_b[i]) << "at: " << i << endl;         \
        freeHost(h_b);                                                \
        freeHost(h_d);                                                \
    }                                                                 \
    TEST(ComplexTests, Test_##Ta##_##Tb##_Conj) {                     \
        SUPPORTED_TYPE_CHECK(Ta);                                     \
        SUPPORTED_TYPE_CHECK(Tb);                                     \
        SUPPORTED_TYPE_CHECK(Tc);                                     \
                                                                      \
        af_dtype ta   = (af_dtype)dtype_traits<Ta>::af_type;          \
        af_dtype tb   = (af_dtype)dtype_traits<Tb>::af_type;          \
        array a       = randu(num, ta);                               \
        array b       = randu(num, tb);                               \
        array c       = complex(a, b);                                \
        array d       = conjg(c);                                     \
        CPLX(Tc) *h_c = c.host<CPLX(Tc)>();                           \
        CPLX(Tc) *h_d = d.host<CPLX(Tc)>();                           \
        for (int i = 0; i < num; i++)                                 \
            ASSERT_EQ(conj(h_c[i]), h_d[i]) << "at: " << i << endl;   \
        freeHost(h_c);                                                \
        freeHost(h_d);                                                \
    }

COMPLEX_TESTS(float, float, float)
COMPLEX_TESTS(double, double, double)
COMPLEX_TESTS(float, double, double)

TEST(Complex, SNIPPET_arith_func_complex) {
    //! [ex_arith_func_complex]
    //!
    // Create a, a 2x3 array
    array a = iota(dim4(2, 3));  // a = [0, 2, 4,
                                 //      1, 3, 5]

    // Create b from a single real array, returning zeros for the imaginary
    // component
    array b = complex(a);  // b = [(0, 0), (2, 0), (4, 0),
                           //      (1, 0), (3, 0), (5, 0)]

    // Create c from two real arrays, one for the real component and one for the
    // imaginary component
    array c = complex(a, a);  // c = [(0, 0), (2, 2), (4, 4),
                              //      (1, 1), (3, 3), (5, 5)]

    // Create d from a single real array for the real component and a single
    // scalar for each imaginary component
    array d = complex(a, 2);  // d = [(0, 2), (2, 2), (4, 2),
                              //      (1, 2), (3, 2), (5, 2)]

    // Create e from a single scalar for each real component and a single real
    // array for the imaginary component
    array e = complex(2, a);  // e = [(2, 0), (2, 2), (2, 4),
                              //      (2, 1), (2, 3), (2, 5)]

    //! [ex_arith_func_complex]

    using std::complex;
    using std::vector;
    vector<float> ha(a.elements());
    a.host(ha.data());

    vector<cfloat> gold_b(a.elements());
    for (int i = 0; i < a.elements(); i++) {
        gold_b[i].real = ha[i];
        gold_b[i].imag = 0;
    }
    ASSERT_VEC_ARRAY_EQ(gold_b, a.dims(), b);

    vector<cfloat> gold_c(a.elements());
    for (int i = 0; i < a.elements(); i++) {
        gold_c[i].real = ha[i];
        gold_c[i].imag = ha[i];
    }
    ASSERT_VEC_ARRAY_EQ(gold_c, a.dims(), c);

    vector<cfloat> gold_d(a.elements());
    for (int i = 0; i < a.elements(); i++) {
        gold_d[i].real = ha[i];
        gold_d[i].imag = 2;
    }
    ASSERT_VEC_ARRAY_EQ(gold_d, a.dims(), d);

    vector<cfloat> gold_e(a.elements());
    for (int i = 0; i < a.elements(); i++) {
        gold_e[i].real = 2;
        gold_e[i].imag = ha[i];
    }
    ASSERT_VEC_ARRAY_EQ(gold_e, a.dims(), e);
}