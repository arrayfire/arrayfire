/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <af/array.h>
#include <af/arith.h>
#include <af/data.h>
#include <testHelpers.hpp>

using namespace std;
using namespace af;

const int num = 10;

#define CPLX(TYPE)  af_c##TYPE

#define COMPLEX_TESTS(Ta, Tb, Tc)                                       \
    TEST(ComplexTests, Test_##Ta##_##Tb)                                \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
        if (noDoubleTests<Tc>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        af::array a = randu(num, ta);                                   \
        af::array b = randu(num, tb);                                   \
        af::array c = af::complex(a, b);                                \
        Ta *h_a = a.host<Ta>();                                         \
        Tb *h_b = b.host<Tb>();                                         \
        CPLX(Tc) *h_c = c.host< CPLX(Tc) >();           \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_EQ(h_c[i], CPLX(Tc)(h_a[i], h_b[i])) <<      \
                "for values: " << h_a[i]  << "," << h_b[i] << std::endl; \
        delete[] h_a;                                                   \
        delete[] h_b;                                                   \
        delete[] h_c;                                                   \
    }                                                                   \
    TEST(ComplexTests, Test_cplx_##Ta##_##Tb##_left)                    \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af::array a = randu(num, ta);                                   \
        Tb h_b = 0.3;                                                   \
        af::array c = af::complex(a, h_b);                              \
        Ta *h_a = a.host<Ta>();                                         \
        CPLX(Ta) *h_c = c.host<CPLX(Ta) >();            \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_EQ(h_c[i], CPLX(Ta)(h_a[i], h_b)) <<         \
                "for values: " << h_a[i]  << "," << h_b << std::endl;   \
        delete[] h_a;                                                   \
        delete[] h_c;                                                   \
    }                                                                   \
                                                                        \
    TEST(ComplexTests, Test_cplx_##Ta##_##Tb##_right)                   \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
                                                                        \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        Ta h_a = 0.3;                                                   \
        af::array b = randu(num, tb);                                   \
        af::array c = af::complex(h_a, b);                              \
        Tb *h_b = b.host<Tb>();                                         \
        CPLX(Tb) *h_c = c.host<CPLX(Tb) >();            \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_EQ(h_c[i], CPLX(Tb)(h_a, h_b[i])) <<         \
                "for values: " << h_a  << "," << h_b[i] << std::endl;   \
        delete[] h_b;                                                   \
        delete[] h_c;                                                   \
    }                                                                   \
    TEST(ComplexTests, Test_##Ta##_##Tb##_Real)                         \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
        if (noDoubleTests<Tc>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        af::array a = randu(num, ta);                                   \
        af::array b = randu(num, tb);                                   \
        af::array c = af::complex(a, b);                                \
        af::array d = af::real(c);                                      \
        Ta *h_a = a.host<Ta>();                                         \
        Tc *h_d = d.host<Tc>();                                         \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_EQ(h_d[i], h_a[i]) << "at: " << i << std::endl;      \
        delete[] h_a;                                                   \
        delete[] h_d;                                                   \
    }                                                                   \
    TEST(ComplexTests, Test_##Ta##_##Tb##_Imag)                         \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
        if (noDoubleTests<Tc>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        af::array a = randu(num, ta);                                   \
        af::array b = randu(num, tb);                                   \
        af::array c = af::complex(a, b);                                \
        af::array d = af::imag(c);                                      \
        Tb *h_b = b.host<Tb>();                                         \
        Tc *h_d = d.host<Tc>();                                         \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_EQ(h_d[i], h_b[i])  << "at: " << i << std::endl;     \
        delete[] h_b;                                                   \
        delete[] h_d;                                                   \
    }                                                                   \
    TEST(ComplexTests, Test_##Ta##_##Tb##_Conj)                         \
    {                                                                   \
        if (noDoubleTests<Ta>()) return;                                \
        if (noDoubleTests<Tb>()) return;                                \
        if (noDoubleTests<Tc>()) return;                                \
                                                                        \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;              \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;              \
        af::array a = randu(num, ta);                                   \
        af::array b = randu(num, tb);                                   \
        af::array c = af::complex(a, b);                                \
        af::array d = af::conjg(c);                                     \
        CPLX(Tc) *h_c = c.host<CPLX(Tc) >();            \
        CPLX(Tc) *h_d = d.host<CPLX(Tc) >();            \
        for (int i = 0; i < num; i++)                                   \
            ASSERT_EQ(conj(h_c[i]), h_d[i])                        \
                << "at: " << i << std::endl;                            \
        delete[] h_c;                                                   \
        delete[] h_d;                                                   \
    }                                                                   \


COMPLEX_TESTS(float, float, float)
COMPLEX_TESTS(double, double, double)
COMPLEX_TESTS(float, double, double)
