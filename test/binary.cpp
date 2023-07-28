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

#include <cfenv>
#include <cmath>

using namespace std;
using namespace af;

const int num = 10000;

#define add(left, right) (left) + (right)
#define sub(left, right) (left) - (right)
#define mul(left, right) (left) * (right)
#define div(left, right) (left) / (right)

typedef std::complex<float> complex_float;
typedef std::complex<double> complex_double;

template<typename T>
T mod(T a, T b) {
    return std::fmod(a, b);
}

af::array randgen(const int num, dtype ty) {
    af::array tmp = round(1 + 2 * af::randu(num, f32)).as(ty);
    tmp.eval();
    return tmp;
}

#define MY_ASSERT_NEAR(aa, bb, cc) ASSERT_NEAR(abs(aa), abs(bb), (cc))

#define BINARY_TESTS(Ta, Tb, Tc, func)                                    \
    TEST(BinaryTests, Test_##func##_##Ta##_##Tb) {                        \
        SUPPORTED_TYPE_CHECK(Ta);                                         \
        SUPPORTED_TYPE_CHECK(Tb);                                         \
        SUPPORTED_TYPE_CHECK(Tc);                                         \
                                                                          \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;                \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;                \
        af::array a = randgen(num, ta);                                   \
        af::array b = randgen(num, tb);                                   \
        af::array c = func(a, b);                                         \
        Ta *h_a     = a.host<Ta>();                                       \
        Tb *h_b     = b.host<Tb>();                                       \
        vector<Tc> gold(num);                                             \
        for (int i = 0; i < num; i++) { gold[i] = func(h_a[i], h_b[i]); } \
        ASSERT_VEC_ARRAY_EQ(gold, dim4(num), c);                          \
        af_free_host(h_a);                                                \
        af_free_host(h_b);                                                \
    }                                                                     \
                                                                          \
    TEST(BinaryTests, Test_##func##_##Ta##_##Tb##_left) {                 \
        SUPPORTED_TYPE_CHECK(Ta);                                         \
        SUPPORTED_TYPE_CHECK(Tb);                                         \
                                                                          \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;                \
        af::array a = randgen(num, ta);                                   \
        Tb h_b      = 3.0;                                                \
        af::array c = func(a, h_b);                                       \
        Ta *h_a     = a.host<Ta>();                                       \
        vector<Tc> gold(num);                                             \
        for (int i = 0; i < num; i++) { gold[i] = func(h_a[i], h_b); }    \
        ASSERT_VEC_ARRAY_EQ(gold, dim4(num), c);                          \
        af_free_host(h_a);                                                \
    }                                                                     \
                                                                          \
    TEST(BinaryTests, Test_##func##_##Ta##_##Tb##_right) {                \
        SUPPORTED_TYPE_CHECK(Ta);                                         \
        SUPPORTED_TYPE_CHECK(Tb);                                         \
                                                                          \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;                \
        Ta h_a      = 5.0;                                                \
        af::array b = randgen(num, tb);                                   \
        af::array c = func(h_a, b);                                       \
        Tb *h_b     = b.host<Tb>();                                       \
        vector<Tc> gold(num);                                             \
        for (int i = 0; i < num; i++) { gold[i] = func(h_a, h_b[i]); }    \
        ASSERT_VEC_ARRAY_EQ(gold, dim4(num), c);                          \
        af_free_host(h_b);                                                \
    }

#define BINARY_TESTS_NEAR_GENERAL(Ta, Tb, Tc, Td, Te, func, err)      \
    TEST(BinaryTestsFloating, Test_##func##_##Ta##_##Tb) {            \
        SUPPORTED_TYPE_CHECK(Ta);                                     \
        SUPPORTED_TYPE_CHECK(Tb);                                     \
        SUPPORTED_TYPE_CHECK(Tc);                                     \
                                                                      \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;            \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;            \
        af::array a = randgen(num, ta);                               \
        af::array b = randgen(num, tb);                               \
        af::array c = func(a, b);                                     \
        Ta *h_a     = a.host<Ta>();                                   \
        Tb *h_b     = b.host<Tb>();                                   \
        Tc *h_c     = c.host<Tc>();                                   \
        for (int i = 0; i < num; i++)                                 \
            MY_ASSERT_NEAR(h_c[i], func(h_a[i], h_b[i]), (err))       \
                << "for values: " << h_a[i] << "," << h_b[i] << endl; \
        af_free_host(h_a);                                            \
        af_free_host(h_b);                                            \
        af_free_host(h_c);                                            \
    }                                                                 \
                                                                      \
    TEST(BinaryTestsFloating, Test_##func##_##Ta##_##Tb##_left) {     \
        SUPPORTED_TYPE_CHECK(Ta);                                     \
        SUPPORTED_TYPE_CHECK(Tb);                                     \
                                                                      \
        af_dtype ta = (af_dtype)dtype_traits<Ta>::af_type;            \
        af::array a = randgen(num, ta);                               \
        Tb h_b      = 0.3;                                            \
        af::array c = func(a, h_b);                                   \
        Ta *h_a     = a.host<Ta>();                                   \
        Td *h_d     = c.host<Td>();                                   \
        for (int i = 0; i < num; i++)                                 \
            MY_ASSERT_NEAR(h_d[i], func(h_a[i], h_b), err)            \
                << "for values: " << h_a[i] << "," << h_b << endl;    \
        af_free_host(h_a);                                            \
        af_free_host(h_d);                                            \
    }                                                                 \
                                                                      \
    TEST(BinaryTestsFloating, Test_##func##_##Ta##_##Tb##_right) {    \
        SUPPORTED_TYPE_CHECK(Ta);                                     \
        SUPPORTED_TYPE_CHECK(Tb);                                     \
        SUPPORTED_TYPE_CHECK(Tc);                                     \
                                                                      \
        af_dtype tb = (af_dtype)dtype_traits<Tb>::af_type;            \
        Ta h_a      = 0.3;                                            \
        af::array b = randgen(num, tb);                               \
        af::array c = func(h_a, b);                                   \
        Tb *h_b     = b.host<Tb>();                                   \
        Te *h_e     = c.host<Te>();                                   \
        for (int i = 0; i < num; i++)                                 \
            MY_ASSERT_NEAR(h_e[i], func(h_a, h_b[i]), err)            \
                << "for values: " << h_a << "," << h_b[i] << endl;    \
        af_free_host(h_b);                                            \
        af_free_host(h_e);                                            \
    }

#define BINARY_TESTS_NEAR(Ta, Tb, Tc, func, err) \
    BINARY_TESTS_NEAR_GENERAL(Ta, Tb, Tc, Ta, Tc, func, err)

#define BINARY_TESTS_FLOAT(func) BINARY_TESTS(float, float, float, func)
#define BINARY_TESTS_DOUBLE(func) BINARY_TESTS(double, double, double, func)
#define BINARY_TESTS_CFLOAT(func) BINARY_TESTS(cfloat, cfloat, cfloat, func)
#define BINARY_TESTS_CDOUBLE(func) BINARY_TESTS(cdouble, cdouble, cdouble, func)

#define BINARY_TESTS_INT(func) BINARY_TESTS(int, int, int, func)
#define BINARY_TESTS_UINT(func) BINARY_TESTS(uint, uint, uint, func)
#define BINARY_TESTS_INTL(func) BINARY_TESTS(intl, intl, intl, func)
#define BINARY_TESTS_UINTL(func) BINARY_TESTS(uintl, uintl, uintl, func)
#define BINARY_TESTS_NEAR_FLOAT(func) \
    BINARY_TESTS_NEAR(float, float, float, func, 1e-5)
#define BINARY_TESTS_NEAR_DOUBLE(func) \
    BINARY_TESTS_NEAR(double, double, double, func, 1e-10)

BINARY_TESTS_FLOAT(add)
BINARY_TESTS_FLOAT(sub)
BINARY_TESTS_FLOAT(mul)
BINARY_TESTS_NEAR(float, float, float, div, 1e-3)  // FIXME
BINARY_TESTS_FLOAT(min)
BINARY_TESTS_FLOAT(max)
BINARY_TESTS_NEAR(float, float, float, mod, 1e-5)  // FIXME

BINARY_TESTS_DOUBLE(add)
BINARY_TESTS_DOUBLE(sub)
BINARY_TESTS_DOUBLE(mul)
BINARY_TESTS_DOUBLE(div)
BINARY_TESTS_DOUBLE(min)
BINARY_TESTS_DOUBLE(max)
BINARY_TESTS_DOUBLE(mod)

BINARY_TESTS_NEAR_FLOAT(atan2)
BINARY_TESTS_NEAR_FLOAT(pow)
BINARY_TESTS_NEAR_FLOAT(hypot)

BINARY_TESTS_NEAR_DOUBLE(atan2)
BINARY_TESTS_NEAR_DOUBLE(pow)
BINARY_TESTS_NEAR_DOUBLE(hypot)

BINARY_TESTS_INT(add)
BINARY_TESTS_INT(sub)
BINARY_TESTS_INT(mul)

BINARY_TESTS_UINT(add)
BINARY_TESTS_UINT(sub)
BINARY_TESTS_UINT(mul)

BINARY_TESTS_INTL(add)
BINARY_TESTS_INTL(sub)
BINARY_TESTS_INTL(mul)

BINARY_TESTS_UINTL(add)
BINARY_TESTS_UINTL(sub)
BINARY_TESTS_UINTL(mul)

BINARY_TESTS_CFLOAT(add)
BINARY_TESTS_CFLOAT(sub)

BINARY_TESTS_CDOUBLE(add)
BINARY_TESTS_CDOUBLE(sub)

// Mixed types
BINARY_TESTS_NEAR(float, double, double, add, 1e-5)
BINARY_TESTS_NEAR(float, double, double, sub, 1e-5)
BINARY_TESTS_NEAR(float, double, double, mul, 1e-5)
BINARY_TESTS_NEAR(float, double, double, div, 1e-5)

BINARY_TESTS_NEAR(cfloat, cdouble, cdouble, add, 1e-5)
BINARY_TESTS_NEAR(cfloat, cdouble, cdouble, sub, 1e-5)
BINARY_TESTS_NEAR(cfloat, cdouble, cdouble, mul, 1e-5)
BINARY_TESTS_NEAR(cfloat, cdouble, cdouble, div, 1e-5)

BINARY_TESTS_NEAR_GENERAL(float, cfloat, cfloat, cfloat, cfloat, add, 1e-5)
BINARY_TESTS_NEAR_GENERAL(float, cfloat, cfloat, cfloat, cfloat, sub, 1e-5)
BINARY_TESTS_NEAR_GENERAL(float, cfloat, cfloat, cfloat, cfloat, mul, 1e-5)
BINARY_TESTS_NEAR_GENERAL(float, cfloat, cfloat, cfloat, cfloat, div, 1e-5)

BINARY_TESTS_NEAR_GENERAL(double, cfloat, cdouble, cdouble, cfloat, add, 1e-5)
BINARY_TESTS_NEAR_GENERAL(double, cfloat, cdouble, cdouble, cfloat, sub, 1e-5)
BINARY_TESTS_NEAR_GENERAL(double, cfloat, cdouble, cdouble, cfloat, mul, 1e-5)
BINARY_TESTS_NEAR_GENERAL(double, cfloat, cdouble, cdouble, cfloat, div, 1e-5)

BINARY_TESTS_NEAR_GENERAL(cfloat, double, cdouble, cfloat, cdouble, add, 1e-5)
BINARY_TESTS_NEAR_GENERAL(cfloat, double, cdouble, cfloat, cdouble, sub, 1e-5)
BINARY_TESTS_NEAR_GENERAL(cfloat, double, cdouble, cfloat, cdouble, mul, 1e-5)
BINARY_TESTS_NEAR_GENERAL(cfloat, double, cdouble, cfloat, cdouble, div, 1e-5)

#define BITOP(func, T, op)                                            \
    TEST(BinaryTests, Test_##func##_##T) {                            \
        af_dtype ty   = (af_dtype)dtype_traits<T>::af_type;           \
        const T vala  = 4095;                                         \
        const T valb  = 3;                                            \
        const T valc  = vala op valb;                                 \
        const int num = 10;                                           \
        af::array a   = af::constant(vala, num, ty);                  \
        af::array b   = af::constant(valb, num, ty);                  \
        af::array c   = a op b;                                       \
        T *h_a        = a.host<T>();                                  \
        T *h_b        = b.host<T>();                                  \
        T *h_c        = c.host<T>();                                  \
        for (int i = 0; i < num; i++)                                 \
            ASSERT_EQ(h_c[i], valc)                                   \
                << "for values: " << h_a[i] << "," << h_b[i] << endl; \
        af_free_host(h_a);                                            \
        af_free_host(h_b);                                            \
        af_free_host(h_c);                                            \
    }

BITOP(bitor, int, |)
BITOP(bitand, int, &)
BITOP(bitxor, int, ^)
BITOP(bitshiftl, int, <<)
BITOP(bitshiftr, int, >>)
BITOP(bitor, uint, |)
BITOP(bitand, uint, &)
BITOP(bitxor, uint, ^)
BITOP(bitshiftl, uint, <<)
BITOP(bitshiftr, uint, >>)

BITOP(bitor, intl, |)
BITOP(bitand, intl, &)
BITOP(bitxor, intl, ^)
BITOP(bitshiftl, intl, <<)
BITOP(bitshiftr, intl, >>)
BITOP(bitor, uintl, |)
BITOP(bitand, uintl, &)
BITOP(bitxor, uintl, ^)
BITOP(bitshiftl, uintl, <<)
BITOP(bitshiftr, uintl, >>)

#define UBITOP(func, T)                                     \
    TEST(BinaryTests, Test_##func##_##T) {                  \
        af_dtype ty   = (af_dtype)dtype_traits<T>::af_type; \
        const T vala  = 127u;                               \
        const T valc  = ~vala;                              \
        const int num = 10;                                 \
        af::array a   = af::constant(vala, num, ty);        \
        af::array b   = af::constant(valc, num, ty);        \
        af::array c   = ~a;                                 \
        ASSERT_ARRAYS_EQ(c, b);                             \
    }

UBITOP(bitnot, int)
UBITOP(bitnot, uint)
UBITOP(bitnot, intl)
UBITOP(bitnot, uintl)
UBITOP(bitnot, uchar)
UBITOP(bitnot, short)
UBITOP(bitnot, ushort)

TEST(BinaryTests, Test_pow_cfloat_float) {
    af::array a        = randgen(num, c32);
    af::array b        = randgen(num, f32);
    af::array c        = af::pow(a, b);
    complex_float *h_a = (complex_float *)a.host<cfloat>();
    float *h_b         = b.host<float>();
    complex_float *h_c = (complex_float *)c.host<cfloat>();
    for (int i = 0; i < num; i++) {
        complex_float res = std::pow(h_a[i], h_b[i]);
        ASSERT_NEAR(real(h_c[i]), real(res), 1E-5)
            << "for real values of: " << h_a[i] << "," << h_b[i] << endl;
        ASSERT_NEAR(imag(h_c[i]), imag(res), 1E-5)
            << "for imag values of: " << h_a[i] << "," << h_b[i] << endl;
    }
    af_free_host(h_a);
    af_free_host(h_b);
    af_free_host(h_c);
}

TEST(BinaryTests, Test_pow_cdouble_cdouble) {
    SUPPORTED_TYPE_CHECK(cdouble);
    af::array a         = randgen(num, c64);
    af::array b         = randgen(num, c64);
    af::array c         = af::pow(a, b);
    complex_double *h_a = (complex_double *)a.host<cdouble>();
    complex_double *h_b = (complex_double *)b.host<cdouble>();
    complex_double *h_c = (complex_double *)c.host<cdouble>();
    for (int i = 0; i < num; i++) {
        complex_double res = std::pow(h_a[i], h_b[i]);
        ASSERT_NEAR(real(h_c[i]), real(res), 1E-10)
            << "for real values of: " << h_a[i] << "," << h_b[i] << endl;
        ASSERT_NEAR(imag(h_c[i]), imag(res), 1E-10)
            << "for imag values of: " << h_a[i] << "," << h_b[i] << endl;
    }
    af_free_host(h_a);
    af_free_host(h_b);
    af_free_host(h_c);
}

TEST(BinaryTests, ISSUE_1762) {
    af::array zero   = af::constant(0, 5, f32);
    af::array result = af::pow(zero, 2);
    vector<complex_float> hres(result.elements());
    result.host(&hres[0]);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(real(hres[i]), 0);
        ASSERT_EQ(imag(hres[i]), 0);
    }
}

template<typename T>
class PowPrecisionTest : public ::testing::TestWithParam<T> {
    void SetUp() { SUPPORTED_TYPE_CHECK(T); }
};

#define DEF_TEST(Sx, T)                                                    \
    using PowPrecisionTest##Sx = PowPrecisionTest<T>;                      \
    TEST_P(PowPrecisionTest##Sx, Issue2304) {                              \
        T param    = GetParam();                                           \
        auto dtype = (af_dtype)dtype_traits<T>::af_type;                   \
        if (noDoubleTests(dtype)) {                                        \
            if (std::abs((double)param) > 10000)                           \
                GTEST_SKIP()                                               \
                    << "Skip larger values because double not supported."; \
        }                                                                  \
        af::array A = af::constant(param, 1, dtype);                       \
        af::array B = af::pow(A, 2);                                       \
        vector<T> hres(1, 0);                                              \
        B.host(&hres[0]);                                                  \
        std::fesetround(FE_TONEAREST);                                     \
        T gold;                                                            \
        if (!af::isDoubleAvailable(af::getDevice())) {                     \
            gold = (T)std::rint(std::pow((float)param, 2.0f));             \
        } else {                                                           \
            gold = (T)std::rint(std::pow((double)param, 2.0));             \
        }                                                                  \
        ASSERT_EQ(hres[0], gold);                                          \
    }

DEF_TEST(ULong, unsigned long long)
DEF_TEST(Long, long long)
DEF_TEST(UInt, unsigned int)
DEF_TEST(Int, int)
DEF_TEST(UShort, unsigned short)
DEF_TEST(Short, short)
DEF_TEST(UChar, unsigned char)

#undef DEF_TEST

INSTANTIATE_TEST_SUITE_P(PositiveValues, PowPrecisionTestULong,
                         testing::Range<unsigned long long>(1, 1e7, 1e6));
INSTANTIATE_TEST_SUITE_P(PositiveValues, PowPrecisionTestLong,
                         testing::Range<long long>(1, 1e7, 1e6));
INSTANTIATE_TEST_SUITE_P(PositiveValues, PowPrecisionTestUInt,
                         testing::Range<unsigned int>(1, 65000, 15e3));
INSTANTIATE_TEST_SUITE_P(PositiveValues, PowPrecisionTestInt,
                         testing::Range<int>(1, 46340, 10e3));
INSTANTIATE_TEST_SUITE_P(PositiveValues, PowPrecisionTestUShort,
                         testing::Range<unsigned short>(1, 255, 100));
INSTANTIATE_TEST_SUITE_P(PositiveValues, PowPrecisionTestShort,
                         testing::Range<short>(1, 180, 50));
INSTANTIATE_TEST_SUITE_P(PositiveValues, PowPrecisionTestUChar,
                         testing::Range<unsigned char>(1, 12, 5));

INSTANTIATE_TEST_SUITE_P(NegativeValues, PowPrecisionTestLong,
                         testing::Range<long long>(-1e7, 0, 1e6));
INSTANTIATE_TEST_SUITE_P(NegativeValues, PowPrecisionTestInt,
                         testing::Range<int>(-46340, 0, 10e3));
INSTANTIATE_TEST_SUITE_P(NegativeValues, PowPrecisionTestShort,
                         testing::Range<short>(-180, 0, 50));

struct result_type_param {
    af_dtype result_;
    af_dtype lhs_;
    af_dtype rhs_;

    result_type_param(af_dtype type) : result_(type), lhs_(type), rhs_(type) {}
    result_type_param(af_dtype result, af_dtype lhs, af_dtype rhs)
        : result_(result), lhs_(lhs), rhs_(rhs) {}
};

ostream &operator<<(ostream &os, const result_type_param &p) {
    os << "{lhs_ = " << p.lhs_ << " rhs_ = " << p.rhs_
       << " result_ = " << p.result_ << "}";
    return os;
}

class ResultType : public testing::TestWithParam<result_type_param> {
   protected:
    af::array lhs;
    af::array rhs;
    af_dtype gold;

    void SetUp() {
        result_type_param params = GetParam();
        gold                     = params.result_;
        if (noHalfTests(params.result_) || noHalfTests(params.lhs_) ||
            noHalfTests(params.rhs_)) {
            GTEST_SKIP() << "Half not supported on this device";
            return;
        } else if (noDoubleTests(params.result_) ||
                   noDoubleTests(params.lhs_) || noDoubleTests(params.rhs_)) {
            GTEST_SKIP() << "Double not supported on this device";
            return;
        }
        lhs = af::array(10, params.lhs_);
        rhs = af::array(10, params.rhs_);
    }
};

std::string print_types(
    const ::testing::TestParamInfo<ResultType::ParamType> info) {
    stringstream ss;
    ss << "lhs_" << info.param.lhs_ << "_rhs_" << info.param.rhs_ << "_result_"
       << info.param.result_;
    return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    SameTypes, ResultType,
    // clang-format off
    ::testing::Values(result_type_param(f32),
                      result_type_param(f64),
                      result_type_param(c32),
                      result_type_param(c64),
                      result_type_param(b8),
                      result_type_param(s32),
                      result_type_param(u32),
                      result_type_param(u8),
                      result_type_param(s64),
                      result_type_param(u64),
                      result_type_param(s16),
                      result_type_param(u16),
                      result_type_param(f16)),
    // clang-format on
    print_types);

INSTANTIATE_TEST_SUITE_P(
    Float, ResultType,
    // clang-format off
    ::testing::Values(result_type_param(f32),
                      result_type_param(f64, f64, f32),
                      result_type_param(c32, c32, f32),
                      result_type_param(c64, c64, f32),
                      result_type_param(f32, b8, f32),
                      result_type_param(f32, s32, f32),
                      result_type_param(f32, u32, f32),
                      result_type_param(f32, u8, f32),
                      result_type_param(f32, s64, f32),
                      result_type_param(f32, u64, f32),
                      result_type_param(f32, s16, f32),
                      result_type_param(f32, u16, f32),
                      result_type_param(f32, f16, f32)),
    // clang-format on
    print_types);

INSTANTIATE_TEST_SUITE_P(
    Double, ResultType,
    ::testing::Values(
        // clang-format off
                      result_type_param(f64, f32, f64),
                      result_type_param(f64, f64, f64),
                      result_type_param(c64, c32, f64),
                      result_type_param(c64, c64, f64),
                      result_type_param(f64, b8,  f64),
                      result_type_param(f64, s32, f64),
                      result_type_param(f64, u32, f64),
                      result_type_param(f64, u8,  f64),
                      result_type_param(f64, s64, f64),
                      result_type_param(f64, u64, f64),
                      result_type_param(f64, s16, f64),
                      result_type_param(f64, u16, f64),
                      result_type_param(f64, f16, f64)),
    // clang-format on
    print_types);

// clang-format off
TEST_P(ResultType, Addition)       {
    ASSERT_EQ(gold, (lhs + rhs).type());
}
TEST_P(ResultType, Subtraction)    {
    ASSERT_EQ(gold, (lhs - rhs).type());
}
TEST_P(ResultType, Multiplication) {
    ASSERT_EQ(gold, (lhs * rhs).type());
}
TEST_P(ResultType, Division)       {
    ASSERT_EQ(gold, (lhs / rhs).type());
}
// clang-format on

template<typename T>
class ResultTypeScalar : public ::testing::Test {
   protected:
    T scalar;
    void SetUp() { scalar = T(1); }
};

typedef ::testing::Types<float, double, unsigned int, int, short,
                         unsigned short, char, unsigned char, half_float::half>
    TestTypes;
TYPED_TEST_SUITE(ResultTypeScalar, TestTypes);

TYPED_TEST(ResultTypeScalar, HalfAddition) {
    SUPPORTED_TYPE_CHECK(half_float::half);
    ASSERT_EQ(f16, (af::array(10, f16) + this->scalar).type());
}

TYPED_TEST(ResultTypeScalar, HalfSubtraction) {
    SUPPORTED_TYPE_CHECK(half_float::half);
    ASSERT_EQ(f16, (af::array(10, f16) - this->scalar).type());
}

TYPED_TEST(ResultTypeScalar, HalfMultiplication) {
    SUPPORTED_TYPE_CHECK(half_float::half);
    ASSERT_EQ(f16, (af::array(10, f16) * this->scalar).type());
}

TYPED_TEST(ResultTypeScalar, HalfDivision) {
    SUPPORTED_TYPE_CHECK(half_float::half);
    ASSERT_EQ(f16, (af::array(10, f16) / this->scalar).type());
}

TYPED_TEST(ResultTypeScalar, FloatAddition) {
    ASSERT_EQ(f32, (af::array(10, f32) + this->scalar).type());
}

TYPED_TEST(ResultTypeScalar, FloatSubtraction) {
    ASSERT_EQ(f32, (af::array(10, f32) - this->scalar).type());
}

TYPED_TEST(ResultTypeScalar, FloatMultiplication) {
    ASSERT_EQ(f32, (af::array(10, f32) * this->scalar).type());
}

TYPED_TEST(ResultTypeScalar, FloatDivision) {
    ASSERT_EQ(f32, (af::array(10, f32) / this->scalar).type());
}

class Broadcast : public ::testing::TestWithParam<std::tuple<dim4, dim4>> {
    void SetUp() override {}
};
/// clang-format off

INSTANTIATE_TEST_SUITE_P(
    CorrectCases, Broadcast,
    ::testing::Combine(
        ::testing::Values(dim4(1), dim4(10), dim4(1, 10), dim4(1, 1, 10),
                          dim4(1, 1, 1, 10), dim4(10, 10), dim4(1, 10, 10),
                          dim4(1, 1, 10, 10), dim4(10, 1, 10),
                          dim4(1, 10, 1, 10), dim4(10, 1, 1, 10),
                          dim4(10, 10, 10), dim4(1, 10, 10, 10),
                          dim4(10, 1, 10, 10), dim4(10, 10, 1, 10),
                          dim4(10, 10, 10, 10)),
        ::testing::Values(dim4(1), dim4(10), dim4(1, 10), dim4(1, 1, 10),
                          dim4(1, 1, 1, 10), dim4(10, 10), dim4(1, 10, 10),
                          dim4(1, 1, 10, 10), dim4(10, 1, 10),
                          dim4(1, 10, 1, 10), dim4(10, 1, 1, 10),
                          dim4(10, 10, 10), dim4(1, 10, 10, 10),
                          dim4(10, 1, 10, 10), dim4(10, 10, 1, 10),
                          dim4(10, 10, 10, 10))),
    [](const ::testing::TestParamInfo<Broadcast::ParamType> info) {
        stringstream ss;
        ss << "lhs_" << get<0>(info.param) << "_rhs_" << get<1>(info.param);
        string s = ss.str();
        std::replace(begin(s), std::end(s), ' ', '_');
        return s;
    });
/// clang-format on

af::dim4 broadcastOut(dim4 lhs, dim4 rhs) {
    dim4 out(1);
    for (int i = 0; i < AF_MAX_DIMS; i++) {
        if (lhs[i] == rhs[i])
            out[i] = lhs[i];
        else if (lhs[i] == 1 && rhs[i] > 1)
            out[i] = rhs[i];
        else if (lhs[i] > 1 && rhs[i] == 1)
            out[i] = lhs[i];
        else {
            std::cout << "incorrect dimension" << lhs << " op " << rhs;
            return dim4(0);
        }
    }
    return out;
}

af::dim4 tileRepeations(dim4 in, dim4 other) {
    af::dim4 out;
    for (int i = 0; i < AF_MAX_DIMS; i++) {
        out[i] = std::max(dim_t(1), other[i] / in[i]);
    }
    return out;
}

TEST_P(Broadcast, Addition) {
    auto params   = GetParam();
    af::array lhs = iota(get<0>(params));
    af::array rhs = constant(1, get<1>(params));

    af::array out = lhs + rhs;

    af::dim4 outdims       = broadcastOut(lhs.dims(), rhs.dims());
    af::dim4 tilerepetions = tileRepeations(lhs.dims(), rhs.dims());
    af::array tiledlhs     = tile(lhs, tilerepetions);

    vector<float> outvec(outdims.elements());
    tiledlhs.host(outvec.data());
    for (auto &out : outvec) { out += 1; }

    ASSERT_VEC_ARRAY_EQ(outvec, outdims, out);
}

TEST_P(Broadcast, Subtraction) {
    auto params   = GetParam();
    af::array lhs = range(get<0>(params));
    af::array rhs = constant(1, get<1>(params));

    af::array out          = lhs - rhs;
    af::dim4 outdims       = broadcastOut(lhs.dims(), rhs.dims());
    af::dim4 tilerepetions = tileRepeations(lhs.dims(), rhs.dims());
    af::array tiledlhs     = tile(lhs, tilerepetions);

    vector<float> outvec(outdims.elements());
    tiledlhs.host(outvec.data());
    for (auto &out : outvec) { out -= 1; }

    ASSERT_VEC_ARRAY_EQ(outvec, outdims, out);
}

TEST_P(Broadcast, Multiplication) {
    auto params   = GetParam();
    af::array lhs = range(get<0>(params));
    af::array rhs = constant(2, get<1>(params));

    af::array out          = lhs * rhs;
    af::dim4 outdims       = broadcastOut(lhs.dims(), rhs.dims());
    af::dim4 tilerepetions = tileRepeations(lhs.dims(), rhs.dims());
    af::array tiledlhs     = tile(lhs, tilerepetions);

    vector<float> outvec(outdims.elements());
    tiledlhs.host(outvec.data());
    for (auto &out : outvec) { out *= 2; }

    ASSERT_VEC_ARRAY_EQ(outvec, outdims, out);
}

TEST_P(Broadcast, Division) {
    auto params   = GetParam();
    af::array lhs = range(get<0>(params));
    af::array rhs = constant(2, get<1>(params));

    af::array out          = lhs / rhs;
    af::dim4 outdims       = broadcastOut(lhs.dims(), rhs.dims());
    af::dim4 tilerepetions = tileRepeations(lhs.dims(), rhs.dims());
    af::array tiledlhs     = tile(lhs, tilerepetions);

    vector<float> outvec(outdims.elements());
    tiledlhs.host(outvec.data());
    for (auto &out : outvec) { out /= 2; }

    ASSERT_VEC_ARRAY_EQ(outvec, outdims, out);
}

TEST_P(Broadcast, AdditionLHSIndexed) {
    auto params   = GetParam();
    af::array lhs = iota(get<0>(params) * 2);
    af::array rhs = constant(1, get<1>(params));

    dim4 lhs_dims = get<0>(params);
    af::array out = lhs(seq(lhs_dims[0]), seq(lhs_dims[1]), seq(lhs_dims[2]),
                        seq(lhs_dims[3])) +
                    rhs;

    af::dim4 outdims       = broadcastOut(get<0>(params), rhs.dims());
    af::array indexedlhs   = lhs(seq(lhs_dims[0]), seq(lhs_dims[1]),
                                 seq(lhs_dims[2]), seq(lhs_dims[3]));
    af::dim4 tilerepetions = tileRepeations(get<0>(params), rhs.dims());
    af::array tiledlhs     = tile(indexedlhs, tilerepetions);

    vector<float> outvec(outdims.elements());
    tiledlhs.host(outvec.data());
    for (auto &out : outvec) { out += 1; }

    ASSERT_VEC_ARRAY_EQ(outvec, outdims, out);
}

TEST_P(Broadcast, AdditionRHSIndexed) {
    auto params   = GetParam();
    af::array lhs = iota(get<0>(params));
    af::array rhs = constant(1, get<1>(params) * 2);

    dim4 rhs_dims = get<1>(params);
    af::array out = lhs + rhs(seq(rhs_dims[0]), seq(rhs_dims[1]),
                              seq(rhs_dims[2]), seq(rhs_dims[3]));

    af::dim4 outdims       = broadcastOut(get<0>(params), get<1>(params));
    af::dim4 tilerepetions = tileRepeations(get<0>(params), get<1>(params));
    af::array tiledlhs     = tile(lhs, tilerepetions);

    vector<float> outvec(outdims.elements());
    tiledlhs.host(outvec.data());
    for (auto &out : outvec) { out += 1; }

    ASSERT_VEC_ARRAY_EQ(outvec, outdims, out);
}

TEST_P(Broadcast, AdditionBothIndexed) {
    auto params   = GetParam();
    af::array lhs = iota(get<0>(params) * 2);
    af::array rhs = constant(1, get<1>(params) * 2);

    dim4 lhs_dims = get<0>(params);
    dim4 rhs_dims = get<1>(params);
    af::array out = lhs(seq(lhs_dims[0]), seq(lhs_dims[1]), seq(lhs_dims[2]),
                        seq(lhs_dims[3])) +
                    rhs(seq(rhs_dims[0]), seq(rhs_dims[1]), seq(rhs_dims[2]),
                        seq(rhs_dims[3]));

    af::dim4 outdims       = broadcastOut(lhs_dims, rhs_dims);
    af::array indexedlhs   = lhs(seq(lhs_dims[0]), seq(lhs_dims[1]),
                                 seq(lhs_dims[2]), seq(lhs_dims[3]));
    af::dim4 tilerepetions = tileRepeations(get<0>(params), get<1>(params));
    af::array tiledlhs     = tile(indexedlhs, tilerepetions);

    vector<float> outvec(outdims.elements());
    tiledlhs.host(outvec.data());
    for (auto &out : outvec) { out += 1; }

    ASSERT_VEC_ARRAY_EQ(outvec, outdims, out);
}

TEST(Broadcast, VectorMatrix2d) {
    dim_t s     = 10;
    af::array A = range(dim4(s, 3), 1);
    af::array B = -range(dim4(3));

    try {
        A + B;
        FAIL();
    } catch (af::exception &e) { ASSERT_EQ(e.err(), AF_ERR_SIZE); }
    try {
        B + A;
        FAIL();
    } catch (af::exception &e) { ASSERT_EQ(e.err(), AF_ERR_SIZE); }
}

TEST(Broadcast, VectorMatrix3d) {
    dim_t s     = 10;
    af::array A = range(dim4(s, s, 3), 2);
    af::array B = -range(dim4(3));

    try {
        A + B;
        FAIL();
    } catch (af::exception &e) { ASSERT_EQ(e.err(), AF_ERR_SIZE); }
    try {
        B + A;
        FAIL();
    } catch (af::exception &e) { ASSERT_EQ(e.err(), AF_ERR_SIZE); }
}

TEST(Broadcast, VectorMatrix4d) {
    dim_t s     = 10;
    af::array A = range(dim4(s, s, s, 3), 3);
    af::array B = -range(dim4(3));

    try {
        A + B;
        FAIL();
    } catch (af::exception &e) { ASSERT_EQ(e.err(), AF_ERR_SIZE); }
    try {
        B + A;
        FAIL();
    } catch (af::exception &e) { ASSERT_EQ(e.err(), AF_ERR_SIZE); }
}

void testAllBroadcast(dim4 dims) {
    af::array A = constant(1, dims);
    for (int k = 0; k < dims.ndims(); ++k) {
        dim4 rdims  = dims;
        rdims[k]    = 1;
        af::array B = constant(-1, rdims);
        af::array C = A + B;
        ASSERT_ARRAYS_EQ(C, constant(0, dims));

        C = B + A;
        ASSERT_ARRAYS_EQ(C, constant(0, dims));
    }
}

TEST(Broadcast, MatrixMatrix2d) { testAllBroadcast(dim4(10, 15)); }

TEST(Broadcast, MatrixMatrix3d) { testAllBroadcast(dim4(10, 15, 20)); }

TEST(Broadcast, MatrixMatrix4d) { testAllBroadcast(dim4(10, 15, 20, 25)); }

TEST(Broadcast, MismatchingDim0) {
    af::array A = range(dim4(2, 3, 5), 1);
    af::array B = -range(dim4(3, 5), 0);

    try {
        A + B;
    } catch (af::exception &e) { ASSERT_EQ(e.err(), AF_ERR_SIZE); }
}

TEST(Broadcast, TestFirstMatchingDim) {
    af::array A = range(dim4(3, 2, 2, 4), 1);
    af::array B = -range(dim4(2));

    try {
        A + B;
    } catch (af::exception &e) { ASSERT_EQ(e.err(), AF_ERR_SIZE); }
}

TEST(Broadcast, ManySlicesVsOneSlice) {
    af::array A = constant(1, dim4(3, 3, 2));
    af::array B = constant(2, dim4(3, 3));
    af::array C = A + B;

    ASSERT_ARRAYS_EQ(C, constant(3, dim4(3, 3, 2)));

    C = B + A;
    ASSERT_ARRAYS_EQ(C, constant(3, dim4(3, 3, 2)));
}

TEST(Broadcast, SubArray) {
    dim_t subdim = 5;
    af::array A  = constant(1, dim4(10, 10, 2));
    af::array B  = constant(2, dim4(5, 5));
    af::array C  = A(seq(subdim), seq(subdim), span) + B;

    ASSERT_ARRAYS_EQ(C, constant(3, dim4(subdim, subdim, 2)));

    C = B + A(seq(subdim), seq(subdim), span);
    ASSERT_ARRAYS_EQ(C, constant(3, dim4(subdim, subdim, 2)));
}

TEST(Broadcast, SubArrays) {
    dim_t subdim = 5;
    af::array A  = constant(1, dim4(10, 10, 2));
    af::array B  = constant(2, dim4(15, 15));

    af::array C =
        A(seq(subdim), seq(subdim), span) + B(seq(subdim), seq(subdim));
    ASSERT_ARRAYS_EQ(C, constant(3, dim4(subdim, subdim, 2)));

    C = B(seq(subdim), seq(subdim)) + A(seq(subdim), seq(subdim), span);
    ASSERT_ARRAYS_EQ(C, constant(3, dim4(subdim, subdim, 2)));
}

TEST(Broadcast, IndexedArray) {
    af::array A = constant(1, dim4(2, 2, 2, 2));
    af::array B = constant(-1, dim4(1, 5));

    af::array idx = range(dim4(2, 2, 2, 2), 0);

    af::array C = A(idx % 2 == 0) + B;
    ASSERT_ARRAYS_EQ(C, constant(0, dim4(8, 5)));

    C = B + A(idx % 2 == 0);
    ASSERT_ARRAYS_EQ(C, constant(0, dim4(8, 5)));
}
