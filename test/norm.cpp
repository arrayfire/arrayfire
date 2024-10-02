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
#include <sstream>

using af::array;
using af::constant;
using af::dim4;
using std::complex;
using std::stringstream;
using std::vector;

std::ostream &operator<<(std::ostream &os, af::normType nt) {
    switch (nt) {
        case AF_NORM_VECTOR_1: os << "AF_NORM_VECTOR_1"; break;
        case AF_NORM_VECTOR_INF: os << "AF_NORM_VECTOR_INF"; break;
        case AF_NORM_VECTOR_2: os << "AF_NORM_VECTOR_2"; break;
        case AF_NORM_VECTOR_P: os << "AF_NORM_VECTOR_P"; break;
        case AF_NORM_MATRIX_1: os << "AF_NORM_MATRIX_1"; break;
        case AF_NORM_MATRIX_INF: os << "AF_NORM_MATRIX_INF"; break;
        case AF_NORM_MATRIX_2: os << "AF_NORM_MATRIX_2"; break;
        case AF_NORM_MATRIX_L_PQ: os << "AF_NORM_MATRIX_L_PQ"; break;
    }
    return os;
}

template<typename T>
double cpu_norm1_impl(af::dim4 &dims, std::vector<T> &value) {
    int M = dims[0];
    int N = dims[1];

    double norm1 = std::numeric_limits<double>::lowest();
    for (int n = 0; n < N; n++) {
        T *columnN = value.data() + n * M;
        double sum = 0;
        for (int m = 0; m < M; m++) { sum += abs(columnN[m]); }
        norm1 = std::max(norm1, sum);
    }
    return norm1;
}

double cpu_norm1(af::array &value) {
    double norm1;
    af::dim4 dims = value.dims();
    if (value.type() == f16) {
        vector<half_float::half> values(value.elements());
        value.host(values.data());
        norm1 = cpu_norm1_impl<half_float::half>(dims, values);
    } else if (value.type() == c32 || value.type() == c64) {
        vector<complex<double> > values(value.elements());
        value.as(c64).host(values.data());
        norm1 = cpu_norm1_impl<complex<double> >(dims, values);
    } else {
        vector<double> values(value.elements());
        value.as(f64).host(values.data());
        norm1 = cpu_norm1_impl<double>(dims, values);
    }
    return norm1;
}

template<typename T>
double cpu_norm_inf_impl(af::dim4 &dims, std::vector<T> &value) {
    int M = dims[0];
    int N = dims[1];

    double norm_inf = std::numeric_limits<double>::lowest();
    for (int m = 0; m < M; m++) {
        T *rowM    = value.data() + m;
        double sum = 0;
        for (int n = 0; n < N; n++) { sum += abs(rowM[n * M]); }
        norm_inf = std::max(norm_inf, sum);
    }
    return norm_inf;
}

double cpu_norm_inf(af::array &value) {
    double norm_inf;
    af::dim4 dims = value.dims();
    if (value.type() == c32 || value.type() == c64) {
        vector<complex<double> > values(value.elements());
        value.as(c64).host(values.data());
        norm_inf = cpu_norm_inf_impl<complex<double> >(dims, values);
    } else {
        vector<double> values(value.elements());
        value.as(f64).host(values.data());
        norm_inf = cpu_norm_inf_impl<double>(dims, values);
    }
    return norm_inf;
}

using norm_params = std::tuple<af::dim4, af::dtype>;
class Norm
    : public ::testing::TestWithParam<std::tuple<af::dim4, af::dtype> > {};

INSTANTIATE_TEST_CASE_P(
    Norm, Norm,
    ::testing::Combine(::testing::Values(dim4(3, 3), dim4(32, 32), dim4(33, 33),
                                         dim4(64, 64), dim4(128, 128),
                                         dim4(129, 129), dim4(256, 256),
                                         dim4(257, 257)),
                       ::testing::Values(f32, f64, c32, c64, f16)),
    [](const ::testing::TestParamInfo<Norm::ParamType> info) {
        stringstream ss;
        using std::get;
        ss << "dims_" << get<0>(info.param)[0] << "_" << get<0>(info.param)[1]
           << "_dtype_" << get<1>(info.param);
        return ss.str();
    });

TEST_P(Norm, Identity_AF_NORM_MATRIX_1) {
    using std::get;
    norm_params param = GetParam();
    if (get<1>(param) == f16) SUPPORTED_TYPE_CHECK(half_float::half);
    if (get<1>(param) == f64) SUPPORTED_TYPE_CHECK(double);

    array identity = af::identity(get<0>(param), get<1>(param));
    double result  = norm(identity, AF_NORM_MATRIX_1);
    double norm1   = cpu_norm1(identity);

    ASSERT_DOUBLE_EQ(norm1, result);
}

TEST_P(Norm, Random_AF_NORM_MATRIX_1) {
    using std::get;
    norm_params param = GetParam();
    if (get<1>(param) == f16) SUPPORTED_TYPE_CHECK(half_float::half);
    if (get<1>(param) == f64) SUPPORTED_TYPE_CHECK(double);

    array in      = af::randu(get<0>(param), get<1>(param)) - 0.5f;
    double result = norm(in, AF_NORM_MATRIX_1);
    double norm1  = cpu_norm1(in);

    ASSERT_NEAR(norm1, result, 2e-4);
}

TEST_P(Norm, Identity_AF_NORM_MATRIX_2_NOT_SUPPORTED) {
    using std::get;
    norm_params param = GetParam();
    if (get<1>(param) == f16) SUPPORTED_TYPE_CHECK(half_float::half);
    if (get<1>(param) == f64) SUPPORTED_TYPE_CHECK(double);
    try {
        double result =
            norm(af::identity(get<0>(param), get<1>(param)), AF_NORM_MATRIX_2);
        FAIL();
    } catch (af::exception &ex) {
        ASSERT_EQ(AF_ERR_NOT_SUPPORTED, ex.err());
        return;
    }
    FAIL();
}

TEST_P(Norm, Identity_AF_NORM_MATRIX_INF) {
    using std::get;
    norm_params param = GetParam();
    if (get<1>(param) == f16) SUPPORTED_TYPE_CHECK(half_float::half);
    if (get<1>(param) == f64) SUPPORTED_TYPE_CHECK(double);
    array in        = af::identity(get<0>(param), get<1>(param));
    double result   = norm(in, AF_NORM_MATRIX_INF);
    double norm_inf = cpu_norm_inf(in);

    ASSERT_DOUBLE_EQ(norm_inf, result);
}

TEST_P(Norm, Random_AF_NORM_MATRIX_INF) {
    using std::get;
    norm_params param = GetParam();
    if (get<1>(param) == f16) SUPPORTED_TYPE_CHECK(half_float::half);
    if (get<1>(param) == f64) SUPPORTED_TYPE_CHECK(double);
    array in        = af::randu(get<0>(param), get<1>(param));
    double result   = norm(in, AF_NORM_MATRIX_INF);
    double norm_inf = cpu_norm_inf(in);

    ASSERT_NEAR(norm_inf, result, 2e-4);
}
