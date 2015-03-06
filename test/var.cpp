/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <string>
#include <vector>
#include <testHelpers.hpp>

using std::string;
using std::vector;
using af::cdouble;
using af::cfloat;

template<typename T>
using helperType = typename std::conditional<std::is_same<T, uchar>::value,
                                             uint,
                                             T>::type;


template<typename T>
using varOutType = typename std::conditional<std::is_same<T, char>::value,
                                              int,
                                              helperType<T>>::type;

//////////////////////////////// CPP ////////////////////////////////////
// test var_all interface using cpp api

template<typename T>
void testCPPVar(T const_value, af::dim4 dims)
{
    typedef varOutType<T> outType;
    if (noDoubleTests<T>()) return;

    using af::array;
    using af::var;

    vector<T> hundred(dims.elements(), const_value);

    outType gold = outType(0);

    array a(dims, &(hundred.front()));
    outType output = var<outType>(a, false);

    ASSERT_NEAR(std::real(output), std::real(gold), 1.0e-3);
    ASSERT_NEAR(std::imag(output), std::imag(gold), 1.0e-3);

    output = var<outType>(a, true);

    ASSERT_NEAR(std::real(output), std::real(gold), 1.0e-3);
    ASSERT_NEAR(std::imag(output), std::imag(gold), 1.0e-3);

    gold = outType(2.5);
    outType tmp[] = { outType(0), outType(1), outType(2), outType(3),
        outType(4) };
    array b(5, tmp);
    output = var<outType>(b, false);

    ASSERT_NEAR(std::real(output), std::real(gold), 1.0e-3);
    ASSERT_NEAR(std::imag(output), std::imag(gold), 1.0e-3);

    gold = outType(2);
    output = var<outType>(b, true);

    ASSERT_NEAR(std::real(output), std::real(gold), 1.0e-3);
    ASSERT_NEAR(std::imag(output), std::imag(gold), 1.0e-3);
}

TEST(Var, CPP_f64)
{
    testCPPVar<double>(2.1, af::dim4(10, 10, 1, 1));
}

TEST(Var, CPP_f32)
{
    testCPPVar<float>(2.1f, af::dim4(10, 5, 2, 1));
}

TEST(Var, CPP_s32)
{
    testCPPVar<int>(2, af::dim4(5, 5, 2, 2));
}

TEST(Var, CPP_u32)
{
    testCPPVar<unsigned>(2, af::dim4(100, 1, 1, 1));
}

TEST(Var, CPP_s8)
{
    testCPPVar<char>(2, af::dim4(5, 5, 2, 2));
}

TEST(Var, CPP_u8)
{
    testCPPVar<uchar>(2, af::dim4(100, 1, 1, 1));
}

TEST(Var, CPP_cfloat)
{
    testCPPVar<cfloat>(cfloat(2.1f), af::dim4(10, 5, 2, 1));
}

TEST(Var, CPP_cdouble)
{
    testCPPVar<cdouble>(cdouble(2.1), af::dim4(10, 10, 1, 1));
}
