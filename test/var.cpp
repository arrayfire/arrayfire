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
using af::array;

template<typename T>
class Var : public ::testing::Test
{

};

typedef ::testing::Types< float, double, cfloat, cdouble, uint, int, uintl, intl, char, uchar> TestTypes;
TYPED_TEST_CASE(Var, TestTypes);

template<typename T>
struct elseType {
   typedef typename cond_type< is_same_type<T, uintl>::value ||
                               is_same_type<T, intl>::value,
                                              double,
                                              T>::type type;
};

template<typename T>
struct varOutType {
   typedef typename cond_type< is_same_type<T, float>::value ||
                               is_same_type<T, int>::value ||
                               is_same_type<T, uint>::value ||
                               is_same_type<T, uchar>::value ||
                               is_same_type<T, char>::value,
                                              float,
                              typename elseType<T>::type>::type type;
};



//////////////////////////////// CPP ////////////////////////////////////
// test var_all interface using cpp api

template<typename T>
void testCPPVar(T const_value, af::dim4 dims)
{
    typedef typename varOutType<T>::type outType;
    if (noDoubleTests<T>()) return;
    if (noDoubleTests<outType>()) return;

    using af::array;
    using af::var;

    vector<T> hundred(dims.elements(), const_value);

    outType gold = outType(0);

    array a(dims, &(hundred.front()));
    outType output = var<outType>(a, false);

    ASSERT_NEAR(::real(output), ::real(gold), 1.0e-3);
    ASSERT_NEAR(::imag(output), ::imag(gold), 1.0e-3);

    output = var<outType>(a, true);

    ASSERT_NEAR(::real(output), ::real(gold), 1.0e-3);
    ASSERT_NEAR(::imag(output), ::imag(gold), 1.0e-3);

    gold = outType(2.5);
    outType tmp[] = { outType(0), outType(1), outType(2), outType(3),
        outType(4) };
    array b(5, tmp);
    output = var<outType>(b, false);

    ASSERT_NEAR(::real(output), ::real(gold), 1.0e-3);
    ASSERT_NEAR(::imag(output), ::imag(gold), 1.0e-3);

    gold = outType(2);
    output = var<outType>(b, true);

    ASSERT_NEAR(::real(output), ::real(gold), 1.0e-3);
    ASSERT_NEAR(::imag(output), ::imag(gold), 1.0e-3);
}

TYPED_TEST(Var, AllCPPSmall)
{
    testCPPVar<TypeParam>(2, af::dim4(10, 10, 1, 1));
}

TYPED_TEST(Var, AllCPPMedium)
{
    testCPPVar<TypeParam>(2, af::dim4(100, 100, 1, 1));
}

TYPED_TEST(Var, AllCPPLarge)
{
    testCPPVar<TypeParam>(2, af::dim4(1000, 1000, 1, 1));
}

TYPED_TEST(Var, DimCPPSmall)
{
    typedef typename varOutType<TypeParam>::type outType;

    if (noDoubleTests<TypeParam>()) return;
    if (noDoubleTests<outType>()) return;

    vector<af::dim4> numDims;
    vector<vector<TypeParam> > in;
    vector<vector<outType> > tests;

    readTests<TypeParam, outType, double> (TEST_DIR"/var/var.data",numDims,in,tests);

    for(size_t i = 0; i < in.size(); i++)
    {
        array input(numDims[i], &in[i].front(), afHost);

        array bout  = var(input, false);
        array nbout = var(input, true);

        array bout1  = var(input, false, 1);
        array nbout1 = var(input, true,  1);

        vector<vector<outType> > h_out(4);

        h_out[0].resize(bout.elements());
        h_out[1].resize(nbout.elements());
        h_out[2].resize(bout1.elements());
        h_out[3].resize(nbout1.elements());

        bout.host(  &h_out[0].front());
        nbout.host( &h_out[1].front());
        bout1.host( &h_out[2].front());
        nbout1.host(&h_out[3].front());

        for(size_t j = 0; j < tests.size(); j++) {
            for(size_t jj = 0; jj < tests[j].size(); jj++) {
                ASSERT_EQ(h_out[j][jj], tests[j][jj]);
            }
        }
    }
}
