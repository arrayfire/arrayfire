/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <ctime>
#include <algorithm>
#include <testHelpers.hpp>

using namespace af;

template<typename T>
class CorrelationCoefficient : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, int, uint, intl, uintl, char, uchar> TestTypes;

// register the type list
TYPED_TEST_CASE(CorrelationCoefficient, TestTypes);

template<typename T>
struct f32HelperType {
   typedef typename cond_type<is_same_type<T, double>::value,
                                             double,
                                             float>::type type;
};

template<typename T>
struct c32HelperType {
   typedef typename cond_type<is_same_type<T, cfloat>::value,
                                             cfloat,
                                             typename f32HelperType<T>::type >::type type;
};

template<typename T>
struct elseType {
   typedef typename cond_type< is_same_type<T, uintl>::value ||
                               is_same_type<T, intl> ::value,
                                              double,
                                              T>::type type;
};

template<typename T>
struct ccOutType {
   typedef typename cond_type< is_same_type<T, float>   ::value ||
                               is_same_type<T, int>     ::value ||
                               is_same_type<T, uint>    ::value ||
                               is_same_type<T, uchar>   ::value ||
                               is_same_type<T, short>   ::value ||
                               is_same_type<T, ushort>  ::value ||
                               is_same_type<T, char>    ::value,
                                              float,
                              typename elseType<T>::type>::type type;
};

TYPED_TEST(CorrelationCoefficient, All)
{
    typedef typename ccOutType<TypeParam>::type outType;
    if (noDoubleTests<TypeParam>()) return;
    if (noDoubleTests<outType>()) return;

    std::vector<af::dim4>      numDims;
    std::vector<std::vector<int> >       in;
    std::vector<std::vector<float> >  tests;

    readTestsFromFile<int,float>(std::string(TEST_DIR "/corrcoef/mat_10x10_scalar.test"),
                                 numDims, in, tests);

    std::vector<TypeParam> input1(in[0].begin(), in[0].end());
    std::vector<TypeParam> input2(in[1].begin(), in[1].end());

    array a(numDims[0], &(input1.front()));
    array b(numDims[1], &(input2.front()));
    outType c = corrcoef<outType>(a, b);

    std::vector<outType> currGoldBar(tests[0].begin(), tests[0].end());
    ASSERT_NEAR(::real(currGoldBar[0]), ::real(c), 1.0e-3);
    ASSERT_NEAR(::imag(currGoldBar[0]), ::imag(c), 1.0e-3);
}
