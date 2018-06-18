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
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::endl;
using std::string;
using std::vector;
using af::array;
using af::cfloat;
using af::cdouble;
using af::freeHost;
using af::dim4;
using af::dtype_traits;

template<typename T>
class Write : public ::testing::Test
{
    public:
        virtual void SetUp() {
        }
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble, int, unsigned, char, unsigned char, short, ushort> TestTypes;

// register the type list
TYPED_TEST_CASE(Write, TestTypes);

template<typename T>
void writeTest(dim4 dims)
{
    if (noDoubleTests<T>()) return;

    array A = randu(dims, (af_dtype) dtype_traits<T>::af_type);
    array B = randu(dims, (af_dtype) dtype_traits<T>::af_type);

    array A_copy = A.copy();
    array B_copy = B.copy();

    T *a_host = A.host<T>();
    T *b_dev  = B.device<T>();

    A.write(b_dev, dims.elements() * sizeof(T), afDevice);
    B.write(a_host, dims.elements() * sizeof(T), afHost);

    array check1 = A != B_copy;     // False so check1 is all 0s
    array check2 = B != A_copy;     // False so check2 is all 0s

    char *h_check1 = check1.host<char>();
    char *h_check2 = check2.host<char>();

    for(int i = 0; i < (int)dims.elements(); i++) {
        ASSERT_EQ(h_check1[i], 0) << "at: " << i << endl;
        ASSERT_EQ(h_check2[i], 0) << "at: " << i << endl;
    }

    freeHost(a_host);
    freeHost(h_check1);
    freeHost(h_check2);
}

TYPED_TEST(Write, Vector0)
{
    writeTest<TypeParam>(dim4(10));
}

TYPED_TEST(Write, Vector1)
{
    writeTest<TypeParam>(dim4(1000));
}

TYPED_TEST(Write, Matrix0)
{
    writeTest<TypeParam>(dim4(64, 8));
}

TYPED_TEST(Write, Matrix1)
{
    writeTest<TypeParam>(dim4(256, 256));
}

TYPED_TEST(Write, Volume0)
{
    writeTest<TypeParam>(dim4(10, 10, 10));
}

TYPED_TEST(Write, Volume1)
{
    writeTest<TypeParam>(dim4(32, 64, 16));
}
