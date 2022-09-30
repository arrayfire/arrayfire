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
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype_traits;
using af::freeHost;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Write : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, cfloat, double, cdouble, int, unsigned, char,
                         unsigned char, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_SUITE(Write, TestTypes);

template<typename T>
void writeTest(dim4 dims) {
    SUPPORTED_TYPE_CHECK(T);

    array A = randu(dims, (af_dtype)dtype_traits<T>::af_type);
    array B = randu(dims, (af_dtype)dtype_traits<T>::af_type);

    array A_copy = A.copy();
    array B_copy = B.copy();

    T *a_host = A.host<T>();
    T *b_dev  = B.device<T>();

    A.write(b_dev, dims.elements() * sizeof(T), afDevice);
    B.write(a_host, dims.elements() * sizeof(T), afHost);

    ASSERT_ARRAYS_EQ(B_copy, A);
    ASSERT_ARRAYS_EQ(A_copy, B);

    freeHost(a_host);
}

TYPED_TEST(Write, Vector0) { writeTest<TypeParam>(dim4(10)); }

TYPED_TEST(Write, Vector1) { writeTest<TypeParam>(dim4(1000)); }

TYPED_TEST(Write, Matrix0) { writeTest<TypeParam>(dim4(64, 8)); }

TYPED_TEST(Write, Matrix1) { writeTest<TypeParam>(dim4(256, 256)); }

TYPED_TEST(Write, Volume0) { writeTest<TypeParam>(dim4(10, 10, 10)); }

TYPED_TEST(Write, Volume1) { writeTest<TypeParam>(dim4(32, 64, 16)); }

TEST(Write, VoidPointer) {
    vector<float> gold(100, 5);

    array a(100);

    void *h_gold = (void *)&gold.front();
    a.write(h_gold, 100 * sizeof(float), afHost);

    ASSERT_VEC_ARRAY_EQ(gold, dim4(100), a);
}
