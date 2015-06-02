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

template<typename T>
class Constant : public ::testing::Test { };

typedef ::testing::Types<float, af::cfloat, double, af::cdouble, int, unsigned, char, uchar, uintl, intl> TestTypes;
TYPED_TEST_CASE(Constant, TestTypes);

template<typename T>
void ConstantCPPCheck(T value) {
    if (noDoubleTests<T>()) return;

    const int num = 1000;
    T val = value;
    dtype dty = (dtype) dtype_traits<T>::af_type;
    af::array in = constant(val, num, dty);

    vector<T> h_in(num);
    in.host(&h_in.front());

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(h_in[i], val);
    }
}

template<typename T>
void ConstantCCheck(T value) {
    if (noDoubleTests<T>()) return;

    const int num = 1000;
    typedef typename af::dtype_traits<T>::base_type BT;
    BT val = ::real(value);
    dtype dty = (dtype) dtype_traits<T>::af_type;
    af_array out;
    dim_t dim[] = {(dim_t)num};
    ASSERT_EQ(AF_SUCCESS, af_constant(&out, val, 1, dim, dty));

    vector<T> h_in(num);
    af_get_data_ptr(&h_in.front(), out);

    for (int i = 0; i < num; i++) {
        ASSERT_EQ(::real(h_in[i]), val);
    }
}

template<typename T>
void IdentityCPPCheck() {
    if (noDoubleTests<T>()) return;

    int num = 1000;
    dtype dty = (dtype) dtype_traits<T>::af_type;
    array out = af::identity(num, num, dty);

    vector<T> h_in(num*num);
    out.host(&h_in.front());

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < num; j++) {
            if(j == i)
                ASSERT_EQ(h_in[i * num + j], T(1));
            else
                ASSERT_EQ(h_in[i * num + j], T(0));
        }
    }

    num = 100;
    out = af::identity(num, num, num, dty);

    h_in.resize(num*num*num);
    out.host(&h_in.front());

    for (int h = 0; h < num; h++) {
       for (int i = 0; i < num; i++) {
           for (int j = 0; j < num; j++) {
               if(j == i)
                   ASSERT_EQ(h_in[i * num + j], T(1));
               else
                   ASSERT_EQ(h_in[i * num + j], T(0));
           }
       }
    }
}

template<typename T>
void IdentityCCheck() {
    if (noDoubleTests<T>()) return;

    static const int num = 1000;
    dtype dty = (dtype) dtype_traits<T>::af_type;
    af_array out;
    dim_t dim[] = {(dim_t)num, (dim_t)num};
    ASSERT_EQ(AF_SUCCESS, af_identity(&out, 2, dim, dty));

    vector<T> h_in(num*num);
    af_get_data_ptr(&h_in.front(), out);

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < num; j++) {
            if(j == i)
                ASSERT_EQ(h_in[i * num + j], T(1));
            else
                ASSERT_EQ(h_in[i * num + j], T(0));
        }
    }
}

template<typename T>
void IdentityCPPError() {
    if (noDoubleTests<T>()) return;

    static const int num = 1000;
    dtype dty = (dtype) dtype_traits<T>::af_type;
    try {
        array out = af::identity(num, 0, 10, dty);
    }
    catch(const af::exception &ex) {
        SUCCEED();
        return;
    }
    FAIL() << "Failed to throw an exception";
}

TYPED_TEST(Constant, basicCPP)
{
    ConstantCPPCheck<TypeParam>(5);
}

TYPED_TEST(Constant, basicC)
{
    ConstantCCheck<TypeParam>(5);
}

TYPED_TEST(Constant, IdentityC)
{
    IdentityCCheck<TypeParam>();
}

TYPED_TEST(Constant, IdentityCPP)
{
    IdentityCPPCheck<TypeParam>();
}

TYPED_TEST(Constant, IdentityCPPError)
{
    IdentityCPPError<TypeParam>();
}
