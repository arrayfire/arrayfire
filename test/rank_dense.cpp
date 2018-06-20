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
#include <af/defines.h>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::abs;
using af::array;
using af::cfloat;
using af::cdouble;
using af::det;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::join;
using af::randu;

template<typename T>
class Rank : public ::testing::Test
{
};

template<typename T>
class Det : public ::testing::Test
{
};

typedef ::testing::Types<float, double, cfloat, cdouble> TestTypes;
TYPED_TEST_CASE(Rank, TestTypes);
TYPED_TEST_CASE(Det, TestTypes);

template<typename T>
void rankSmall()
{
    if (noDoubleTests<T>()) return;
    if (noLAPACKTests()) return;

    T ha[] = {1, 4, 7, 2, 5, 8, 3, 6, 20};
    array a(3, 3, ha);

    ASSERT_EQ(3, (int)rank(a));
}

template<typename T>
void rankBig(const int num)
{
    if (noDoubleTests<T>()) return;
    if (noLAPACKTests()) return;

    dtype dt = (dtype)dtype_traits<T>::af_type;
    array a = randu(num, num, dt);
    ASSERT_EQ(num, (int)rank(a));

    array b = randu(num, num/2, dt);
    ASSERT_EQ(num/2, (int)rank(b));
    ASSERT_EQ(num/2, (int)rank(transpose(b)));
}

template<typename T>
void rankLow(const int num)
{
    if (noDoubleTests<T>()) return;
    if (noLAPACKTests()) return;

    dtype dt = (dtype)dtype_traits<T>::af_type;

    array a = randu(3 * num, num, dt);
    array b = randu(3 * num, num, dt);
    array c = a + 0.2 * b;
    array in = join(1, a, b, c);

    // The last third is just a linear combination of first and second thirds
    ASSERT_EQ(2 * num, (int)rank(in));
}

TYPED_TEST(Rank, small)
{
    rankSmall<TypeParam>();
}

TYPED_TEST(Rank, big)
{
    rankBig<TypeParam>(1024);
}

TYPED_TEST(Rank, low)
{
    rankBig<TypeParam>(512);
}

template<typename T>
void detTest()
{
    if (noDoubleTests<T>()) return;
    if (noLAPACKTests()) return;

    dtype dt = (dtype)dtype_traits<T>::af_type;

    vector<dim4> numDims;

    vector<vector<float> >   in;
    vector<vector<float> >   tests;
    readTests<float,float,float>(string(TEST_DIR"/lapack/detSmall.test"),numDims,in,tests);
    dim4 dims       = numDims[0];

    array input = array(dims, &(in[0].front())).as(dt);
    T output = det<T>(input);

    ASSERT_NEAR(abs((T)tests[0][0]), abs(output), 1e-6);
}

TYPED_TEST(Det, Small)
{
    detTest<TypeParam>();
}
