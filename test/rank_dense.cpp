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
using af::cfloat;
using af::cdouble;

template<typename T>
class Rank : public ::testing::Test
{
};

template<typename T>
class Det : public ::testing::Test
{
};

typedef ::testing::Types<float, double, af::cfloat, af::cdouble> TestTypes;
TYPED_TEST_CASE(Rank, TestTypes);
TYPED_TEST_CASE(Det, TestTypes);

template<typename T>
void rankSmall()
{
    if (noDoubleTests<T>()) return;
    if (noLAPACKTests()) return;

    T ha[] = {1, 4, 7, 2, 5, 8, 3, 6, 20};
    af::array a(3, 3, ha);

    ASSERT_EQ(3, (int)af::rank(a));
}

template<typename T>
void rankBig(const int num)
{
    if (noDoubleTests<T>()) return;
    if (noLAPACKTests()) return;

    af::dtype dt = (af::dtype)af::dtype_traits<T>::af_type;
    af::array a = af::randu(num, num, dt);
    ASSERT_EQ(num, (int)af::rank(a));

    af::array b = af::randu(num, num/2, dt);
    ASSERT_EQ(num/2, (int)af::rank(b));
    ASSERT_EQ(num/2, (int)af::rank(transpose(b)));
}

template<typename T>
void rankLow(const int num)
{
    if (noDoubleTests<T>()) return;
    if (noLAPACKTests()) return;

    af::dtype dt = (af::dtype)af::dtype_traits<T>::af_type;

    af::array a = af::randu(3 * num, num, dt);
    af::array b = af::randu(3 * num, num, dt);
    af::array c = a + 0.2 * b;
    af::array in = join(1, a, b, c);

    // The last third is just a linear combination of first and second thirds
    ASSERT_EQ(2 * num, (int)af::rank(in));
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

    af::dtype dt = (af::dtype)af::dtype_traits<T>::af_type;

    vector<af::dim4> numDims;

    vector<vector<float> >   in;
    vector<vector<float> >   tests;
    readTests<float,float,float>(string(TEST_DIR"/lapack/detSmall.test"),numDims,in,tests);
    af::dim4 dims       = numDims[0];

    af::array input = af::array(dims, &(in[0].front())).as(dt);
    T output = af::det<T>(input);

    ASSERT_NEAR(abs((T)tests[0][0]), abs(output), 1e-6);
}

TYPED_TEST(Det, Small)
{
    detTest<TypeParam>();
}
