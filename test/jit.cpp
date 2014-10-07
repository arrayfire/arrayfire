#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <af/arith.h>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::af_cfloat;
using af::af_cdouble;

static void test()
{
    af_array lhs, rhs, out, out2, out3;

    dim_type num = 20;
    ASSERT_EQ(AF_SUCCESS, af_constant(&lhs, 1, 1, &num, f32));
    ASSERT_EQ(AF_SUCCESS, af_constant(&rhs, 1, 1, &num, f32));

    af_print(lhs);
    af_print(rhs);

    ASSERT_EQ(AF_SUCCESS, af_add(&out, lhs, rhs));
    ASSERT_EQ(AF_SUCCESS, af_mul(&out2, out, lhs));
    ASSERT_EQ(AF_SUCCESS, af_sub(&out3, out, out2));

    af_print(out3);
}

TEST(jit, simple)
{
    test();
}
