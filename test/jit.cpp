#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <af/arith.h>
#include <af/reduce.h>
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

#define AF(fn) do {                             \
        fn;                                     \
    } while(0)

static void test()
{
    af_array x, y;
    af_array x2, y2;
    af_array s, r, R, o, d, D;
    af_array out;

    dim_type num = 10;
    double scale = 4.0 / (double)(num);

    AF(af_constant(&x, 1.0, 1, &num, f32));
    AF(af_constant(&y, 1.0, 1, &num, f32));

    AF(af_add(&r, x, y));
    AF(af_add(&R, x, r));
    AF(af_print(r));
    AF(af_print(R));
}

TEST(jit, simple)
{
    test();
}
