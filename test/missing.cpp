/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>
#include <af/image.h>
#include <af/lapack.h>
#include <af/random.h>

using namespace af;

TEST(MissingFunctionTests, Dummy) {
    array A = randu(10, 10, f32);
    af_print(A);
    af_print(arg(A));
    af_print(arg(complex(A, A)));
    af_print(trunc(3 * A));
    af_print(factorial(ceil(2 * A)));
    af_print(pow2(A));
    af_print(root(2, A));
    af_print(A - 0.5);
    af_print(sign(A - 0.5));
    af_print(minfilt(A, 3, 3) - erode(A, constant(1, 3, 3)));
    af_print(maxfilt(A, 3, 3) - dilate(A, constant(1, 3, 3)));
    printf("%lf\n", norm(A));
}
