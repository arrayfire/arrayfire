/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>

#include <cstdlib>

using af::array;
using af::constant;
using af::dim4;
using af::end;
using af::fft;
using af::info;
using af::randu;
using af::scan;
using af::seq;
using af::setDevice;
using af::sin;
using af::sort;

template<typename T>
class ArrayDeathTest : public ::testing::Test {};

void deathTest() {
    info();
    setDevice(0);

    array A = randu(5, 3, f32);

    array B = sin(A) + 1.5;

    B(seq(0, 2), 1) = B(seq(0, 2), 1) * -1;

    array C = fft(B);

    array c = C.row(end);

    dim4 dims(16, 4, 1, 1);
    array r = constant(2, dims);

    array S = scan(r, 0, AF_BINARY_MUL);

    float d[] = {1, 2, 3, 4, 5, 6};
    array D(2, 3, d, afHost);

    D.col(0) = D.col(end);

    array vals, inds;
    sort(vals, inds, A);

    _exit(0);
}

TEST(ArrayDeathTest, ProxyMoveAssignmentOperator) {
    EXPECT_EXIT(deathTest(), ::testing::ExitedWithCode(0), "");
}
