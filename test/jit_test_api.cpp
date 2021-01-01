/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/data.h>

namespace af {
int getMaxJitLen(void);

void setMaxJitLen(const int jitLen);
}  // namespace af

TEST(JIT, UnitMaxHeight) {
    const int oldMaxJitLen = af::getMaxJitLen();
    af::setMaxJitLen(1);
    af::array a = af::constant(1, 10);
    af::array b = af::constant(2, 10);
    af::array c = a * b;
    af::array d = b * c;
    c.eval();
    d.eval();
    af::setMaxJitLen(oldMaxJitLen);
}

TEST(JIT, ZeroMaxHeight) {
    EXPECT_THROW({ af::setMaxJitLen(0); }, af::exception);
}
