/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <string>
#include <vector>
#include <stdexcept>
#include <testHelpers.hpp>

using std::string;
using std::vector;

TEST(fft2, CPP_4D)
{
    af::array a = af::randu(1024, 1024, 32);
    af::array b = af::fft2(a);

    af::array A = af::moddims(a, 1024, 1024, 4, 8);
    af::array B = af::fft2(A);

    af::cfloat *h_b = b.host<af::cfloat>();
    af::cfloat *h_B = B.host<af::cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_B[i]) << "at: " << i << std::endl;
    }

    delete[] h_b;
    delete[] h_B;
}

TEST(ifft2, CPP_4D)
{
    af::array a = af::randu(1024, 1024, 32, c32);
    af::array b = af::ifft2(a);

    af::array A = af::moddims(a, 1024, 1024, 4, 8);
    af::array B = af::ifft2(A);

    af::cfloat *h_b = b.host<af::cfloat>();
    af::cfloat *h_B = B.host<af::cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_B[i]) << "at: " << i << std::endl;
    }

    delete[] h_b;
    delete[] h_B;
}
