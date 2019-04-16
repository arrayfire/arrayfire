/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <stdexcept>
#include <string>
#include <vector>

using af::array;
using af::cfloat;
using af::fft2;
using af::ifft2;
using af::moddims;
using af::randu;
using std::endl;
using std::string;
using std::vector;

TEST(fft2, CPP_4D) {
    array a = randu(1024, 1024, 32);
    array b = fft2(a);

    array A = moddims(a, 1024, 1024, 4, 8);
    array B = fft2(A);

    cfloat *h_b = b.host<cfloat>();
    cfloat *h_B = B.host<cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_B[i]) << "at: " << i << endl;
    }

    af_free_host(h_b);
    af_free_host(h_B);
}

TEST(ifft2, CPP_4D) {
    array a = randu(1024, 1024, 32, c32);
    array b = ifft2(a);

    array A = moddims(a, 1024, 1024, 4, 8);
    array B = ifft2(A);

    cfloat *h_b = b.host<cfloat>();
    cfloat *h_B = B.host<cfloat>();

    for (int i = 0; i < (int)a.elements(); i++) {
        ASSERT_EQ(h_b[i], h_B[i]) << "at: " << i << endl;
    }

    af_free_host(h_b);
    af_free_host(h_B);
}
