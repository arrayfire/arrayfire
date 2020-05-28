/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
#include <iostream>

TEST(Memory, recover) {
    cleanSlate();  // Clean up everything done so far

    try {
        array vec[100];

        // Trying to allocate 1 Terrabyte of memory and trash the memory manager
        // should crash memory manager
        for (int i = 0; i < 1000; i++) {
            vec[i] = randu(1024, 1024, 256);  // Allocating 1GB
        }

        FAIL();
    } catch (exception &ae) {
        ASSERT_EQ(ae.err(), AF_ERR_NO_MEM);

        const int num   = 1000 * 1000;
        const float val = 1.0;

        array a    = constant(val, num);  // This should work as expected
        float *h_a = a.host<float>();
        for (int i = 0; i < 1000 * 1000; i++) { ASSERT_EQ(h_a[i], val); }
        freeHost(h_a);
    }
}
