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
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;

static void cleanSlate()
{
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    af::deviceGC();

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 0u);
    ASSERT_EQ(lock_buffers, 0u);
    ASSERT_EQ(alloc_bytes, 0u);
    ASSERT_EQ(lock_bytes, 0u);
}

TEST(Memory, recover)
{
    cleanSlate(); // Clean up everything done so far

    try {
        af::array vec[100];

        // Trying to allocate 1 Terrabyte of memory and trash the memory manager
        // should crash memory manager
        for (int i = 0; i < 1000; i++) {
            vec[i] = af::randu(1024, 1024, 256); //Allocating 1GB
        }

        ASSERT_EQ(true, false); //Is there a simple assert statement?
    } catch (af::exception &ae) {

        ASSERT_EQ(ae.err(), AF_ERR_NO_MEM);

        const int num = 1000 * 1000;
        const float val = 1.0;

        af::array a = af::constant(val, num); // This should work as expected
        float *h_a = a.host<float>();
        for (int i = 0; i < 1000 * 1000; i++) {
            ASSERT_EQ(h_a[i], val);
        }
        delete[] h_a;
    }

}
