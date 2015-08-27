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

const size_t step_bytes = 1024;

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

    af::setMemStepSize(step_bytes);

    ASSERT_EQ(af::getMemStepSize(), step_bytes);
}

// This test should be by itself as it leaks memory intentionally
TEST(Memory, lock)
{

    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate(); // Clean up everything done so far

    const dim_t num = step_bytes / sizeof(float);

    std::vector<float> in(num);

    af_array arr = 0;
    ASSERT_EQ(AF_SUCCESS, af_create_array(&arr, &in[0], 1, &num, f32));

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, step_bytes);
    ASSERT_EQ(lock_bytes, step_bytes);

    // arr1 gets released by end of the following code block
    {
        af::array a(arr);
        a.lock();

        // No new memory should be allocated
        af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                          &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);
        ASSERT_EQ(alloc_bytes, step_bytes);
        ASSERT_EQ(lock_bytes, step_bytes);
    }

    // Making sure all unlocked buffers are freed
    af::deviceGC();

    af::deviceMemInfo(&alloc_bytes, &alloc_buffers,
                      &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, step_bytes);
    ASSERT_EQ(lock_bytes, step_bytes);
}
