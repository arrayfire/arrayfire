/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/ #include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <testHelpers.hpp>
#include <af/internal.h>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using af::cfloat;
using af::cdouble;

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

TEST(MemCopy, MemCopyLaunchConfigOptimization)
{
    const int nx = 1024;
    const int ny = 1024 * 1024;

    cleanSlate();
    {
        af::array in = af::randu(nx, ny, u8);

        // Optimization by doing more work per thread/block - only fermi
        // There are enough blocks on kepler etc
        af::array cp1 = in(af::seq(2), af::span).copy();

        // Optimization by combining threads.x * threads.y as dim0 = 1
        af::array cp0 = in(af::seq(1), af::span).copy();

    }

    {
        af::array in = af::randu(ny, nx, u8);

        // Optimization by doing more work per thread/block
        af::array cp1 = in(af::span, af::seq(2)).copy();

        // Optimization by combining threads.x * threads.y as dim1 = 1
        af::array cp0 = in(af::span, af::seq(0)).copy();
    }
}

TEST(Join, JoinLaunchConfigOptimization)
{
    const int nx = 32;
    const int ny = 4 * 1024 * 1024;

    af::deviceGC();
    {
        af::array in = af::randu(nx, ny, u8);
        af::array joined = af::join(0, in, in);
    }
}
