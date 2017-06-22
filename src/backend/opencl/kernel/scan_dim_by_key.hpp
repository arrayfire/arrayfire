/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <traits.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
namespace opencl
{
namespace kernel
{
    template<typename Ti, typename Tk, typename To, af_op_t op, bool inclusive_scan>
    void scan_dim(Param &out, const Param &in, const Param &key, int dim);
}
}
