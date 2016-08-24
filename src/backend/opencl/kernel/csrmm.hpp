/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#pragma once
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <cache.hpp>
#include <type_util.hpp>
#include "scan_dim.hpp"
#include "reduce.hpp"
#include "scan_first.hpp"
#include "config.hpp"

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    namespace kernel
    {
        template<typename T>
        void csrmm_nt(Param out,
                      const Param &values, const Param &rowIdx, const Param &colIdx,
                      const Param &rhs, const T alpha, const T beta)
        {
        }
    }
}
