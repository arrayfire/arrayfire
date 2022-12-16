/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>

namespace arrayfire {
namespace opencl {
namespace kernel {
template<typename Ti, typename Tk, typename To, af_op_t op>
void scanFirstByKey(Param &out, const Param &in, const Param &key,
                    const bool inclusive_scan);
}
}  // namespace opencl
}  // namespace arrayfire
