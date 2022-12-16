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
#include <optypes.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {
template<typename Ti, typename Tk, typename To, af_op_t op>
void scan_first_by_key(Param<To> out, CParam<Ti> in, CParam<Tk> key,
                       bool inclusive_scan);
}
}  // namespace cuda
}  // namespace arrayfire
