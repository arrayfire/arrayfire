/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

namespace arrayfire {
namespace cuda {

typedef struct {
    int offs[4];
    int strds[4];
    int steps[4];
    bool isSeq[4];
    unsigned int* ptr[4];
} AssignKernelParam;

using IndexKernelParam = AssignKernelParam;

}  // namespace cuda
}  // namespace arrayfire
