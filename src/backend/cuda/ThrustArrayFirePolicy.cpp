/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <ThrustArrayFirePolicy.hpp>

namespace cuda {

cudaStream_t get_stream(ThrustArrayFirePolicy /*unused*/) {
    return getActiveStream();
}

cudaError_t synchronize_stream(ThrustArrayFirePolicy /*unused*/) {
    return cudaStreamSynchronize(getActiveStream());
}

}  // namespace cuda
