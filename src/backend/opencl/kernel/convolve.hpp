/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel/convolve/conv_common.hpp>

namespace opencl
{

namespace kernel
{

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
static const int MAX_CONV1_FILTER_LEN = 129;
static const int MAX_CONV2_FILTER_LEN = 17;
static const int MAX_CONV3_FILTER_LEN = 5;

/*
 * convolution kernel wrappers are split to multiple files to
 * facilitate faster compilation times as the template instantiations
 * are too big in number.
 * all conv[1|2|3] functions used below are declared in the hpp
 * file under the folder 'kernel/convovel' with their implementations
 * written in corresponding conv[1|2|3].cpp files under the same folder.
 */
template<typename T, typename accType, int baseDim, bool expand>
void convolve_nd(Param out, const Param signal, const Param filter, ConvolveBatchKind kind)
{
    conv_kparam_t param;

    for (int i=0; i<3; ++i) {
        param.o[i] = 0;
        param.s[i] = 0;
    }
    param.launchMoreBlocks = kind==CONVOLVE_BATCH_SAME || kind==CONVOLVE_BATCH_KERNEL;
    param.outHasNoOffset = kind==CONVOLVE_BATCH_SIGNAL || kind==CONVOLVE_BATCH_NONE;
    param.inHasNoOffset  = kind!=CONVOLVE_BATCH_SAME;

    prepareKernelArgs<T>(param, out.info.dims, filter.info.dims, baseDim);

    switch(baseDim) {
        case 1: conv1<T, accType, expand>(param, out, signal, filter); break;
        case 2: conv2<T, accType, expand>(param, out, signal, filter); break;
        case 3: conv3<T, accType, expand>(param, out, signal, filter); break;
    }

    CL_DEBUG_FINISH(getQueue());
    bufferFree(param.impulse);
}

}

}
