/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/laswp.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <types.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;


namespace opencl
{

namespace kernel
{

static const int NTHREADS = 256;
static const int MAX_PIVOTS =  32;

typedef struct {
    int npivots;
    int ipiv[MAX_PIVOTS];
} zlaswp_params_t;


template<typename T>
void laswp(int n, cl_mem in, size_t offset, int ldda,
           int k1, int k2, const int *ipiv, int inci)
{

    static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
    static std::map<int, Program*>  swpProgs;
    static std::map<int, Kernel*> swpKernels;

    int device = getActiveDeviceId();

    std::call_once(compileFlags[device], [device] () {

            std::ostringstream options;
            options << " -D T=" << dtype_traits<T>::getName()
                    << " -D MAX_PIVOTS=" << MAX_PIVOTS;

            if (std::is_same<T, double>::value ||
                std::is_same<T, cdouble>::value) {
                options << " -D USE_DOUBLE";
            }

            cl::Program prog;
            buildProgram(prog, laswp_cl, laswp_cl_len, options.str());
            swpProgs[device] = new Program(prog);

            swpKernels[device] = new Kernel(*swpProgs[device], "laswp");
        });

    int groups = divup(n, NTHREADS);
    NDRange local(NTHREADS);
    NDRange global(groups * local[0]);
    zlaswp_params_t params;

    auto laswpOp = make_kernel<int, cl_mem, unsigned long long,
                               int, zlaswp_params_t>(*swpKernels[device]);

    for( int k = k1-1; k < k2; k += MAX_PIVOTS ) {

        int pivots_left = k2-k;

        params.npivots = pivots_left > MAX_PIVOTS ? MAX_PIVOTS : pivots_left;

        for( int j = 0; j < params.npivots; ++j ) {
            params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }

        unsigned long long k_offset = offset + k*ldda;

        laswpOp(EnqueueArgs(getQueue(), global, local),
                n, in, k_offset, ldda, params);
    }

}

}
}
