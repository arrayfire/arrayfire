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
#include <cache.hpp>
#include <common/complex.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/resize.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>

namespace opencl {
namespace kernel {
static const int RESIZE_TX = 16;
static const int RESIZE_TY = 16;

template<typename T>
using wtype_t = typename std::conditional<std::is_same<T, double>::value,
                                          double, float>::type;

template<typename T>
using vtype_t = typename std::conditional<common::is_complex<T>::value, T,
                                          wtype_t<T>>::type;

template<typename T, af_interp_type method>
void resize(Param out, const Param in) {
    typedef typename dtype_traits<T>::base_type BT;

    std::string refName = std::string("reorder_kernel_") +
                          std::string(dtype_traits<T>::getName()) +
                          std::to_string(method);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        options << " -D VT=" << dtype_traits<vtype_t<T>>::getName();
        options << " -D WT=" << dtype_traits<wtype_t<BT>>::getName();

        switch (method) {
            case AF_INTERP_NEAREST: options << " -D INTERP=NEAREST"; break;
            case AF_INTERP_BILINEAR: options << " -D INTERP=BILINEAR"; break;
            case AF_INTERP_LOWER: options << " -D INTERP=LOWER"; break;
            default: break;
        }

        if (static_cast<af_dtype>(dtype_traits<T>::af_type) == c32 ||
            static_cast<af_dtype>(dtype_traits<T>::af_type) == c64) {
            options << " -D CPLX=1";
            options << " -D TB=" << dtype_traits<BT>::getName();
        } else {
            options << " -D CPLX=0";
        }

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value)
            options << " -D USE_DOUBLE";

        const char* ker_strs[] = {resize_cl};
        const int ker_lens[]   = {resize_cl_len};
        cl::Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new cl::Program(prog);
        entry.ker  = new cl::Kernel(*entry.prog, "resize_kernel");

        addKernelToCache(device, refName, entry);
    }

    auto resizeOp =
        cl::KernelFunctor<cl::Buffer, const KParam, const cl::Buffer,
                          const KParam, const int, const int, const float,
                          const float>(*entry.ker);

    cl::NDRange local(RESIZE_TX, RESIZE_TY, 1);

    int blocksPerMatX = divup(out.info.dims[0], local[0]);
    int blocksPerMatY = divup(out.info.dims[1], local[1]);
    cl::NDRange global(local[0] * blocksPerMatX * in.info.dims[2],
                       local[1] * blocksPerMatY * in.info.dims[3], 1);

    double xd = (double)in.info.dims[0] / (double)out.info.dims[0];
    double yd = (double)in.info.dims[1] / (double)out.info.dims[1];

    float xf = (float)xd, yf = (float)yd;

    resizeOp(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *in.data, in.info, blocksPerMatX, blocksPerMatY, xf, yf);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
