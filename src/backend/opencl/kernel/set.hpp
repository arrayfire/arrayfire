/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel_headers/set.hpp>
#include <cl.hpp>
#include <platform.hpp>
#include <traits.hpp>
#include <mutex>
#include <backend.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl {
namespace kernel {

//TODO: Build only once for each instance of a kernel.  NOTE: Static objects in
//      different instances of templates are the same.
template<typename T>
void
set(Buffer &ptr, T val, const size_t &elements)
{
    static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
    static Program            setProgs[DeviceManager::MAX_DEVICES];
    static Kernel           setKernels[DeviceManager::MAX_DEVICES];

    int device = getActiveDeviceId();

    std::call_once( compileFlags[device], [device] () {
                Program::Sources setSrc;
                setSrc.emplace_back(set_cl, set_cl_len);

                setProgs[device] = Program(getContext(), setSrc);

                string opt = string("-D T=") + dtype_traits<T>::getName();
                setProgs[device].build(opt.c_str());

                setKernels[device] = Kernel(setProgs[device], "set");
            });

    auto setKern = make_kernel<Buffer, T, const unsigned long>(setKernels[device]);
    setKern(EnqueueArgs(getQueue(), NDRange(elements)), ptr, val, elements);
}

}
}
