/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <platform.hpp>
#include <kernel_headers/set.hpp>
#include <traits.hpp>
#include <mutex>
#include <map>
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
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>   setProgs;
        static std::map<int, Kernel *> setKernels;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                    Program::Sources setSrc;
                    setSrc.emplace_back(set_cl, set_cl_len);

                    setProgs[device] = new Program(getContext(), setSrc);

                    string opt = string("-D T=") + dtype_traits<T>::getName();
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        opt << " -D USE_DOUBLE";
                    }
                    setProgs[device]->build(opt.c_str());

                    setKernels[device] = new Kernel(*setProgs[device], "set");
                });

        auto setKern = make_kernel<Buffer, T, const unsigned long>(setKernels[device]);
        setKern(EnqueueArgs(getQueue(), NDRange(elements)), ptr, val, elements);
        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error &err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}
}
