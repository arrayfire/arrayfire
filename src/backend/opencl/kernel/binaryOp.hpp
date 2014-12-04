/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel_headers/binaryOp.hpp>
#include <cl.hpp>
#include <platform.hpp>
#include <sstream>
#include <string>
#include <mutex>
#include <map>
#include <traits.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;
using std::declval;

namespace opencl
{
namespace kernel
{

    template<typename OP> std::string stringOp() { return std::string("+"); }

//TODO: Build only once for each instance of a kernel.  NOTE: Static objects in
//      different instances of templates are the same.
template<typename R, typename T, typename U, typename OP>
void
binaryOp(Buffer out, const Buffer lhs, const Buffer rhs, const size_t elements)
{
    static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
    static std::map<int, Program*>  bopProgs;
    static std::map<int, Kernel*> bopKernels;

    int device = getActiveDeviceId();

    std::call_once( compileFlags[device], [device] () {
                Program::Sources setSrc;
                setSrc.emplace_back(binaryOp_cl, binaryOp_cl_len);

                bopProgs[device] = new Program(getContext(), setSrc);

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                << " -D U=" << dtype_traits<U>::getName()
                << " -D R=" << dtype_traits<R>::getName()
                << " -D OP=" << stringOp<OP>();
                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                bopProgs[device]->build(options.str().c_str());

                bopKernels[device] = new Kernel(*bopProgs[device], "binaryOp");
            });

    auto binOp = make_kernel<Buffer, Buffer, Buffer, const unsigned long>(*bopKernels[device]);
    binOp(EnqueueArgs(getQueue(), NDRange(elements)), out, lhs, rhs, elements);
}

}
}
