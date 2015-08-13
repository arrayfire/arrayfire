/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <platform.hpp>
#include <af/defines.h>
#include <kernel_headers/random.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <program.hpp>

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
        static const uint REPEAT  = 32;
        static const uint THREADS = 256;

        static uint random_seed[2] = {0, 0};
        static unsigned counter;

        template<typename T, bool isRandu>
        struct random_name
        {
            const char *name()
            {
                return "randi";
            }
        };

        template<typename T>
        struct random_name<T, false>
        {
            const char *name()
            {
                return "randn";
            }
        };

        template<>
        struct random_name<float, true>
        {
            const char *name()
            {
                return "randu";
            }
        };

        template<>
        struct random_name<char, true>
        {
            const char *name()
            {
                return "randu";
            }
        };

        template<>
        struct random_name<double, true>
        {
            const char *name()
            {
                return "randu";
            }
        };

        template<typename> static bool isDouble() { return false; }
        template<> STATIC_ bool isDouble<double>() { return true; }
        template<> STATIC_ bool isDouble<cdouble>() { return true; }

        template<typename T, bool isRandu>
        void random(cl::Buffer out, int elements)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>  ranProgs;
                static std::map<int, Kernel*>   ranKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                        Program::Sources setSrc;
                        setSrc.emplace_back(random_cl, random_cl_len);

                        std::ostringstream options;
                        options << " -D T=" << dtype_traits<T>::getName()
                                << " -D repeat="<< REPEAT
#if defined(OS_MAC) // Because apple is "special"
                                << " -D IS_APPLE"
                                << " -D log10_val=" << std::log(10.0)
#endif
                                << " -D " << random_name<T, isRandu>().name();

                        if (std::is_same<T, double>::value) {
                            options << " -D USE_DOUBLE";
                            options << " -D IS_64";
                        }

                        if (std::is_same<T, char>::value) {
                            options << " -D IS_BOOL";
                        }

                        cl::Program prog;
                        buildProgram(prog, random_cl, random_cl_len, options.str());
                        ranProgs[device] = new Program(prog);
                        ranKernels[device] = new Kernel(*ranProgs[device], "random");
                    });

                auto randomOp = make_kernel<cl::Buffer, uint, uint, uint, uint>(*ranKernels[device]);

                uint groups = divup(elements, THREADS * REPEAT);
                counter += divup(elements, THREADS * groups);

                NDRange local(THREADS, 1);
                NDRange global(THREADS * groups, 1);

                randomOp(EnqueueArgs(getQueue(), global, local),
                         out, elements, counter, random_seed[0], random_seed[1]);
                CL_DEBUG_FINISH(getQueue());
            } catch(cl::Error ex) {
                CL_TO_AF_ERROR(ex);
            }
        }
    }
}
