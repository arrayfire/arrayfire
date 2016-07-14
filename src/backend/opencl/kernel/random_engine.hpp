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
//#include <kernel_headers/random.hpp>
#include <kernel_headers/random_engine_philox.hpp>
#include <kernel_headers/random_engine_threefry.hpp>
#include <kernel_headers/random_engine_write.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <program.hpp>

#include <iostream>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    namespace kernel
    {
        static const uint THREADS = 256;

        template <typename T, af_random_type Type>
        void uniformDistribution(cl::Buffer out, size_t elements, const uintl seed, uintl &counter)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>  ranProgs;
                static std::map<int, Kernel*>   ranKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                        std::string kernelString;
                        switch (Type) {
                            case AF_RANDOM_PHILOX : kernelString = std::string(random_engine_write_cl, random_engine_write_cl_len) +
                                                    std::string(random_engine_philox_cl, random_engine_philox_cl_len); break;
                            case AF_RANDOM_THREEFRY : kernelString = std::string(random_engine_write_cl, random_engine_write_cl_len) +
                                                    std::string(random_engine_threefry_cl, random_engine_threefry_cl_len); break;
                                                    //THROW
                        }
                        uint elementsPerBlock = THREADS*4*sizeof(uint)/sizeof(T);

                        Program::Sources setSrc;
                        setSrc.emplace_back(kernelString.c_str(), kernelString.length());

                        std::ostringstream options;
                        options << " -D T=" << dtype_traits<T>::getName()
                                << " -D THREADS=" << THREADS
                                << " -D ELEMENTS_PER_BLOCK=" << elementsPerBlock;
#if defined(OS_MAC) // Because apple is "special"
                        options << " -D IS_APPLE"
                                << " -D log10_val=" << std::log(10.0);
#endif

                        cl::Program prog;
                        buildProgram(prog, kernelString.c_str(), kernelString.length(), options.str());
                        ranProgs[device] = new Program(prog);
                        ranKernels[device] = new Kernel(*ranProgs[device], "uniformDistribution");
                    });

                    auto randomEngineOp = KernelFunctor<cl::Buffer, uint, uint, uint, uint>(*ranKernels[device]);

                    uint elementsPerBlock = THREADS*4*sizeof(uint)/sizeof(T);
                    uint groups = divup(elements, elementsPerBlock);
                    counter += elements;

                    NDRange local(THREADS, 1);
                    NDRange global(THREADS * groups, 1);

                    uint hi = seed>>32;
                    uint lo = seed;

                    randomEngineOp(EnqueueArgs(getQueue(), global, local),
                            out, elements, counter, lo, hi);
                    CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error ex) {
                CL_TO_AF_ERROR(ex);
            }
        }
    }
}
