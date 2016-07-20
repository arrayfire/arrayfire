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
#include <type_util.hpp>
#include <cache.hpp>
#include "names.hpp"
#include "config.hpp"

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

        template <typename T>
        static Kernel get_random_engine_kernel(af_random_type type, std::string distribution, uint elementsPerBlock)
        {
            int kerIdx;
            std::string engineName;
            std::string kernelString;
            if (distribution == std::string("uniformDistribution")) {
                kerIdx = 0;
            } if (distribution == std::string("normalDistribution")) {
                kerIdx = 1;
            }
            switch (type) {
                case AF_RANDOM_PHILOX : engineName = "Philox";
                                        kernelString = std::string(random_engine_write_cl, random_engine_write_cl_len) +
                                        std::string(random_engine_philox_cl, random_engine_philox_cl_len); break;
                case AF_RANDOM_THREEFRY : engineName = "Threefry";
                                        kernelString = std::string(random_engine_write_cl, random_engine_write_cl_len) +
                                        std::string(random_engine_threefry_cl, random_engine_threefry_cl_len); break;
                                        //THROW
            }
            std::string ref_name =
                std::string("random_engine_kernel_") + engineName +
                std::string("_") + std::string(dtype_traits<T>::getName());
            int device = getActiveDeviceId();
            kc_t::iterator idx = kernelCaches[device].find(ref_name);
            kc_entry_t entry;
            if (idx == kernelCaches[device].end()) {
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
                entry.prog = new Program(prog);
                entry.ker = new Kernel[2];
                entry.ker[0] = Kernel(*entry.prog, "uniformDistribution");
                entry.ker[1] = Kernel(*entry.prog, "normalDistribution");
                kernelCaches[device][ref_name] = entry;
            } else {
                entry = idx->second;
            }

            return entry.ker[kerIdx];
        }

        template <typename T>
        static void randomDistribution(af_random_type type, cl::Buffer out, size_t elements, const uintl seed, uintl &counter, std::string distribution)
        {
            try {
                uint elementsPerBlock = THREADS*4*sizeof(uint)/sizeof(T);
                uint groups = divup(elements, elementsPerBlock);

                uint hi = seed>>32;
                uint lo = seed;

                NDRange local(THREADS, 1);
                NDRange global(THREADS * groups, 1);

                Kernel ker = get_random_engine_kernel<T>(type, distribution, elementsPerBlock);
                auto randomEngineOp = KernelFunctor<cl::Buffer, uint, uint, uint, uint>(ker);

                randomEngineOp(EnqueueArgs(getQueue(), global, local),
                        out, elements, counter, hi, lo);

                counter += elements;
                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }

        template <typename T, af_random_type Type>
        void uniformDistribution(cl::Buffer out, size_t elements, const uintl seed, uintl &counter)
        {
            randomDistribution<T>(Type, out, elements, seed, counter, "uniformDistribution");
        }

        template <typename T, af_random_type Type>
        void normalDistribution(cl::Buffer out, size_t elements, const uintl seed, uintl &counter)
        {
            randomDistribution<T>(Type, out, elements, seed, counter, "normalDistribution");
        }

    }
}
