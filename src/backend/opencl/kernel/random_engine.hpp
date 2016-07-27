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
#include <kernel_headers/random_engine_philox.hpp>
#include <kernel_headers/random_engine_threefry.hpp>
#include <kernel_headers/random_engine_write.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>
#include <dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <program.hpp>
#include <type_util.hpp>
#include <cache.hpp>

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
        static Kernel get_random_engine_kernel(const af_random_type type, const int kerIdx, const uint elementsPerBlock)
        {
            using std::string;
            using std::to_string;
            string engineName;
            const char *ker_strs[2];
            int ker_lens[2];
            ker_strs[0] = random_engine_write_cl;
            ker_lens[0] = random_engine_write_cl_len;
            switch (type) {
                case AF_RANDOM_PHILOX   : engineName = "Philox";
                                        ker_strs[1] = random_engine_philox_cl;
                                        ker_lens[1] = random_engine_philox_cl_len;
                                        break;
                case AF_RANDOM_THREEFRY : engineName = "Threefry";
                                        ker_strs[1] = random_engine_threefry_cl;
                                        ker_lens[1] = random_engine_threefry_cl_len;
                                        break;
                default                 : AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
            }

            string ref_name =
                "random_engine_kernel_" + engineName +
                "_" + string(dtype_traits<T>::getName()) +
                "_" + to_string(kerIdx);
            int device = getActiveDeviceId();
            kc_t::iterator idx = kernelCaches[device].find(ref_name);
            kc_entry_t entry;
            if (idx == kernelCaches[device].end()) {
                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D THREADS=" << THREADS
                        << " -D ELEMENTS_PER_BLOCK=" << elementsPerBlock
                        << " -D RAND_DIST=" << kerIdx;
                if (std::is_same<T, double>::value) {
                    options << " -D USE_DOUBLE";
                }
#if defined(OS_MAC) // Because apple is "special"
                options << " -D IS_APPLE"
                        << " -D log10_val=" << std::log(10.0);
#endif
                cl::Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                entry.prog = new Program(prog);
                entry.ker = new Kernel(*entry.prog, "generate");
                kernelCaches[device][ref_name] = entry;
            } else {
                entry = idx->second;
            }

            return entry.ker[kerIdx];
        }

        template <typename T>
        static void randomDistribution(cl::Buffer out, const size_t elements, const af_random_type type, const uintl seed, uintl &counter, int kerIdx)
        {
            try {
                uint elementsPerBlock = THREADS*4*sizeof(uint)/sizeof(T);
                uint groups = divup(elements, elementsPerBlock);

                uint hi = seed>>32;
                uint lo = seed;

                NDRange local(THREADS, 1);
                NDRange global(THREADS * groups, 1);

                Kernel ker = get_random_engine_kernel<T>(type, kerIdx, elementsPerBlock);
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

        template <typename T>
        void uniformDistribution(cl::Buffer out, const size_t elements, const af_random_type type, const uintl seed, uintl &counter)
        {
            randomDistribution<T>(out, elements, type, seed, counter, 0);
        }

        template <typename T>
        void normalDistribution(cl::Buffer out, const size_t elements, const af_random_type type, const uintl seed, uintl &counter)
        {
            randomDistribution<T>(out, elements, type, seed, counter, 1);
        }

    }
}
