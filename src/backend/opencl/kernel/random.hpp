#pragma once

#include <af/defines.h>
#include <kernel_headers/random.hpp>
#include <cl.hpp>
#include <ctx.hpp>
#include <traits.hpp>
#include <helper.hpp>
#include <sstream>
#include <string>
#include <iostream>

#define divup(a, b) (((a)+(b)-1) / (b))

using cl::Buffer;
using cl::Program;
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
        static uint random_seed[2];

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
        struct random_name<double, true>
        {
            const char *name()
            {
                return "randu";
            }
        };

        template<typename T, bool isRandu>
        void random(cl::Buffer out, dim_type elements)
        {
            static unsigned counter;

            Program::Sources src;
            src.emplace_back(random_cl, random_cl_len);
            Program prog(getCtx(0), src);

            std::ostringstream options;
            options << " -D T=" << dtype_traits<T>::getName()
                    << " -D repeat="<< REPEAT
                    << " -D "<< random_name<T, isRandu>().name();
            prog.build(options.str().c_str());

            auto randomOp = make_kernel<cl::Buffer, uint, uint, uint, uint>(prog, "random");

            uint groups = divup(elements, THREADS * REPEAT);
            counter += divup(elements, THREADS * groups);

            NDRange local(THREADS, 1);
            NDRange global(THREADS * groups, 1);

            randomOp(EnqueueArgs(getQueue(0), global, local),
                     out, elements, counter, random_seed[0], random_seed[1]);
        }
    }
}
