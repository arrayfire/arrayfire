#include <cl.hpp>
#include <traits.hpp>
#include <program.hpp>
#include <kernel_headers/KParam.hpp>
#include <platform.hpp>
#include <debug_opencl.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    void buildProgram(cl::Program &prog, const char *ker_str, int ker_len, std::string options)
    {
        Program::Sources setSrc;
        setSrc.emplace_back(KParam_hpp, KParam_hpp_len);
        setSrc.emplace_back(ker_str, ker_len);

        static std::string defaults =
            std::string(" -D dim_type=") +
            std::string(dtype_traits<dim_type>::getName());

        try {

            prog = cl::Program(getContext(), setSrc);
            prog.build((defaults + options).c_str());

        } catch (...) {

            SHOW_BUILD_INFO(prog);
            throw;
        }

        return;
    }
}

