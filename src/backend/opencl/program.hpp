#pragma once
#include <cl.hpp>
#include <string>
#include <mutex>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    void buildProgram(cl::Program &prog,
                      const char *ker_str, const int ker_len, std::string options);

    void buildProgram(cl::Program &prog,
                      const int num_files,
                      const char **ker_str, const int *ker_len, std::string options);
}
