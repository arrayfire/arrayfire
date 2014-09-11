#include <cl.hpp>
#include <string>

namespace opencl
{
    void buildProgram(cl::Program &prog, const char *ker_str, int ker_len, std::string options);
}
