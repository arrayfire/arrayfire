#include <cl.hpp>
#include <string>

namespace opencl
{
    void buildProgram(cl::Program &prog,
                      const char *ker_str, const int ker_len, std::string options);

    void buildProgram(cl::Program &prog,
                      const int num_files,
                      const char **ker_str, const int *ker_len, std::string options);
}
