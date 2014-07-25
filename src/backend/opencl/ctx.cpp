#include <ctx.hpp>
#include <cl.hpp>
#include <vector>

namespace opencl
{

using std::vector;
using cl::Platform;
using cl::Device;

void ctxCB(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    printf("Context Error: %s\n", errinfo); fflush(stdout);
}

const cl::Context&
getCtx(unsigned char idx)
{
    static std::vector<cl::Platform> platforms(0);
    static std::vector<cl::Context> contexts(0);
    if(contexts.empty()) {
        Platform::get(&platforms);
        for(auto platform : platforms) {
            vector<cl_context_properties> prop = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
            contexts.emplace_back(CL_DEVICE_TYPE_CPU, &prop.front(), ctxCB);
        }
    }
    return contexts[idx];
}

cl::CommandQueue&
getQueue(unsigned char idx) {
    static std::vector<cl::CommandQueue> queues;
    if(queues.empty()) {
        queues.emplace_back(getCtx(0));
    }
    return queues[idx];
}

}
