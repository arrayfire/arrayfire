#include <kernel_headers/binaryOp.hpp>
#include <cl.hpp>
#include <platform.hpp>
#include <sstream>
#include <string>
#include <mutex>
#include <traits.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;
using std::declval;

namespace opencl
{
namespace kernel
{

    template<typename OP> std::string stringOp() { return std::string("+"); }

//TODO: Build only once for each instance of a kernel.  NOTE: Static objects in
//      different instances of templates are the same.
template<typename R, typename T, typename U, typename OP>
void
binaryOp(Buffer out, const Buffer lhs, const Buffer rhs, const size_t elements)
{
    static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
    static Program            bopProgs[DeviceManager::MAX_DEVICES];
    static Kernel           bopKernels[DeviceManager::MAX_DEVICES];

    int device = getActiveDeviceId();

    std::call_once( compileFlags[device], [device] () {
                Program::Sources setSrc;
                setSrc.emplace_back(binaryOp_cl, binaryOp_cl_len);

                bopProgs[device] = Program(getContext(), setSrc);

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                << " -D U=" << dtype_traits<U>::getName()
                << " -D R=" << dtype_traits<R>::getName()
                << " -D OP=" << stringOp<OP>();
                bopProgs[device].build(options.str().c_str());

                bopKernels[device] = Kernel(bopProgs[device], "binaryOp");
            });

    auto binOp = make_kernel<Buffer, Buffer, Buffer, const unsigned long>(bopKernels[device]);
    binOp(EnqueueArgs(getQueue(), NDRange(elements)), out, lhs, rhs, elements);
}

}
}
