#include <kernel_headers/binaryOp.hpp>
#include <cl.hpp>
#include <ctx.hpp>
#include <sstream>
#include <string>
#include <traits.hpp>

using cl::Buffer;
using cl::Program;
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
    Program::Sources setSrc;
    setSrc.emplace_back(binaryOp_cl, binaryOp_cl_len);
    Program prog(getCtx(0), setSrc);

    std::ostringstream options;
    options << " -D T=" << dtype_traits<T>::getName()
            << " -D U=" << dtype_traits<U>::getName()
            << " -D R=" << dtype_traits<R>::getName()
            << " -D OP=" << stringOp<OP>();
    prog.build(options.str().c_str());

    auto binOp = make_kernel<Buffer, Buffer, Buffer, const unsigned long>(prog, "binaryOp");
    binOp(EnqueueArgs(getQueue(0), NDRange(elements)), out, lhs, rhs, elements);
}

}
}
