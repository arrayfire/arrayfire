#include <kernel_headers/set.hpp>
#include <cl.hpp>
#include <ctx.hpp>

using cl::Buffer;
using cl::Program;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl {
namespace kernel {

//TODO: Build only once for each instance of a kernel.  NOTE: Static objects in
//      different instances of templates are the same.
template<typename T>
void
set(Buffer &ptr, T val, const size_t &elements)
{
    Program::Sources setSrc;
    setSrc.push_back(std::make_pair((const char *)&set_cl[0], size_t( set_cl_len)));
    Program prog(getCtx(0), setSrc);
    string opt = string("-D T=") + af::dtype_traits<T>::getName();
    prog.build(opt.c_str());

    auto setKern = make_kernel<Buffer, T, const unsigned long>(prog, "set");
    setKern(EnqueueArgs(getQueue(0), NDRange(elements)), ptr, val, elements);
}

}
}
