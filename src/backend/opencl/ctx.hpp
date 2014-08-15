
namespace cl
{
class Context;
class CommandQueue;
}

namespace opencl
{

const cl::Context& getCtx(unsigned char idx);
cl::CommandQueue& getQueue(unsigned char idx);

}
