#include "config.hpp"
namespace opencl
{
namespace kernel
{

    std::ostream&
    operator<<(std::ostream &out, const cfloat& var)
    {
        out << "{" << var.s[0] << "," << var.s[1] << "}";
        return out;
    }

    std::ostream&
    operator<<(std::ostream &out, const cdouble& var)
    {
        out << "{" << var.s[0] << "," << var.s[1] << "}";
        return out;
    }
}
}
