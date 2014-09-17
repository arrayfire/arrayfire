#pragma once
namespace opencl
{
namespace kernel
{

    static std::ostream&
    operator<<(std::ostream &out, const cfloat& var)
    {
        out << "{" << var.s[0] << "," << var.s[1] << "}";
        return out;
    }

    static std::ostream&
    operator<<(std::ostream &out, const cdouble& var)
    {
        out << "{" << var.s[0] << "," << var.s[1] << "}";
        return out;
    }

    static const uint THREADS_PER_GROUP = 256;
    static const uint THREADS_X = 32;
    static const uint THREADS_Y = THREADS_PER_GROUP / THREADS_X;
    static const uint REPEAT    = 32;
}
}
