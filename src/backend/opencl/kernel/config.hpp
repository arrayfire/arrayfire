#pragma once
#include <iostream>
#include <types.hpp>

namespace opencl
{
namespace kernel
{

    std::ostream&
    operator<<(std::ostream &out, const cfloat& var);

    std::ostream&
    operator<<(std::ostream &out, const cdouble& var);

    static const uint THREADS_PER_GROUP = 256;
    static const uint THREADS_X = 32;
    static const uint THREADS_Y = THREADS_PER_GROUP / THREADS_X;
    static const uint REPEAT    = 32;
}
}
