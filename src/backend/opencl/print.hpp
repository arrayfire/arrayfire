#pragma once
#include <backend.hpp>
#include <iostream>

namespace opencl
{
    static std::ostream&
    operator<<(std::ostream &out, const cfloat& var)
    {
        out << var.s[0] << " " << var.s[1] << "i";
        return out;
    }

    static std::ostream&
    operator<<(std::ostream &out, const cdouble& var)
    {
        out << var.s[0] << " " << var.s[1] << "i";
        return out;
    }
}
