#pragma once
#include <iostream>
#include <backend.hpp>

namespace cuda
{
    static std::ostream&
    operator<<(std::ostream &out, const cfloat& var)
    {
        out << var.x << " " << var.y << "i";
        return out;
    }

    static std::ostream&
    operator<<(std::ostream &out, const cdouble& var)
    {
        out << var.x << " " << var.y << "i";
        return out;
    }
}
