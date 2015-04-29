/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <ostream>
#include <backend.hpp>

namespace cuda
{
    static std::ostream&
    operator<<(std::ostream &out, const cfloat& var)
    {
        out << "(" << var.x << "," << var.y << ")";
        return out;
    }

    static std::ostream&
    operator<<(std::ostream &out, const cdouble& var)
    {
        out << "(" << var.x << "," << var.y << ")";
        return out;
    }
}
