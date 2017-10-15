/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <program.hpp>
#include <map>
#include <string>

namespace cl {
  class Program;
  class Kernel;
}

namespace opencl
{
    typedef struct {
        cl::Program* prog;
        cl::Kernel* ker;
    } kc_entry_t;

    typedef std::map<std::string, kc_entry_t> kc_t;
}
