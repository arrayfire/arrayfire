/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/util.h>
#include <iostream>
#include "error.hpp"

using namespace std;

namespace af
{
    void print(const char *exp, const array &arr)
    {
        std::cout << exp << std::endl;
        AF_THROW(af_print_array(arr.get()));
        return;
    }
}
