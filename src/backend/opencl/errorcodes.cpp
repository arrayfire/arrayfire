/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <errorcodes.hpp>

#include <boost/compute/exception/opencl_error.hpp>

std::string getErrorMessage(int error_code)
{
    return boost::compute::opencl_error::to_string(error_code);
}
