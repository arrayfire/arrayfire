/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace opencl
{

template<typename inType, typename outType, bool isColor>
Array<outType> bilateral(const Array<inType> &in, const float &s_sigma, const float &c_sigma);

}
