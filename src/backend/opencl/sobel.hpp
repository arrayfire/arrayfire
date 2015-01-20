/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <utility>

namespace opencl
{

template<typename T>
std::pair< Array<T>*, Array<T>* > 
sobelDerivatives(const Array<T> &img, const unsigned &ker_size);

}
