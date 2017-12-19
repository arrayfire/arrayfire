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
template<typename T>
Array<T>  meanshift(const Array<T> &in,
                    const float &spatialSigma, const float &chromaticSigma,
                    const unsigned& numIterations, const bool& isColor);
}
