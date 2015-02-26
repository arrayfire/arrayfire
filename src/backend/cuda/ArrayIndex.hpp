/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace cuda
{

template<typename in_t, typename idx_t>
Array<in_t> arrayIndex(const Array<in_t> &input, const Array<idx_t> &indices, const unsigned dim);

}
