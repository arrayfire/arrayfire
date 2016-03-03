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

template<typename T, af_sparse_storage>
void dense2storage(Array<T> &values, Array<int> &rowIdx, Array<int> &colIdx,
                   const Array<T> in);

}
