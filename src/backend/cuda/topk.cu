/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/topk.hpp>
#include <topk.hpp>
#include <Array.hpp>
#include <af/dim4.hpp>

namespace cuda
{
template<typename T>
void topk(Array<T>& ovals, Array<uint>& oidxs, const Array<T>& ivals,
          const int k, const int dim, const af::topkFunction order) {
    dim4 outDims = ivals.dims();
    outDims[dim] = k;

    ovals = createEmptyArray<T>(outDims);
    oidxs = createEmptyArray<uint>(outDims);

    kernel::topk<T>(ovals, oidxs, ivals, k, dim, order);
}

#define INSTANTIATE(T)\
template void topk<T>(Array<T>&, Array<uint>&, const Array<T>&, \
                      const int, const int, const af::topkFunction);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(int   )
INSTANTIATE(uint  )
}
