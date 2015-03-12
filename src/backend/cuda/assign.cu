/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <handle.hpp>
#include <assign.hpp>
//#include <kernel/assign.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T>
void assign(Array<T>& out, const af_index_t idxrs[], const Array<T>& rhs)
{
    CUDA_NOT_SUPPORTED();
}

#define INSTANTIATE(T) \
    template void assign<T>(Array<T>& out, const af_index_t idxrs[], const Array<T>& rhs);

INSTANTIATE(cdouble)
INSTANTIATE(double )
INSTANTIATE(cfloat )
INSTANTIATE(float  )
INSTANTIATE(uintl  )
INSTANTIATE(uint   )
INSTANTIATE(intl   )
INSTANTIATE(int    )
INSTANTIATE(uchar  )
INSTANTIATE(char   )

}
