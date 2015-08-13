/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

#include <svd.hpp>          // opencl backend function header

#include <err_opencl.hpp>               // error check functions and Macros
                                        // specific to opencl backend

namespace opencl
{

template<typename T>
void svd(Array<T> &s, Array<T> &u, Array<T> &vt, const Array<T> &in)
{
    OPENCL_NOT_SUPPORTED();
}

template<typename T>
void svdInPlace(Array<T> &s, Array<T> &u, Array<T> &vt, Array<T> &in)
{
    OPENCL_NOT_SUPPORTED();
}


#define INSTANTIATE(T)                                                  \
    template void svd(Array<T> &s, Array<T> &u, Array<T> &vt, const Array<T> &in); \
    template void svdInPlace(Array<T> &s, Array<T> &u, Array<T> &vt, Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(double)

}
