/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>  // header with oneapi backend specific
                      // Array class implementation that inherits
                      // ArrayInfo base class

#include <exampleFunction.hpp>  // oneapi backend function header

#include <err_oneapi.hpp>  // error check functions and Macros
                           // specific to oneapi backend

// #include <kernel/exampleFunction.hpp>  // this header under the folder
//  src/oneapi/kernel
//  defines the OneAPI kernel wrapper
//  function to which the main computation of your
//  algorithm should be relayed to

using af::dim4;

namespace arrayfire {
namespace oneapi {

template<typename T>
Array<T> exampleFunction(const Array<T> &a, const Array<T> &b,
                         const af_someenum_t method) {
    ONEAPI_NOT_SUPPORTED("");
    dim4 outputDims;  // this should be '= in.dims();' in most cases
                      // but would definitely depend on the type of
                      // algorithm you are implementing.

    Array<T> out = createEmptyArray<T>(outputDims);
    // Please use the create***Array<T> helper
    // functions defined in Array.hpp to create
    // different types of Arrays. Please check the
    // file to know what are the different types you
    // can create.

    // Relay the actual computation to OneAPI kernel wrapper
    // kernel::exampleFunc<T>(out, a, b, method);

    return out;  // return the result
}

#define INSTANTIATE(T)                                                         \
    template Array<T> exampleFunction<T>(const Array<T> &a, const Array<T> &b, \
                                         const af_someenum_t method);

// INSTANTIATIONS for all the types which
// are present in the switch case statement
// in src/api/c/exampleFunction.cpp should be available
INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

}  // namespace oneapi
}  // namespace arrayfire
