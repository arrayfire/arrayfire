/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>  // header with cpu backend specific
                      // Array class implementation that inherits
                      // ArrayInfo base class

#include <exampleFunction.hpp>         // cpu backend function header
#include <kernel/exampleFunction.hpp>  // Function implementation header

#include <err_cpu.hpp>  // error check functions and Macros
                        // specific to cpu backend
#include <platform.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace cpu {

template<typename T>
Array<T> exampleFunction(const Array<T> &a, const Array<T> &b,
                         const af_someenum_t method) {
    dim4 outputDims;  // this should be '= in.dims();' in most cases
                      // but would definitely depend on the type of
                      // algorithm you are implementing.

    Array<T> out = createEmptyArray<T>(outputDims);
    // Please use the create***Array<T> helper
    // functions defined in Array.hpp to create
    // different types of Arrays. Please check the
    // file to know what are the different types you
    // can create.

    // Enqueue the function call on the worker thread
    // This code will be present in src/backend/cpu/kernel/exampleFunction.hpp
    getQueue().enqueue(kernel::exampleFunction<T>, out, a, b, method);

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

}  // namespace cpu
}  // namespace arrayfire
