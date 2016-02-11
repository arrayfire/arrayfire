/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>                    // header with cpu backend specific
                                        // Array class implementation that inherits
                                        // ArrayInfo base class

#include <exampleFunction.hpp>          // cpu backend function header

#include <err_cpu.hpp>                  // error check functions and Macros
                                        // specific to cpu backend

using af::dim4;

namespace cpu
{

template<typename T>
Array<T> exampleFunction(const Array<T> &in, const af_someenum_t method)
{
    in.eval();                          // All input Arrays should call eval mandatorily
                                        // in CPU backend function implementations. Since
                                        // the cpu fns are asynchronous launches, any Arrays
                                        // that are either views/JIT nodes needs to evaluated
                                        // before they are passed onto functions that are
                                        // enqueued onto the queues.

    dim4 outputDims;                    // this should be '= in.dims();' in most cases
                                        // but would definitely depend on the type of
                                        // algorithm you are implementing.

    Array<T> out = createEmptyArray<T>(outputDims);
                                        // Please use the create***Array<T> helper
                                        // functions defined in Array.hpp to create
                                        // different types of Arrays. Please check the
                                        // file to know what are the different types you
                                        // can create.

    //dim4 in_dims    = in.dims();        // you can retrieve dimensions

    //dim_t in_offset = in.getOffset(); // you can retrieve the offset - used when given array
                                        // is an sub-array pointing to some other array and
                                        // doesn't have memory of its own

    //dim4 in_strides = in.strides();     // you can retrieve strides

    //const T* src = in.get();            // cpu::Array<T>::get returns the pointer to the
                                        // memory allocated for that Array

    //T* dst = out.get();

    // Implement your algorithm and write results to dst

    return out;                         // return the result
}


#define INSTANTIATE(T)  \
    template Array<T> exampleFunction<T>(const Array<T> &in, const af_someenum_t method);

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

}
