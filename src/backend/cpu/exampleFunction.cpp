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
Array<T> exampleFunction(const Array<T> &a, const Array<T> &b, const af_someenum_t method)
{
    a.eval();                           // All input Arrays should call eval mandatorily
                                        // in CPU backend function implementations. Since
                                        // the cpu fns are asynchronous launches, any Arrays
                                        // that are either views/JIT nodes needs to evaluated
                                        // before they are passed onto functions that are
                                        // enqueued onto the queues.
    b.eval();

    dim4 outputDims;                    // this should be '= in.dims();' in most cases
                                        // but would definitely depend on the type of
                                        // algorithm you are implementing.

    Array<T> out = createEmptyArray<T>(outputDims);
                                        // Please use the create***Array<T> helper
                                        // functions defined in Array.hpp to create
                                        // different types of Arrays. Please check the
                                        // file to know what are the different types you
                                        // can create.

    dim4 aDims    = a.dims();           // you can retrieve dimensions
    dim4 bDims    = b.dims();
    dim4 oDims    = out.dims();

    dim_t aOffset = a.getOffset();      // you can retrieve the offset - used when given array
                                        // is an sub-array pointing to some other array and
                                        // doesn't have memory of its own
    dim_t bOffset = b.getOffset();

    dim4 aStrides = a.strides();        // you can retrieve strides
    dim4 bStrides = b.strides();
    dim4 oStrides = out.strides();

    const T* src1 = a.get();            // cpu::Array<T>::get returns the pointer to the
                                        // memory allocated for that Array
    const T* src2 = b.get();            // cpu::Array<T>::get returns the pointer to the
                                        // memory allocated for that Array
    src1 += aOffset;
    src2 += bOffset;

    T* dst = out.get();

    // Implement your algorithm and write results to dst
    for(int j=0; j<aDims[1]; ++j) {
        for (int i=0; i<aDims[0]; ++i) {

            int src1Idx = i*aStrides[0] + j*aStrides[1];
            int src2Idx = i*bStrides[0] + j*bStrides[1];
            int dstIdx  = i*oStrides[0] + j*oStrides[1];

            // kernel algorithm goes here
            switch(method) {
                case 1: dst[dstIdx] = src1[src1Idx] + src2[src2Idx]; break;
                case 2: dst[dstIdx] = src1[src1Idx] - src2[src2Idx]; break;
                case 3: dst[dstIdx] = src1[src1Idx] * src2[src2Idx]; break;
                case 4: dst[dstIdx] = src1[src1Idx] / src2[src2Idx]; break;
            }
        }
    }

    return out;                         // return the result
}


#define INSTANTIATE(T)  \
    template Array<T> exampleFunction<T>(const Array<T> &a, const Array<T> &b, const af_someenum_t method);

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
