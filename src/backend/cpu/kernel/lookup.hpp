/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <vector>
#include <Array.hpp>
#include <utility.hpp>

namespace cpu
{
namespace kernel
{

using af::dim4;

template<typename in_t, typename idx_t>
void lookup(Array<in_t> out, Array<in_t> const input,
            Array<idx_t> const indices, unsigned const dim)
{
    const dim4 iDims    = input.dims();
    const dim4 oDims    = out.dims();
    const dim4 iStrides = input.strides();
    const dim4 oStrides = out.strides();
    const in_t *inPtr   = input.get();
    const idx_t *idxPtr = indices.get();

    in_t *outPtr = out.get();

    for (dim_t l=0; l<oDims[3]; ++l) {

        dim_t iLOff = iStrides[3]*(dim==3 ? trimIndex((dim_t)idxPtr[l], iDims[3]): l);
        dim_t oLOff = l*oStrides[3];

        for (dim_t k=0; k<oDims[2]; ++k) {

            dim_t iKOff = iStrides[2]*(dim==2 ? trimIndex((dim_t)idxPtr[k], iDims[2]): k);
            dim_t oKOff = k*oStrides[2];

            for (dim_t j=0; j<oDims[1]; ++j) {

                dim_t iJOff = iStrides[1]*(dim==1 ? trimIndex((dim_t)idxPtr[j], iDims[1]): j);
                dim_t oJOff = j*oStrides[1];

                for (dim_t i=0; i<oDims[0]; ++i) {

                    dim_t iIOff = iStrides[0]*(dim==0 ? trimIndex((dim_t)idxPtr[i], iDims[0]): i);
                    dim_t oIOff = i*oStrides[0];

                    outPtr[oLOff+oKOff+oJOff+oIOff] = inPtr[iLOff+iKOff+iJOff+iIOff];
                }
            }
        }
    }
}

}
}
