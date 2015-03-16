/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <lookup.hpp>
#include <err_cpu.hpp>
#include <cstdlib>

namespace cpu
{

static inline
dim_type trimIndex(dim_type idx, const dim_type &len)
{
    dim_type ret_val = idx;
    dim_type offset  = abs(ret_val)%len;
    if (ret_val<0) {
        ret_val = offset-1;
    } else if (ret_val>=len) {
        ret_val = len-offset-1;
    }
    return ret_val;
}

template<typename in_t, typename idx_t>
Array<in_t> lookup(const Array<in_t> &input, const Array<idx_t> &indices, const unsigned dim)
{
    const dim4 iDims = input.dims();
    const dim4 iStrides = input.strides();

    const in_t *inPtr = input.get();
    const idx_t *idxPtr = indices.get();

    dim4 oDims(1);
    for (dim_type d=0; d<4; ++d)
        oDims[d] = (d==int(dim) ? indices.elements() : iDims[d]);

    Array<in_t> out = createEmptyArray<in_t>(oDims);

    dim4 oStrides = out.strides();

    in_t *outPtr = out.get();

    for (dim_type l=0; l<oDims[3]; ++l) {

        dim_type iLOff = iStrides[3]*(dim==3 ? trimIndex((dim_type)idxPtr[l], iDims[3]): l);
        dim_type oLOff = l*oStrides[3];

        for (dim_type k=0; k<oDims[2]; ++k) {

            dim_type iKOff = iStrides[2]*(dim==2 ? trimIndex((dim_type)idxPtr[k], iDims[2]): k);
            dim_type oKOff = k*oStrides[2];

            for (dim_type j=0; j<oDims[1]; ++j) {

                dim_type iJOff = iStrides[1]*(dim==1 ? trimIndex((dim_type)idxPtr[j], iDims[1]): j);
                dim_type oJOff = j*oStrides[1];

                for (dim_type i=0; i<oDims[0]; ++i) {

                    dim_type iIOff = iStrides[0]*(dim==0 ? trimIndex((dim_type)idxPtr[i], iDims[0]): i);
                    dim_type oIOff = i*oStrides[0];

                    outPtr[oLOff+oKOff+oJOff+oIOff] = inPtr[iLOff+iKOff+iJOff+iIOff];
                }
            }
        }
    }

    return out;
}

#define INSTANTIATE(T)  \
    template Array<T>  lookup<T, float   >(const Array<T> &input, const Array<float   > &indices, const unsigned dim); \
    template Array<T>  lookup<T, double  >(const Array<T> &input, const Array<double  > &indices, const unsigned dim); \
    template Array<T>  lookup<T, int     >(const Array<T> &input, const Array<int     > &indices, const unsigned dim); \
    template Array<T>  lookup<T, unsigned>(const Array<T> &input, const Array<unsigned> &indices, const unsigned dim); \
    template Array<T>  lookup<T, uchar   >(const Array<T> &input, const Array<uchar   > &indices, const unsigned dim);

INSTANTIATE(float   );
INSTANTIATE(cfloat  );
INSTANTIATE(double  );
INSTANTIATE(cdouble );
INSTANTIATE(int     );
INSTANTIATE(unsigned);
INSTANTIATE(uchar   );
INSTANTIATE(char    );

}
