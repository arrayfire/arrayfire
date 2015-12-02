/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <scan.hpp>
#include <ops.hpp>
#include <platform.hpp>
#include <async_queue.hpp>

using af::dim4;

namespace cpu
{

template<af_op_t op, typename Ti, typename To, int D>
struct scan_dim
{
    void operator()(Array<To> out, dim_t outOffset,
                    const Array<Ti> in, dim_t inOffset,
                    const int dim) const
    {
        const dim4 odims    = out.dims();
        const dim4 ostrides = out.strides();
        const dim4 istrides = in.strides();

        const int D1 = D - 1;
        for (dim_t i = 0; i < odims[D1]; i++) {
            scan_dim<op, Ti, To, D1> func;
            getQueue().enqueue(func,
                    out, outOffset + i * ostrides[D1],
                    in, inOffset + i * istrides[D1], dim);
            if (D1 == dim) break;
        }
    }
};

template<af_op_t op, typename Ti, typename To>
struct scan_dim<op, Ti, To, 0>
{
    void operator()(Array<To> output, dim_t outOffset,
                    const Array<Ti> input,  dim_t inOffset,
                    const int dim) const
    {
        const Ti* in = input.get() + inOffset;
              To* out= output.get()+ outOffset;

        const dim4 ostrides = output.strides();
        const dim4 istrides = input.strides();
        const dim4 idims    = input.dims();

        dim_t istride = istrides[dim];
        dim_t ostride = ostrides[dim];

        Transform<Ti, To, op> transform;
        // FIXME: Change the name to something better
        Binary<To, op> scan;

        To out_val = scan.init();
        for (dim_t i = 0; i < idims[dim]; i++) {
            To in_val = transform(in[i * istride]);
            out_val = scan(in_val, out_val);
            out[i * ostride] = out_val;
        }
    }
};

template<af_op_t op, typename Ti, typename To>
Array<To> scan(const Array<Ti>& in, const int dim)
{
    dim4 dims     = in.dims();
    Array<To> out = createValueArray<To>(dims, 0);
    out.eval();
    in.eval();

    switch (in.ndims()) {
        case 1:
            scan_dim<op, Ti, To, 1> func1;
            getQueue().enqueue(func1, out, 0, in, 0, dim);
            break;
        case 2:
            scan_dim<op, Ti, To, 2> func2;
            getQueue().enqueue(func2, out, 0, in, 0, dim);
            break;
        case 3:
            scan_dim<op, Ti, To, 3> func3;
            getQueue().enqueue(func3, out, 0, in, 0, dim);
            break;
        case 4:
            scan_dim<op, Ti, To, 4> func4;
            getQueue().enqueue(func4, out, 0, in, 0, dim);
            break;
    }

    return out;
}

#define INSTANTIATE(ROp, Ti, To)                                        \
    template Array<To> scan<ROp, Ti, To>(const Array<Ti> &in, const int dim); \

//accum
INSTANTIATE(af_add_t, float  , float  )
INSTANTIATE(af_add_t, double , double )
INSTANTIATE(af_add_t, cfloat , cfloat )
INSTANTIATE(af_add_t, cdouble, cdouble)
INSTANTIATE(af_add_t, int    , int    )
INSTANTIATE(af_add_t, uint   , uint   )
INSTANTIATE(af_add_t, intl   , intl   )
INSTANTIATE(af_add_t, uintl  , uintl  )
INSTANTIATE(af_add_t, char   , int    )
INSTANTIATE(af_add_t, uchar  , uint   )
INSTANTIATE(af_add_t, short  , int    )
INSTANTIATE(af_add_t, ushort , uint   )
INSTANTIATE(af_notzero_t, char  , uint)

}
