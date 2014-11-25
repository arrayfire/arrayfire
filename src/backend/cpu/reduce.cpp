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
#include <reduce.hpp>
#include <ops.hpp>

using af::dim4;

namespace cpu
{
    template<af_op_t op, typename Ti, typename To, int D>
    struct reduce_dim
    {
        void operator()(To *out, const dim4 ostrides, const dim4 odims,
                        const Ti *in , const dim4 istrides, const dim4 idims,
                        const int dim)
        {
            const int D1 = D - 1;
            for (dim_type i = 0; i < odims[D1]; i++) {
                reduce_dim<op, Ti, To, D1>()(out + i * ostrides[D1],
                                             ostrides, odims,
                                             in  + i * istrides[D1],
                                             istrides, idims,
                                             dim);
            }
        }
    };

    template<af_op_t op, typename Ti, typename To>
    struct reduce_dim<op, Ti, To, 0>
    {
        void operator()(To *out, const dim4 ostrides, const dim4 odims,
                        const Ti *in , const dim4 istrides, const dim4 idims,
                        const int dim)
        {

            dim_type stride = istrides[dim];

            Transform<Ti, To, op> transform;
            Binary<To, op> reduce;

            To out_val = reduce.init();
            for (dim_type i = 0; i < idims[dim]; i++) {
                To in_val = transform(in[i * stride]);
                out_val = reduce(in_val, out_val);
            }

            *out = out_val;
        }
    };

    template<af_op_t op, typename Ti, typename To>
    Array<To>* reduce(const Array<Ti> &in, const int dim)
    {
        dim4 odims = in.dims();
        odims[dim] = 1;

        Array<To> *out = createEmptyArray<To>(odims);

        switch (in.ndims()) {
        case 1:
            reduce_dim<op, Ti, To, 1>()(out->get(), out->strides(), out->dims(),
                                        in.get(), in.strides(), in.dims(), dim);
            break;

        case 2:
            reduce_dim<op, Ti, To, 2>()(out->get(), out->strides(), out->dims(),
                                        in.get(), in.strides(), in.dims(), dim);
            break;

        case 3:
            reduce_dim<op, Ti, To, 3>()(out->get(), out->strides(), out->dims(),
                                        in.get(), in.strides(), in.dims(), dim);
            break;

        case 4:
            reduce_dim<op, Ti, To, 4>()(out->get(), out->strides(), out->dims(),
                                        in.get(), in.strides(), in.dims(), dim);
            break;
        }

        return out;
    }

    template<af_op_t op, typename Ti, typename To>
    To reduce_global(const Array<Ti> &in)
    {
        Transform<Ti, To, op> transform;
        Binary<To, op> reduce;

        To out = reduce.init();

        // Decrement dimension of select dimension
        af::dim4 dims = in.dims();
        af::dim4 strides = in.strides();
        const Ti *inPtr = in.get();

        for(dim_type l = 0; l < dims[3]; l++) {
            dim_type off3 = l * strides[3];

            for(dim_type k = 0; k < dims[2]; k++) {
                dim_type off2 = k * strides[2];

                for(dim_type j = 0; j < dims[1]; j++) {
                    dim_type off1 = j * strides[1];

                    for(dim_type i = 0; i < dims[0]; i++) {
                        dim_type idx = i + off1 + off2 + off3;

                        To val = transform(inPtr[idx]);
                        out = reduce(val, out);
                    }
                }
            }
        }

        return out;
    }

#define INSTANTIATE(ROp, Ti, To)                                        \
    template Array<To>* reduce<ROp, Ti, To>(const Array<Ti> &in, const int dim); \
    template To reduce_global<ROp, Ti, To>(const Array<Ti> &in);

    //min
    INSTANTIATE(af_min_t, float  , float  )
    INSTANTIATE(af_min_t, double , double )
    INSTANTIATE(af_min_t, cfloat , cfloat )
    INSTANTIATE(af_min_t, cdouble, cdouble)
    INSTANTIATE(af_min_t, int    , int    )
    INSTANTIATE(af_min_t, uint   , uint   )
    INSTANTIATE(af_min_t, char   , char   )
    INSTANTIATE(af_min_t, uchar  , uchar  )

    //max
    INSTANTIATE(af_max_t, float  , float  )
    INSTANTIATE(af_max_t, double , double )
    INSTANTIATE(af_max_t, cfloat , cfloat )
    INSTANTIATE(af_max_t, cdouble, cdouble)
    INSTANTIATE(af_max_t, int    , int    )
    INSTANTIATE(af_max_t, uint   , uint   )
    INSTANTIATE(af_max_t, char   , char   )
    INSTANTIATE(af_max_t, uchar  , uchar  )

    //sum
    INSTANTIATE(af_add_t, float  , float  )
    INSTANTIATE(af_add_t, double , double )
    INSTANTIATE(af_add_t, cfloat , cfloat )
    INSTANTIATE(af_add_t, cdouble, cdouble)
    INSTANTIATE(af_add_t, int    , int    )
    INSTANTIATE(af_add_t, uint   , uint   )
    INSTANTIATE(af_add_t, char   , int    )
    INSTANTIATE(af_add_t, uchar  , uint   )

    // count
    INSTANTIATE(af_notzero_t, float  , uint)
    INSTANTIATE(af_notzero_t, double , uint)
    INSTANTIATE(af_notzero_t, cfloat , uint)
    INSTANTIATE(af_notzero_t, cdouble, uint)
    INSTANTIATE(af_notzero_t, int    , uint)
    INSTANTIATE(af_notzero_t, uint   , uint)
    INSTANTIATE(af_notzero_t, char   , uint)
    INSTANTIATE(af_notzero_t, uchar  , uint)

    //anytrue
    INSTANTIATE(af_or_t, float  , uchar)
    INSTANTIATE(af_or_t, double , uchar)
    INSTANTIATE(af_or_t, cfloat , uchar)
    INSTANTIATE(af_or_t, cdouble, uchar)
    INSTANTIATE(af_or_t, int    , uchar)
    INSTANTIATE(af_or_t, uint   , uchar)
    INSTANTIATE(af_or_t, char   , uchar)
    INSTANTIATE(af_or_t, uchar  , uchar)

    //alltrue
    INSTANTIATE(af_and_t, float  , uchar)
    INSTANTIATE(af_and_t, double , uchar)
    INSTANTIATE(af_and_t, cfloat , uchar)
    INSTANTIATE(af_and_t, cdouble, uchar)
    INSTANTIATE(af_and_t, int    , uchar)
    INSTANTIATE(af_and_t, uint   , uchar)
    INSTANTIATE(af_and_t, char   , uchar)
    INSTANTIATE(af_and_t, uchar  , uchar)
}
