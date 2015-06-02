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
#include <reduce.hpp>
#include <ops.hpp>
#include <functional>
#include <complex>

using af::dim4;

namespace cpu
{
    template<af_op_t op, typename Ti, typename To, int D>
    struct reduce_dim
    {
        void operator()(To *out, const dim4 &ostrides, const dim4 &odims,
                        const Ti *in , const dim4 &istrides, const dim4 &idims,
                        const int dim)
        {
            static const int D1 = D - 1;
            static reduce_dim<op, Ti, To, D1> reduce_dim_next;
            for (dim_t i = 0; i < odims[D1]; i++) {
                 reduce_dim_next(out + i * ostrides[D1],
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

        Transform<Ti, To, op> transform;
        Binary<To, op> reduce;
        void operator()(To *out, const dim4 &ostrides, const dim4 &odims,
                        const Ti *in , const dim4 &istrides, const dim4 &idims,
                        const int dim)
        {
            dim_t stride = istrides[dim];

            To out_val = reduce.init();
            for (dim_t i = 0; i < idims[dim]; i++) {
                To in_val = transform(in[i * stride]);
                out_val = reduce(in_val, out_val);
            }

            *out = out_val;
        }
    };

    template<af_op_t op, typename Ti, typename To>
    using reduce_dim_func = std::function<void(To*,const dim4&, const dim4&,
                                                const Ti*, const dim4&, const dim4&,
                                                const int)>;

    template<af_op_t op, typename Ti, typename To>
    Array<To> reduce(const Array<Ti> &in, const int dim)
    {
        dim4 odims = in.dims();
        odims[dim] = 1;

        Array<To> out = createEmptyArray<To>(odims);
        static reduce_dim_func<op, Ti, To>  reduce_funcs[4] = { reduce_dim<op, Ti, To, 1>()
                                                              , reduce_dim<op, Ti, To, 2>()
                                                              , reduce_dim<op, Ti, To, 3>()
                                                              , reduce_dim<op, Ti, To, 4>()};

        reduce_funcs[in.ndims() - 1](out.get(), out.strides(), out.dims(),
                                    in.get(), in.strides(), in.dims(), dim);

        return out;
    }

    template<af_op_t op, typename Ti, typename To>
    To reduce_all(const Array<Ti> &in)
    {
        Transform<Ti, To, op> transform;
        Binary<To, op> reduce;

        To out = reduce.init();

        // Decrement dimension of select dimension
        af::dim4 dims = in.dims();
        af::dim4 strides = in.strides();
        const Ti *inPtr = in.get();

        for(dim_t l = 0; l < dims[3]; l++) {
            dim_t off3 = l * strides[3];

            for(dim_t k = 0; k < dims[2]; k++) {
                dim_t off2 = k * strides[2];

                for(dim_t j = 0; j < dims[1]; j++) {
                    dim_t off1 = j * strides[1];

                    for(dim_t i = 0; i < dims[0]; i++) {
                        dim_t idx = i + off1 + off2 + off3;

                        To val = transform(inPtr[idx]);
                        out = reduce(val, out);
                    }
                }
            }
        }

        return out;
    }

#define INSTANTIATE(ROp, Ti, To)                                        \
    template Array<To> reduce<ROp, Ti, To>(const Array<Ti> &in, const int dim); \
    template To reduce_all<ROp, Ti, To>(const Array<Ti> &in);

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

    //sum
    INSTANTIATE(af_mul_t, float  , float  )
    INSTANTIATE(af_mul_t, double , double )
    INSTANTIATE(af_mul_t, cfloat , cfloat )
    INSTANTIATE(af_mul_t, cdouble, cdouble)
    INSTANTIATE(af_mul_t, int    , int    )
    INSTANTIATE(af_mul_t, uint   , uint   )
    INSTANTIATE(af_mul_t, char   , int    )
    INSTANTIATE(af_mul_t, uchar  , uint   )

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
    INSTANTIATE(af_or_t, float  , char)
    INSTANTIATE(af_or_t, double , char)
    INSTANTIATE(af_or_t, cfloat , char)
    INSTANTIATE(af_or_t, cdouble, char)
    INSTANTIATE(af_or_t, int    , char)
    INSTANTIATE(af_or_t, uint   , char)
    INSTANTIATE(af_or_t, char   , char)
    INSTANTIATE(af_or_t, uchar  , char)

    //alltrue
    INSTANTIATE(af_and_t, float  , char)
    INSTANTIATE(af_and_t, double , char)
    INSTANTIATE(af_and_t, cfloat , char)
    INSTANTIATE(af_and_t, cdouble, char)
    INSTANTIATE(af_and_t, int    , char)
    INSTANTIATE(af_and_t, uint   , char)
    INSTANTIATE(af_and_t, char   , char)
    INSTANTIATE(af_and_t, uchar  , char)
}
