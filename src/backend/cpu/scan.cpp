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

using af::dim4;

namespace cpu
{
    template<af_op_t op, typename Ti, typename To, int D>
    struct scan_dim
    {
        void operator()(To *out, const dim4 ostrides, const dim4 odims,
                        const Ti *in , const dim4 istrides, const dim4 idims,
                        const int dim)
        {
            const int D1 = D - 1;
            for (dim_t i = 0; i < odims[D1]; i++) {
                scan_dim<op, Ti, To, D1>()(out + i * ostrides[D1],
                                           ostrides, odims,
                                           in  + i * istrides[D1],
                                           istrides, idims,
                                           dim);
                if (D1 == dim) break;
            }
        }
    };

    template<af_op_t op, typename Ti, typename To>
    struct scan_dim<op, Ti, To, 0>
    {
        void operator()(To *out, const dim4 ostrides, const dim4 odims,
                        const Ti *in , const dim4 istrides, const dim4 idims,
                        const int dim)
        {

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
        dim4 dims = in.dims();

        Array<To> out = createValueArray<To>(dims, 0);

        switch (in.ndims()) {
        case 1:
            scan_dim<op, Ti, To, 1>()(out.get(), out.strides(), out.dims(),
                                      in.get(), in.strides(), in.dims(), dim);
            break;

        case 2:
            scan_dim<op, Ti, To, 2>()(out.get(), out.strides(), out.dims(),
                                      in.get(), in.strides(), in.dims(), dim);
            break;

        case 3:
            scan_dim<op, Ti, To, 3>()(out.get(), out.strides(), out.dims(),
                                      in.get(), in.strides(), in.dims(), dim);
            break;

        case 4:
            scan_dim<op, Ti, To, 4>()(out.get(), out.strides(), out.dims(),
                                      in.get(), in.strides(), in.dims(), dim);
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
    INSTANTIATE(af_notzero_t, char  , uint   )

}
