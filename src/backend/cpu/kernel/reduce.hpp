/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <ops.hpp>

namespace cpu {
namespace kernel {

template<af_op_t op, typename Ti, typename To, int D>
struct reduce_dim {
    void operator()(Param<To> out, const dim_t outOffset, CParam<Ti> in,
                    const dim_t inOffset, const int dim, bool change_nan,
                    double nanval) {
        static const int D1 = D - 1;
        reduce_dim<op, Ti, To, D1> reduce_dim_next;

        const af::dim4 ostrides = out.strides();
        const af::dim4 istrides = in.strides();
        const af::dim4 odims    = out.dims();

        for (dim_t i = 0; i < odims[D1]; i++) {
            reduce_dim_next(out, outOffset + i * ostrides[D1], in,
                            inOffset + i * istrides[D1], dim, change_nan,
                            nanval);
        }
    }
};

template<af_op_t op, typename Ti, typename To>
struct reduce_dim<op, Ti, To, 0> {
    Transform<Ti, To, op> transform;
    Binary<To, op> reduce;
    void operator()(Param<To> out, const dim_t outOffset, CParam<Ti> in,
                    const dim_t inOffset, const int dim, bool change_nan,
                    double nanval) {
        const af::dim4 istrides = in.strides();
        const af::dim4 idims    = in.dims();

        To* const outPtr      = out.get() + outOffset;
        Ti const* const inPtr = in.get() + inOffset;
        dim_t stride          = istrides[dim];

        To out_val = Binary<To, op>::init();
        for (dim_t i = 0; i < idims[dim]; i++) {
            To in_val = transform(inPtr[i * stride]);
            if (change_nan) in_val = IS_NAN(in_val) ? nanval : in_val;
            out_val = reduce(in_val, out_val);
        }

        *outPtr = out_val;
    }
};

}  // namespace kernel
}  // namespace cpu
