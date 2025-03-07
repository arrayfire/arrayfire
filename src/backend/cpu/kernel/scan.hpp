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
#include <common/Binary.hpp>
#include <common/Transform.hpp>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<af_op_t op, typename Ti, typename To, int D, bool inclusive_scan>
struct scan_dim {
    void operator()(Param<To> out, dim_t outOffset, CParam<Ti> in,
                    dim_t inOffset, const int dim) const {
        const af::dim4 odims    = out.dims();
        const af::dim4 ostrides = out.strides();
        const af::dim4 istrides = in.strides();

        const int D1 = D - 1;
        for (dim_t i = 0; i < odims[D1]; i++) {
            scan_dim<op, Ti, To, D1, inclusive_scan> func;
            func(out, outOffset + i * ostrides[D1], in,
                 inOffset + i * istrides[D1], dim);
            if (D1 == dim) break;
        }
    }
};

template<af_op_t op, typename Ti, typename To, bool inclusive_scan>
struct scan_dim<op, Ti, To, 0, inclusive_scan> {
    void operator()(Param<To> output, dim_t outOffset, CParam<Ti> input,
                    dim_t inOffset, const int dim) const {
        const Ti* in = input.get() + inOffset;
        To* out      = output.get() + outOffset;

        const af::dim4 ostrides = output.strides();
        const af::dim4 istrides = input.strides();
        const af::dim4 idims    = input.dims();

        dim_t istride = istrides[dim];
        dim_t ostride = ostrides[dim];

        common::Transform<Ti, To, op> transform;
        // FIXME: Change the name to something better
        common::Binary<To, op> scan;

        To out_val = common::Binary<To, op>::init();
        for (dim_t i = 0; i < idims[dim]; i++) {
            To in_val = transform(in[i * istride]);
            out_val   = scan(in_val, out_val);
            if (!inclusive_scan) {
                // The loop shifts the output index by 1.
                // The last index wraps around and writes the first element.
                if (i == (idims[dim] - 1)) {
                    out[0] = common::Binary<To, op>::init();
                } else {
                    out[(i + 1) * ostride] = out_val;
                }
            } else {
                out[i * ostride] = out_val;
            }
        }
    }
};

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
