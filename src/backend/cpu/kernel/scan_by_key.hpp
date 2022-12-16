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

template<af_op_t op, typename Ti, typename Tk, typename To, int D>
struct scan_dim_by_key {
    bool inclusive_scan;
    scan_dim_by_key(bool inclusiveSanKey) : inclusive_scan(inclusiveSanKey) {}

    void operator()(Param<To> out, dim_t outOffset, CParam<Tk> key,
                    dim_t keyOffset, CParam<Ti> in, dim_t inOffset,
                    const int dim) const {
        const af::dim4 odims    = out.dims();
        const af::dim4 ostrides = out.strides();
        const af::dim4 kstrides = key.strides();
        const af::dim4 istrides = in.strides();

        const int D1 = D - 1;
        for (dim_t i = 0; i < odims[D1]; i++) {
            scan_dim_by_key<op, Ti, Tk, To, D1> func(inclusive_scan);
            func(out, outOffset + i * ostrides[D1], key,
                 keyOffset + i * kstrides[D1], in, inOffset + i * istrides[D1],
                 dim);
            if (D1 == dim) break;
        }
    }
};

template<af_op_t op, typename Ti, typename Tk, typename To>
struct scan_dim_by_key<op, Ti, Tk, To, 0> {
    bool inclusive_scan;
    scan_dim_by_key(bool inclusiveSanKey) : inclusive_scan(inclusiveSanKey) {}

    void operator()(Param<To> output, dim_t outOffset, CParam<Tk> keyinput,
                    dim_t keyOffset, CParam<Ti> input, dim_t inOffset,
                    const int dim) const {
        const Ti* in  = input.get() + inOffset;
        const Tk* key = keyinput.get() + keyOffset;
        To* out       = output.get() + outOffset;

        const af::dim4 ostrides = output.strides();
        const af::dim4 kstrides = keyinput.strides();
        const af::dim4 istrides = input.strides();
        const af::dim4 idims    = input.dims();

        dim_t istride = istrides[dim];
        dim_t kstride = kstrides[dim];
        dim_t ostride = ostrides[dim];

        common::Transform<Ti, To, op> transform;
        // FIXME: Change the name to something better
        common::Binary<To, op> scan;

        To out_val = common::Binary<To, op>::init();
        Tk key_val = key[0];

        dim_t k = !inclusive_scan;
        if (!inclusive_scan) { out[0] = common::Binary<To, op>::init(); }

        for (dim_t i = 0; i < idims[dim] - (!inclusive_scan); i++, k++) {
            To in_val = transform(in[i * istride]);
            if (key[k * kstride] != key_val) {
                out_val =
                    !inclusive_scan ? common::Binary<To, op>::init() : in_val;
                key_val = key[k * kstride];
            } else {
                out_val = scan(in_val, out_val);
            }
            out[k * ostride] = out_val;
        }
    }
};

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
