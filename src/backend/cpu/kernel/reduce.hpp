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
#include <common/half.hpp>

namespace arrayfire {
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
    common::Transform<data_t<Ti>, compute_t<To>, op> transform;
    common::Binary<compute_t<To>, op> reduce;
    void operator()(Param<To> out, const dim_t outOffset, CParam<Ti> in,
                    const dim_t inOffset, const int dim, bool change_nan,
                    double nanval) {
        const af::dim4 istrides = in.strides();
        const af::dim4 idims    = in.dims();

        data_t<To> *const outPtr      = out.get() + outOffset;
        data_t<Ti> const *const inPtr = in.get() + inOffset;
        dim_t stride                  = istrides[dim];

        compute_t<To> out_val = common::Binary<compute_t<To>, op>::init();
        for (dim_t i = 0; i < idims[dim]; i++) {
            compute_t<To> in_val = transform(inPtr[i * stride]);
            if (change_nan) in_val = IS_NAN(in_val) ? nanval : in_val;
            out_val = reduce(in_val, out_val);
        }

        *outPtr = data_t<To>(out_val);
    }
};

template<typename Tk>
void n_reduced_keys(Param<Tk> okeys, int *n_reduced, CParam<Tk> keys) {
    const af::dim4 kdims = keys.dims();

    Tk *const outKeysPtr      = okeys.get();
    Tk const *const inKeysPtr = keys.get();

    int nkeys      = 0;
    Tk current_key = inKeysPtr[0];
    for (dim_t i = 0; i < kdims[0]; i++) {
        Tk keyval = inKeysPtr[i];

        if (keyval != current_key) {
            outKeysPtr[nkeys] = current_key;
            current_key       = keyval;
            ++nkeys;
        }

        if (i == (kdims[0] - 1)) { outKeysPtr[nkeys] = current_key; }
    }

    *n_reduced = nkeys + 1;
}

template<af_op_t op, typename Ti, typename Tk, typename To, int D>
struct reduce_dim_by_key {
    void operator()(Param<To> ovals, const dim_t ovOffset, CParam<Tk> keys,
                    CParam<Ti> vals, const dim_t vOffset, int *n_reduced,
                    const int dim, bool change_nan, double nanval) {
        static const int D1 = D - 1;
        reduce_dim_by_key<op, Ti, Tk, To, D1> reduce_by_key_dim_next;

        const af::dim4 ovstrides = ovals.strides();
        const af::dim4 vstrides  = vals.strides();
        const af::dim4 vdims     = ovals.dims();

        if (D1 == dim) {
            reduce_by_key_dim_next(ovals, ovOffset, keys, vals, vOffset,
                                   n_reduced, dim, change_nan, nanval);
        } else {
            for (dim_t i = 0; i < vdims[D1]; i++) {
                reduce_by_key_dim_next(ovals, ovOffset + (i * ovstrides[D1]),
                                       keys, vals, vOffset + (i * vstrides[D1]),
                                       n_reduced, dim, change_nan, nanval);
            }
        }
    }
};

template<af_op_t op, typename Ti, typename Tk, typename To>
struct reduce_dim_by_key<op, Ti, Tk, To, 0> {
    common::Transform<data_t<Ti>, compute_t<To>, op> transform;
    common::Binary<compute_t<To>, op> reduce;
    void operator()(Param<To> ovals, const dim_t ovOffset, CParam<Tk> keys,
                    CParam<Ti> vals, const dim_t vOffset, int *n_reduced,
                    const int dim, bool change_nan, double nanval) {
        const af::dim4 vstrides = vals.strides();
        const af::dim4 vdims    = vals.dims();

        const af::dim4 ovstrides = ovals.strides();

        data_t<Tk> const *const inKeysPtr = keys.get();
        data_t<Ti> const *const inValsPtr = vals.get();
        data_t<To> *const outValsPtr      = ovals.get();

        int keyidx                = 0;
        compute_t<Tk> current_key = compute_t<Tk>(inKeysPtr[0]);
        compute_t<To> out_val     = reduce.init();

        dim_t istride = vstrides[dim];
        dim_t ostride = ovstrides[dim];

        for (dim_t i = 0; i < vdims[dim]; i++) {
            compute_t<Tk> keyval = inKeysPtr[i];

            if (keyval == current_key) {
                compute_t<To> in_val =
                    transform(inValsPtr[vOffset + (i * istride)]);
                if (change_nan) in_val = IS_NAN(in_val) ? nanval : in_val;
                out_val = reduce(in_val, out_val);

            } else {
                outValsPtr[ovOffset + (keyidx * ostride)] = out_val;

                current_key = keyval;
                out_val     = transform(inValsPtr[vOffset + (i * istride)]);
                if (change_nan) out_val = IS_NAN(out_val) ? nanval : out_val;
                ++keyidx;
            }

            if (i == (vdims[dim] - 1)) {
                outValsPtr[ovOffset + (keyidx * ostride)] = out_val;
            }
        }
    }
};
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
