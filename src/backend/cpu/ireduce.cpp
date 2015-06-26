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
#include <ireduce.hpp>

using af::dim4;

namespace cpu
{
    template<typename T> double cabs(const T in) { return (double)in; }
    static double cabs(const char in) { return (double)(in > 0); }
    static double cabs(const cfloat &in) { return (double)abs(in); }
    static double cabs(const cdouble &in) { return (double)abs(in); }

    template<af_op_t op, typename T>
    struct MinMaxOp
    {
        T m_val;
        uint m_idx;
        MinMaxOp(T val, uint idx) :
            m_val(val), m_idx(idx)
        {
        }

        void operator()(T val, uint idx)
        {
            if (cabs(val) < cabs(m_val) ||
                (cabs(val) == cabs(m_val) &&
                 idx > m_idx)) {
                m_val = val;
                m_idx = idx;
            }
        }
    };

    template<typename T>
    struct MinMaxOp<af_max_t, T>
    {
        T m_val;
        uint m_idx;
        MinMaxOp(T val, uint idx) :
            m_val(val), m_idx(idx)
        {
        }

        void operator()(T val, uint idx)
        {
            if (cabs(val) > cabs(m_val) ||
                (cabs(val) == cabs(m_val) &&
                 idx <= m_idx)) {
                m_val = val;
                m_idx = idx;
            }
        }
    };

    template<af_op_t op, typename T, int D>
    struct ireduce_dim
    {
        void operator()(T *out, const dim4 ostrides, const dim4 odims,
                        uint *loc,
                        const T *in , const dim4 istrides, const dim4 idims,
                        const int dim)
        {
            const int D1 = D - 1;
            for (dim_t i = 0; i < odims[D1]; i++) {
                ireduce_dim<op, T, D1>()(out + i * ostrides[D1],
                                         ostrides, odims,
                                         loc + i * ostrides[D1],
                                         in  + i * istrides[D1],
                                         istrides, idims,
                                         dim);
            }
        }
    };

    template<af_op_t op, typename T>
    struct ireduce_dim<op, T, 0>
    {
        void operator()(T *out, const dim4 ostrides, const dim4 odims,
                        uint *loc,
                        const T *in , const dim4 istrides, const dim4 idims,
                        const int dim)
        {

            dim_t stride = istrides[dim];
            MinMaxOp<op, T> Op(in[0], 0);
            for (dim_t i = 0; i < idims[dim]; i++) {
                Op(in[i * stride], i);
            }

            *out = Op.m_val;
            *loc = Op.m_idx;
        }
    };

    template<af_op_t op, typename T>
    void ireduce(Array<T> &out, Array<uint> &loc,
                 const Array<T> &in, const int dim)
    {
        dim4 odims = in.dims();
        odims[dim] = 1;

        switch (in.ndims()) {
        case 1:
            ireduce_dim<op, T, 1>()(out.get(), out.strides(), out.dims(),
                                    loc.get(),
                                    in.get(), in.strides(), in.dims(), dim);
            break;

        case 2:
            ireduce_dim<op, T, 2>()(out.get(), out.strides(), out.dims(),
                                    loc.get(),
                                    in.get(), in.strides(), in.dims(), dim);
            break;

        case 3:
            ireduce_dim<op, T, 3>()(out.get(), out.strides(), out.dims(),
                                    loc.get(),
                                    in.get(), in.strides(), in.dims(), dim);
            break;

        case 4:
            ireduce_dim<op, T, 4>()(out.get(), out.strides(), out.dims(),
                                    loc.get(),
                                    in.get(), in.strides(), in.dims(), dim);
            break;
        }
    }

    template<af_op_t op, typename T>
    T ireduce_all(unsigned *loc, const Array<T> &in)
    {
        af::dim4 dims = in.dims();
        af::dim4 strides = in.strides();
        const T *inPtr = in.get();

        MinMaxOp<op, T> Op(inPtr[0], 0);

        for(dim_t l = 0; l < dims[3]; l++) {
            dim_t off3 = l * strides[3];

            for(dim_t k = 0; k < dims[2]; k++) {
                dim_t off2 = k * strides[2];

                for(dim_t j = 0; j < dims[1]; j++) {
                    dim_t off1 = j * strides[1];

                    for(dim_t i = 0; i < dims[0]; i++) {
                        dim_t idx = i + off1 + off2 + off3;
                        Op(inPtr[idx], idx);
                    }
                }
            }
        }

        *loc = Op.m_idx;
        return Op.m_val;
    }

#define INSTANTIATE(ROp, T)                                             \
    template void ireduce<ROp, T>(Array<T> &out, Array<uint> &loc,      \
                                  const Array<T> &in, const int dim);   \
    template T ireduce_all<ROp, T>(unsigned *loc, const Array<T> &in);  \

    //min
    INSTANTIATE(af_min_t, float  )
    INSTANTIATE(af_min_t, double )
    INSTANTIATE(af_min_t, cfloat )
    INSTANTIATE(af_min_t, cdouble)
    INSTANTIATE(af_min_t, int    )
    INSTANTIATE(af_min_t, uint   )
    INSTANTIATE(af_min_t, intl   )
    INSTANTIATE(af_min_t, uintl  )
    INSTANTIATE(af_min_t, char   )
    INSTANTIATE(af_min_t, uchar  )

    //max
    INSTANTIATE(af_max_t, float  )
    INSTANTIATE(af_max_t, double )
    INSTANTIATE(af_max_t, cfloat )
    INSTANTIATE(af_max_t, cdouble)
    INSTANTIATE(af_max_t, int    )
    INSTANTIATE(af_max_t, uint   )
    INSTANTIATE(af_max_t, intl   )
    INSTANTIATE(af_max_t, uintl  )
    INSTANTIATE(af_max_t, char   )
    INSTANTIATE(af_max_t, uchar  )
}
