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

namespace cpu
{
namespace kernel
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
    void operator()(Param<T> output, Param<uint> locParam, const dim_t outOffset,
                    CParam<T> input, const dim_t inOffset, const int dim)
    {
        const af::dim4 odims    = output.dims();
        const af::dim4 ostrides = output.strides();
        const af::dim4 istrides = input.strides();
        const int D1 = D - 1;
        for (dim_t i = 0; i < odims[D1]; i++) {
            ireduce_dim<op, T, D1>()(output, locParam, outOffset + i * ostrides[D1],
                                     input, inOffset + i * istrides[D1], dim);
        }
    }
};

template<af_op_t op, typename T>
struct ireduce_dim<op, T, 0>
{
    void operator()(Param<T> output, Param<uint> locParam, const dim_t outOffset,
                    CParam<T> input, const dim_t inOffset, const int dim)
    {
        const af::dim4 idims = input.dims();
        const af::dim4 istrides = input.strides();

        T const * const in = input.get();
        T * out = output.get();
        uint * loc = locParam.get();

        dim_t stride = istrides[dim];
        MinMaxOp<op, T> Op(in[inOffset], 0);
        for (dim_t i = 0; i < idims[dim]; i++) {
            Op(in[inOffset + i * stride], i);
        }

        out[outOffset] = Op.m_val;
        loc[outOffset] = Op.m_idx;
    }
};

}
}
