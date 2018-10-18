/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/dim4.hpp>
#include <complex>
#include <err_cpu.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include <JIT/UnaryNode.hpp>
#include <Array.hpp>

namespace cpu
{

template<typename To, typename Ti>
struct UnOp<To, Ti, af_cast_t>
{
    void eval(JIT::array<To> &out,
              const JIT::array<Ti> &in, int lim)
    {
        for (int i = 0; i < lim; i++) {
            out[i] = To(in[i]);
        }
    }
};

template<typename To>
struct UnOp<To, std::complex<float>, af_cast_t>
{
    typedef std::complex<float> Ti;
    void eval(JIT::array<To> &out,
              const JIT::array<Ti> &in, int lim)
    {
        for (int i = 0; i < lim; i++) {
            out[i] = To(std::abs(in[i]));
        }
    }
};

template<typename To>
struct UnOp<To, std::complex<double>, af_cast_t>
{
    typedef std::complex<double> Ti;
    void eval(JIT::array<To> &out,
              const JIT::array<Ti> &in, int lim)
    {
        for (int i = 0; i < lim; i++) {
            out[i] = To(std::abs(in[i]));
        }
    }
};

// DO NOT REMOVE THE TWO SPECIALIZATIONS BELOW
// These specializations are required because we partially specialize when Ti = std::complex<T>
// The partial specializations above expect output to be real.
// so they To(std::abs(v)) instead of To(v) which results in incorrect values when To is complex.

template<>
struct UnOp<std::complex<float>, std::complex<double>, af_cast_t>
{
    typedef std::complex<double> Ti;
    typedef std::complex<float> To;
    void eval(JIT::array<To> &out,
              const JIT::array<Ti> &in, int lim)
    {
        for (int i = 0; i < lim; i++) {
            out[i] = To(in[i]);
        }
    }
};

template<>
struct UnOp<std::complex<double>, std::complex<float>, af_cast_t>
{
    typedef std::complex<float> Ti;
    typedef std::complex<double> To;
    void eval(JIT::array<To> &out,
              const JIT::array<Ti> &in, int lim)
    {
        for (int i = 0; i < lim; i++) {
            out[i] = To(in[i]);
        }
    }
};

#define CAST_B8(T)                                      \
    template<>                                          \
    struct UnOp<char, T, af_cast_t>                     \
    {                                                   \
        void eval(JIT::array<char> &out,                \
                  const JIT::array<T> &in, int lim)     \
        {                                               \
            for (int i = 0; i < lim; i++) {             \
                out[i] = char(in[i] != 0);              \
            }                                           \
        }                                               \
    };                                                  \

CAST_B8(float)
CAST_B8(double)
CAST_B8(int)
CAST_B8(uchar)
CAST_B8(char)

template<typename To, typename Ti>
struct CastWrapper
{
    Array<To> operator()(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode<To, Ti, af_cast_t> *node = new JIT::UnaryNode<To, Ti, af_cast_t>(in_node);
        return createNodeArray<To>(in.dims(), JIT::Node_ptr(
                                       reinterpret_cast<JIT::Node *>(node)));
    }
};

template<typename T>
struct CastWrapper<T, T>
{
    Array<T> operator()(const Array<T> &in)
    {
        return in;
    }
};

template<typename To, typename Ti>
Array<To> cast(const Array<Ti> &in)
{
    CastWrapper<To, Ti> cast_op;
    return cast_op(in);
}

}
