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
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <complex>
#include <err_cpu.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include <TNJ/UnaryNode.hpp>
#include <Array.hpp>

namespace cpu
{

template<typename To, typename Ti>
struct UnOp<To, Ti, af_cast_t>
{
    To eval(Ti in)
    {
        return To(in);
    }
};

template<typename To>
struct UnOp<To, std::complex<float>, af_cast_t>
{
    To eval(std::complex<float> in)
    {
        return To(std::abs(in));
    }
};

template<typename To>
struct UnOp<To, std::complex<double>, af_cast_t>
{
    To eval(std::complex<double> in)
    {
        return To(std::abs(in));
    }
};

template<>
struct UnOp<std::complex<float>, std::complex<double>, af_cast_t>
{
    std::complex<float> eval(std::complex<double> in)
    {
        return std::complex<float>(in);
    }
};

template<>
struct UnOp<std::complex<double>, std::complex<float>, af_cast_t>
{
    std::complex<double> eval(std::complex<float> in)
    {
        return std::complex<double>(in);
    }
};

#define CAST_B8(T)                              \
    template<>                                  \
    struct UnOp<char, T, af_cast_t>             \
    {                                           \
        char eval(T in)                         \
        {                                       \
            return char(in != 0);               \
        }                                       \
    };                                          \

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
        TNJ::Node_ptr in_node = in.getNode();
        TNJ::UnaryNode<To, Ti, af_cast_t> *node = new TNJ::UnaryNode<To, Ti, af_cast_t>(in_node);
        return createNodeArray<To>(in.dims(), TNJ::Node_ptr(
                                       reinterpret_cast<TNJ::Node *>(node)));
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
