/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <err_cpu.hpp>
#include <jit/UnaryNode.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include <af/dim4.hpp>
#include <complex>

namespace arrayfire {
namespace cpu {

template<typename To, typename Ti>
struct UnOp<To, Ti, af_cast_t> {
    void eval(jit::array<To> &out, const jit::array<Ti> &in, int lim) {
        for (int i = 0; i < lim; i++) { out[i] = To(in[i]); }
    }
};

/// NOTE(umar): The next specializations have multiple eval functions because
/// the f16 data type needs to be converted to and from the compute type.
/// Here, we have specializations for real numbers as well as the complex
/// numbers
/// TODO(umar): make a macro to reduce repeat code

template<typename To>
struct UnOp<To, arrayfire::common::half, af_cast_t> {
    typedef arrayfire::common::half Ti;

    void eval(jit::array<To> &out, const jit::array<Ti> &in, int lim) {
        for (int i = 0; i < lim; i++) {
            float val = static_cast<float>(in[i]);
            out[i]    = To(val);
        }
    }

    void eval(jit::array<To> &out, const jit::array<float> &in, int lim) {
        for (int i = 0; i < lim; i++) { out[i] = To(in[i]); }
    }
};

template<typename Ti>
struct UnOp<arrayfire::common::half, Ti, af_cast_t> {
    typedef arrayfire::common::half To;

    void eval(jit::array<To> &out, const jit::array<Ti> &in, int lim) {
        for (int i = 0; i < lim; i++) {
            float val = static_cast<float>(in[i]);
            out[i]    = To(val);
        }
    }

    void eval(jit::array<float> &out, const jit::array<Ti> &in, int lim) {
        for (int i = 0; i < lim; i++) { out[i] = float(in[i]); }
    }
};

template<>
struct UnOp<arrayfire::common::half, std::complex<float>, af_cast_t> {
    typedef arrayfire::common::half To;
    typedef std::complex<float> Ti;

    void eval(jit::array<To> &out, const jit::array<Ti> &in, int lim) {
        for (int i = 0; i < lim; i++) {
            float val = std::abs(in[i]);
            out[i]    = To(val);
        }
    }

    void eval(jit::array<float> &out, const jit::array<Ti> &in, int lim) {
        for (int i = 0; i < lim; i++) { out[i] = std::abs(in[i]); }
    }
};

template<>
struct UnOp<arrayfire::common::half, std::complex<double>, af_cast_t> {
    typedef arrayfire::common::half To;
    typedef std::complex<double> Ti;

    void eval(jit::array<To> &out, const jit::array<Ti> &in, int lim) {
        for (int i = 0; i < lim; i++) {
            float val = std::abs(in[i]);
            out[i]    = To(val);
        }
    }

    void eval(jit::array<float> &out, const jit::array<Ti> &in, int lim) {
        for (int i = 0; i < lim; i++) { out[i] = std::abs(in[i]); }
    }
};

template<typename To>
struct UnOp<To, std::complex<float>, af_cast_t> {
    typedef std::complex<float> Ti;
    void eval(jit::array<To> &out, const jit::array<Ti> &in, int lim) {
        for (int i = 0; i < lim; i++) { out[i] = To(std::abs(in[i])); }
    }
};

template<typename To>
struct UnOp<To, std::complex<double>, af_cast_t> {
    typedef std::complex<double> Ti;
    void eval(jit::array<To> &out, const jit::array<Ti> &in, int lim) {
        for (int i = 0; i < lim; i++) { out[i] = To(std::abs(in[i])); }
    }
};

// DO NOT REMOVE THE TWO SPECIALIZATIONS BELOW
// These specializations are required because we partially specialize when
// Ti = std::complex<T> The partial specializations above expect output to
// be real. so they To(std::abs(v)) instead of To(v) which results in
// incorrect values when To is complex.

template<>
struct UnOp<std::complex<float>, std::complex<double>, af_cast_t> {
    typedef std::complex<double> Ti;
    typedef std::complex<float> To;
    void eval(jit::array<To> &out, const jit::array<Ti> &in, int lim) {
        for (int i = 0; i < lim; i++) { out[i] = To(in[i]); }
    }
};

template<>
struct UnOp<std::complex<double>, std::complex<float>, af_cast_t> {
    typedef std::complex<float> Ti;
    typedef std::complex<double> To;
    void eval(jit::array<To> &out, const jit::array<Ti> &in, int lim) {
        for (int i = 0; i < lim; i++) { out[i] = To(in[i]); }
    }
};

#define CAST_B8(T)                                                           \
    template<>                                                               \
    struct UnOp<char, T, af_cast_t> {                                        \
        void eval(jit::array<char> &out, const jit::array<T> &in, int lim) { \
            for (int i = 0; i < lim; i++) { out[i] = char(in[i] != 0); }     \
        }                                                                    \
    };

CAST_B8(float)
CAST_B8(double)
CAST_B8(int)
CAST_B8(uchar)
CAST_B8(char)

}  // namespace cpu
}  // namespace arrayfire
