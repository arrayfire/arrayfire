/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fft.hpp>

#include <Array.hpp>
#include <copy.hpp>
#include <fftw3.h>
#include <platform.hpp>
#include <types.hpp>
#include <af/dim4.hpp>

#include <array>
#include <type_traits>

using af::dim4;
using std::array;

namespace arrayfire {
namespace cpu {

template<typename T>
struct fftw_transform;

#define TRANSFORM(PRE, TY)                                             \
    template<>                                                         \
    struct fftw_transform<TY> {                                        \
        typedef PRE##_plan plan_t;                                     \
        typedef PRE##_complex ctype_t;                                 \
                                                                       \
        template<typename... Args>                                     \
        plan_t create(Args... args) {                                  \
            return PRE##_plan_many_dft(args...);                       \
        }                                                              \
        void execute(plan_t plan) { return PRE##_execute(plan); }      \
        void destroy(plan_t plan) { return PRE##_destroy_plan(plan); } \
    };

TRANSFORM(fftwf, cfloat)
TRANSFORM(fftw, cdouble)

template<typename To, typename Ti>
struct fftw_real_transform;

#define TRANSFORM_REAL(PRE, To, Ti, POST)                              \
    template<>                                                         \
    struct fftw_real_transform<To, Ti> {                               \
        typedef PRE##_plan plan_t;                                     \
        typedef PRE##_complex ctype_t;                                 \
                                                                       \
        template<typename... Args>                                     \
        plan_t create(Args... args) {                                  \
            return PRE##_plan_many_dft_##POST(args...);                \
        }                                                              \
        void execute(plan_t plan) { return PRE##_execute(plan); }      \
        void destroy(plan_t plan) { return PRE##_destroy_plan(plan); } \
    };

TRANSFORM_REAL(fftwf, cfloat, float, r2c)
TRANSFORM_REAL(fftw, cdouble, double, r2c)
TRANSFORM_REAL(fftwf, float, cfloat, c2r)
TRANSFORM_REAL(fftw, double, cdouble, c2r)

inline array<int, AF_MAX_DIMS> computeDims(const int rank, const dim4 &idims) {
    array<int, AF_MAX_DIMS> retVal = {};
    for (int i = 0; i < rank; i++) { retVal[i] = idims[(rank - 1) - i]; }
    return retVal;
}

void setFFTPlanCacheSize(size_t numPlans) { UNUSED(numPlans); }

template<typename T>
void fft_inplace(Array<T> &in, const int rank, const bool direction) {
    auto func = [=](Param<T> in, const af::dim4 iDataDims) {
        const af::dim4 idims = in.dims();

        auto t_dims   = computeDims(rank, idims);
        auto in_embed = computeDims(rank, iDataDims);

        const af::dim4 istrides = in.strides();

        using ctype_t = typename fftw_transform<T>::ctype_t;
        typename fftw_transform<T>::plan_t plan;

        fftw_transform<T> transform;

        int batch = 1;
        for (int i = rank; i < 4; i++) { batch *= idims[i]; }

        plan = transform.create(
            rank, t_dims.data(), batch, reinterpret_cast<ctype_t *>(in.get()),
            in_embed.data(), static_cast<int>(istrides[0]),
            static_cast<int>(istrides[rank]),
            reinterpret_cast<ctype_t *>(in.get()), in_embed.data(),
            static_cast<int>(istrides[0]), static_cast<int>(istrides[rank]),
            direction ? FFTW_FORWARD : FFTW_BACKWARD,
            FFTW_ESTIMATE);  // NOLINT(hicpp-signed-bitwise)

        transform.execute(plan);
        transform.destroy(plan);
    };
    getQueue().enqueue(func, in, in.getDataDims());
}

template<typename Tc, typename Tr>
Array<Tc> fft_r2c(const Array<Tr> &in, const int rank) {
    dim4 odims    = in.dims();
    odims[0]      = odims[0] / 2 + 1;
    Array<Tc> out = createEmptyArray<Tc>(odims);

    auto func = [=](Param<Tc> out, const af::dim4 oDataDims, CParam<Tr> in,
                    const af::dim4 iDataDims) {
        af::dim4 idims = in.dims();

        auto t_dims    = computeDims(rank, idims);
        auto in_embed  = computeDims(rank, iDataDims);
        auto out_embed = computeDims(rank, oDataDims);

        const af::dim4 istrides = in.strides();
        const af::dim4 ostrides = out.strides();

        using ctype_t = typename fftw_real_transform<Tc, Tr>::ctype_t;
        using plan_t  = typename fftw_real_transform<Tc, Tr>::plan_t;
        plan_t plan;

        fftw_real_transform<Tc, Tr> transform;

        int batch = 1;
        for (int i = rank; i < 4; i++) { batch *= idims[i]; }

        plan = transform.create(
            rank, t_dims.data(), batch, const_cast<Tr *>(in.get()),
            in_embed.data(), static_cast<int>(istrides[0]),
            static_cast<int>(istrides[rank]),
            reinterpret_cast<ctype_t *>(out.get()), out_embed.data(),
            static_cast<int>(ostrides[0]), static_cast<int>(ostrides[rank]),
            FFTW_ESTIMATE);

        transform.execute(plan);
        transform.destroy(plan);
    };

    getQueue().enqueue(func, out, out.getDataDims(), in, in.getDataDims());

    return out;
}

template<typename Tr, typename Tc>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims, const int rank) {
    Array<Tr> out = createEmptyArray<Tr>(odims);

    auto func = [=](Param<Tr> out, const af::dim4 oDataDims, CParam<Tc> in,
                    const af::dim4 iDataDims, const af::dim4 odims) {
        auto t_dims    = computeDims(rank, odims);
        auto in_embed  = computeDims(rank, iDataDims);
        auto out_embed = computeDims(rank, oDataDims);

        const af::dim4 istrides = in.strides();
        const af::dim4 ostrides = out.strides();

        using ctype_t = typename fftw_real_transform<Tr, Tc>::ctype_t;
        using plan_t  = typename fftw_real_transform<Tr, Tc>::plan_t;
        plan_t plan;

        fftw_real_transform<Tr, Tc> transform;

        int batch = 1;
        for (int i = rank; i < 4; i++) { batch *= odims[i]; }

        // By default, fftw estimate flag is sufficient for most transforms.
        // However, complex to real transforms modify the input data memory
        // while performing the transformation. To avoid that, we need to pass
        // FFTW_PRESERVE_INPUT also. This flag however only works for 1D
        // transforms and for higher level transformations, a copy of input
        // data is passed onto the upstream FFTW calls.
        unsigned int flags = FFTW_ESTIMATE;  // NOLINT(hicpp-signed-bitwise)
        if (rank == 1) {
            flags |= FFTW_PRESERVE_INPUT;  // NOLINT(hicpp-signed-bitwise)
        }

        plan = transform.create(
            rank, t_dims.data(), batch,
            reinterpret_cast<ctype_t *>(const_cast<Tc *>(in.get())),
            in_embed.data(), static_cast<int>(istrides[0]),
            static_cast<int>(istrides[rank]), out.get(), out_embed.data(),
            static_cast<int>(ostrides[0]), static_cast<int>(ostrides[rank]),
            flags);

        transform.execute(plan);
        transform.destroy(plan);
    };

#ifdef USE_MKL
    getQueue().enqueue(func, out, out.getDataDims(), in, in.getDataDims(),
                       odims);
#else
    if (rank > 1 || odims.ndims() > 1) {
        // FFTW does not have a input preserving algorithm for multidimensional
        // c2r FFTs
        Array<Tc> in_ = copyArray<Tc>(in);
        getQueue().enqueue(func, out, out.getDataDims(), in_, in.getDataDims(),
                           odims);
    } else {
        getQueue().enqueue(func, out, out.getDataDims(), in, in.getDataDims(),
                           odims);
    }
#endif

    return out;
}

#define INSTANTIATE(T) \
    template void fft_inplace<T>(Array<T> &, const int, const bool);

INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

#define INSTANTIATE_REAL(Tr, Tc)                                             \
    template Array<Tc> fft_r2c<Tc, Tr>(const Array<Tr> &, const int);        \
    template Array<Tr> fft_c2r<Tr, Tc>(const Array<Tc> &in, const dim4 &odi, \
                                       const int);

INSTANTIATE_REAL(float, cfloat)
INSTANTIATE_REAL(double, cdouble)

}  // namespace cpu
}  // namespace arrayfire
