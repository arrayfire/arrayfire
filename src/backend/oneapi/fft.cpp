/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/dispatch.hpp>

#include <fft.hpp>

#include <copy.hpp>
#include <err_oneapi.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <af/dim4.hpp>

#include <array>
using std::array;

using af::dim4;

#include <oneapi/mkl/dfti.hpp>

namespace arrayfire {
namespace oneapi {

void setFFTPlanCacheSize(size_t numPlans) {}

inline array<int, AF_MAX_DIMS> computeDims(const int rank, const dim4 &idims) {
    array<int, AF_MAX_DIMS> retVal = {};
    for (int i = 0; i < rank; i++) { retVal[i] = idims[(rank - 1) - i]; }
    return retVal;
}

template<typename T>
void fft_inplace(Array<T> &in, const int rank, const bool direction) {
    const dim4 idims    = in.dims();
    const dim4 istrides = in.strides();

    constexpr bool is_single = std::is_same_v<T, cfloat>;
    constexpr auto precision = (is_single)
                                   ? ::oneapi::mkl::dft::precision::SINGLE
                                   : ::oneapi::mkl::dft::precision::DOUBLE;
    using desc_ty =
        ::oneapi::mkl::dft::descriptor<precision,
                                       ::oneapi::mkl::dft::domain::COMPLEX>;

    auto desc = [rank, &idims]() {
        if (rank == 1) return desc_ty(idims[0]);
        if (rank == 2) return desc_ty({idims[1], idims[0]});
        if (rank == 3) return desc_ty({idims[2], idims[1], idims[0]});
        return desc_ty({idims[3], idims[2], idims[1], idims[0]});
    }();

    desc.set_value(::oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= idims[i]; }
    desc.set_value(::oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                   (int64_t)batch);

    desc.set_value(::oneapi::mkl::dft::config_param::BWD_DISTANCE,
                   istrides[rank]);
    desc.set_value(::oneapi::mkl::dft::config_param::FWD_DISTANCE,
                   istrides[rank]);

    desc.commit(getQueue());
    if (direction)
        ::oneapi::mkl::dft::compute_forward(desc, *in.get());
    else
        ::oneapi::mkl::dft::compute_backward(desc, *in.get());
}

template<typename Tc, typename Tr>
Array<Tc> fft_r2c(const Array<Tr> &in, const int rank) {
    const dim4 idims    = in.dims();
    const dim4 istrides = in.strides();
    Array<Tc> out       = createEmptyArray<Tc>(
        dim4({idims[0] / 2 + 1, idims[1], idims[2], idims[3]}));
    const dim4 ostrides = out.strides();

    constexpr bool is_single = std::is_same_v<Tr, float>;
    constexpr auto precision = (is_single)
                                   ? ::oneapi::mkl::dft::precision::SINGLE
                                   : ::oneapi::mkl::dft::precision::DOUBLE;
    using desc_ty =
        ::oneapi::mkl::dft::descriptor<precision,
                                       ::oneapi::mkl::dft::domain::REAL>;

    auto desc = [rank, &idims]() {
        if (rank == 1) return desc_ty(idims[0]);
        if (rank == 2) return desc_ty({idims[1], idims[0]});
        if (rank == 3) return desc_ty({idims[2], idims[1], idims[0]});
        return desc_ty({idims[3], idims[2], idims[1], idims[0]});
    }();

    desc.set_value(::oneapi::mkl::dft::config_param::PLACEMENT,
                   DFTI_NOT_INPLACE);

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= idims[i]; }
    desc.set_value(::oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                   (int64_t)batch);

    desc.set_value(::oneapi::mkl::dft::config_param::BWD_DISTANCE,
                   ostrides[rank]);
    desc.set_value(::oneapi::mkl::dft::config_param::FWD_DISTANCE,
                   istrides[rank]);

    const std::int64_t fft_output_strides[5] = {
        0, ostrides[(rank == 2) ? 1 : 0], ostrides[(rank == 2) ? 0 : 1],
        ostrides[2], ostrides[3]};
    desc.set_value(::oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                   fft_output_strides, rank);

    desc.commit(getQueue());
    ::oneapi::mkl::dft::compute_forward(desc, *in.get(), *out.get());

    return out;
}

template<typename Tr, typename Tc>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims, const int rank) {
    const dim4 idims    = in.dims();
    const dim4 istrides = in.strides();
    Array<Tr> out       = createEmptyArray<Tr>(odims);
    const dim4 ostrides = out.strides();

    constexpr bool is_single = std::is_same_v<Tr, float>;
    constexpr auto precision = (is_single)
                                   ? ::oneapi::mkl::dft::precision::SINGLE
                                   : ::oneapi::mkl::dft::precision::DOUBLE;
    using desc_ty =
        ::oneapi::mkl::dft::descriptor<precision,
                                       ::oneapi::mkl::dft::domain::REAL>;

    auto desc = [rank, &odims]() {
        if (rank == 1) return desc_ty(odims[0]);
        if (rank == 2) return desc_ty({odims[1], odims[0]});
        if (rank == 3) return desc_ty({odims[2], odims[1], odims[0]});
        return desc_ty({odims[3], odims[2], odims[1], odims[0]});
    }();

    desc.set_value(::oneapi::mkl::dft::config_param::PLACEMENT,
                   DFTI_NOT_INPLACE);

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= idims[i]; }
    desc.set_value(::oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                   (int64_t)batch);

    desc.set_value(::oneapi::mkl::dft::config_param::BWD_DISTANCE,
                   istrides[rank]);
    desc.set_value(::oneapi::mkl::dft::config_param::FWD_DISTANCE,
                   ostrides[rank]);

    const std::int64_t fft_output_strides[5] = {
        0, ostrides[(rank == 2) ? 1 : 0], ostrides[(rank == 2) ? 0 : 1],
        ostrides[2], ostrides[3]};
    desc.set_value(::oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                   fft_output_strides, rank);

    desc.commit(getQueue());
    ::oneapi::mkl::dft::compute_backward(desc, *in.get(), *out.get());
    return out;
}

#define INSTANTIATE(T) \
    template void fft_inplace<T>(Array<T> &, const int, const bool);

INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

#define INSTANTIATE_REAL(Tr, Tc)                                        \
    template Array<Tc> fft_r2c<Tc, Tr>(const Array<Tr> &, const int);   \
    template Array<Tr> fft_c2r<Tr, Tc>(const Array<Tc> &, const dim4 &, \
                                       const int);

INSTANTIATE_REAL(float, cfloat)
INSTANTIATE_REAL(double, cdouble)
}  // namespace oneapi
}  // namespace arrayfire
