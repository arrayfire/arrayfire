/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fft.hpp>

#include <common/dispatch.hpp>
#include <copy.hpp>
#include <err_oneapi.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <onefft.hpp>
#include <platform.hpp>
#include <af/dim4.hpp>

#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/exceptions.hpp>

#include <cstdint>
#include <memory>

using std::make_shared;

using af::dim4;

namespace arrayfire {
namespace oneapi {

void setFFTPlanCacheSize(size_t numPlans) {}

std::string genPlanHashStr(int rank, ::oneapi::mkl::dft::precision precision,
                           ::oneapi::mkl::dft::domain domain,
                           const bool isInPlace, const dim_t *n,
                           std::int64_t *istrides, int ibatch,
                           std::int64_t *ostrides, int obatch, int nbatch) {
    // create the key string
    char key_str_temp[64];
    sprintf(key_str_temp, "%d:", rank);

    std::string key_string(key_str_temp);

    if (precision == ::oneapi::mkl::dft::precision::SINGLE) {
        key_string.append("S:");
    } else if (precision == ::oneapi::mkl::dft::precision::DOUBLE) {
        key_string.append("D:");
    }
    if (domain == ::oneapi::mkl::dft::domain::REAL) {
        key_string.append("R:");
    } else if (domain == ::oneapi::mkl::dft::domain::COMPLEX) {
        key_string.append("C:");
    }
    if (isInPlace) {
        key_string.append("IIP:");
    } else {
        key_string.append("OOP:");
    }

    for (int r = 0; r < rank; ++r) {
        sprintf(key_str_temp, "%lld:", n[r]);
        key_string.append(std::string(key_str_temp));
    }

    if (istrides != nullptr) {
        for (int r = 0; r < rank + 1; ++r) {
            sprintf(key_str_temp, "%ld:", istrides[r]);
            key_string.append(std::string(key_str_temp));
        }
        sprintf(key_str_temp, "%d:", ibatch);
        key_string.append(std::string(key_str_temp));
    }

    if (ostrides != nullptr) {
        for (int r = 0; r < rank + 1; ++r) {
            sprintf(key_str_temp, "%ld:", ostrides[r]);
            key_string.append(std::string(key_str_temp));
        }
        sprintf(key_str_temp, "%d:", obatch);
        key_string.append(std::string(key_str_temp));
    }

    sprintf(key_str_temp, "%d", nbatch);
    key_string.append(std::string(key_str_temp));

    return key_string;
}

std::vector<std::int64_t> computeStrides(const int rank, const dim4 istrides,
                                         const dim_t offset) {
    if (rank == 2) return {offset, istrides[1], istrides[0]};
    if (rank == 3) return {offset, istrides[2], istrides[1], istrides[0]};
    if (rank == 4)
        return {offset, istrides[3], istrides[2], istrides[1], istrides[0]};
    return {offset, istrides[0]};
}

template<::oneapi::mkl::dft::precision precision,
         ::oneapi::mkl::dft::domain domain>
PlanType findPlan(int rank, const bool isInPlace, const dim_t *idims,
                  std::int64_t *istrides, int ibatch, std::int64_t *ostrides,
                  int obatch, int nbatch) {
    using desc_ty = ::oneapi::mkl::dft::descriptor<precision, domain>;

    std::string key_string =
        genPlanHashStr(rank, precision, domain, isInPlace, idims, istrides,
                       ibatch, ostrides, obatch, nbatch);

    PlanCache &planner               = arrayfire::oneapi::fftManager();
    std::shared_ptr<PlanType> retVal = (planner.find(key_string));
    if (retVal) { return *retVal; }

    desc_ty *desc = [rank, &idims]() {
        if (rank == 1) return new desc_ty(static_cast<int64_t>(idims[0]));
        if (rank == 2) return new desc_ty({idims[1], idims[0]});
        if (rank == 3) return new desc_ty({idims[2], idims[1], idims[0]});
        return new desc_ty({idims[3], idims[2], idims[1], idims[0]});
    }();

    if (rank > 1) {
        desc->set_value(::oneapi::mkl::dft::config_param::INPUT_STRIDES,
                        istrides);
        desc->set_value(::oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                        ostrides);
    }

    if (isInPlace) {
        desc->set_value(::oneapi::mkl::dft::config_param::PLACEMENT,
                        DFTI_INPLACE);
    } else {
        desc->set_value(::oneapi::mkl::dft::config_param::PLACEMENT,
                        DFTI_NOT_INPLACE);
    }

    desc->set_value(::oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                    (int64_t)nbatch);

    desc->set_value(::oneapi::mkl::dft::config_param::FWD_DISTANCE, ibatch);
    desc->set_value(::oneapi::mkl::dft::config_param::BWD_DISTANCE, obatch);

    if constexpr (domain == ::oneapi::mkl::dft::domain::COMPLEX) {
        desc->set_value(::oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                        DFTI_COMPLEX_COMPLEX);
    } else {
        desc->set_value(
            ::oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
            DFTI_COMPLEX_COMPLEX);
        desc->set_value(::oneapi::mkl::dft::config_param::PACKED_FORMAT,
                        DFTI_CCE_FORMAT);
    }

    try {
        desc->commit(getQueue());
    } catch (::oneapi::mkl::device_bad_alloc &e) {
        // If plan creation fails, clean up the memory we hold on to and try
        // again
        arrayfire::oneapi::signalMemoryCleanup();
        desc->commit(getQueue());
    }

    // push the plan into plan cache
    std::shared_ptr<void> ptr(desc);
    planner.push(key_string, make_shared<PlanType>(ptr));
    return ptr;
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

    // TODO[STF]: WTF
    // getOffset() for s0 throwing Invalid Descriptor when targeting gpu
    // on CPU, results are wrong but does not throw
    // strides not working? TODO: test standalone oneMKL
    // perhaps in.getDataDims() needed instead of in.dims()?
    std::vector<std::int64_t> fft_input_strides =
        computeStrides(rank, istrides, 0);
    // computeStrides(rank, istrides, in.getOffset()); //TODO[STF]: WTF,
    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= idims[i]; }

    const bool isInPlace = true;
    PlanType descP = findPlan<precision, ::oneapi::mkl::dft::domain::COMPLEX>(
        rank, isInPlace, idims.get(), fft_input_strides.data(), istrides[rank],
        fft_input_strides.data(), istrides[rank], batch);

    desc_ty *desc = (desc_ty *)descP.get();

    if (direction)
        ::oneapi::mkl::dft::compute_forward(*desc, *in.get());
    else
        ::oneapi::mkl::dft::compute_backward(*desc, *in.get());
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

    std::vector<std::int64_t> fft_input_strides =
        computeStrides(rank, istrides, in.getOffset());
    std::vector<std::int64_t> fft_output_strides =
        computeStrides(rank, ostrides, out.getOffset());

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= idims[i]; }

    const bool isInPlace = false;
    PlanType descP = findPlan<precision, ::oneapi::mkl::dft::domain::REAL>(
        rank, isInPlace, idims.get(), fft_input_strides.data(), istrides[rank],
        fft_output_strides.data(), ostrides[rank], batch);

    desc_ty *desc = (desc_ty *)descP.get();

    ::oneapi::mkl::dft::compute_forward(*desc, *in.get(), *out.get());

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

    std::vector<std::int64_t> fft_input_strides =
        computeStrides(rank, istrides, in.getOffset());
    std::vector<std::int64_t> fft_output_strides =
        computeStrides(rank, ostrides, out.getOffset());

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= odims[i]; }

    const bool isInPlace = false;
    PlanType descP = findPlan<precision, ::oneapi::mkl::dft::domain::REAL>(
        rank, isInPlace, odims.get(), fft_input_strides.data(), ostrides[rank],
        fft_output_strides.data(), istrides[rank], batch);

    desc_ty *desc = (desc_ty *)descP.get();

    ::oneapi::mkl::dft::compute_backward(*desc, *in.get(), *out.get());
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
