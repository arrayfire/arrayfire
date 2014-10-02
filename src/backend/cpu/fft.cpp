#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <fft.hpp>
#include <err_cpu.hpp>
#include <fftw3.h>
#include <copy.hpp>

using af::dim4;

namespace cpu
{

template<int rank>
void computeDims(int *rdims, const dim4 &idims)
{
    if (rank==3) {
        rdims[0] = idims[2];
        rdims[1] = idims[1];
        rdims[2] = idims[0];
    } else if(rank==2) {
        rdims[0] = idims[1];
        rdims[1] = idims[0];
    } else {
        rdims[0] = idims[0];
    }
}

#define TRANSFORM(FUNC, T, CAST_T, PREFIX, DIRECTION)               \
    template<> void FUNC##w_common<T>(Array<T> &arr, int rank)      \
    {                                                               \
        int rank_dims[3];                                           \
        const dim4 dims = arr.dims();                               \
        switch(rank) {                                              \
            case 1: computeDims<1>(rank_dims, dims); break;         \
            case 2: computeDims<2>(rank_dims, dims); break;         \
            case 3: computeDims<3>(rank_dims, dims); break;         \
        }                                                           \
        const dim4 strides = arr.strides();                         \
        PREFIX##_plan plan = PREFIX##_plan_many_dft  (              \
                                            rank,                   \
                                            rank_dims,              \
                                            (int)dims[rank],        \
                                            (CAST_T*)arr.get(),     \
                                            NULL, (int)strides[0],  \
                                            (int)strides[rank],     \
                                            (CAST_T*)arr.get(),     \
                                            NULL, (int)strides[0],  \
                                            (int)strides[rank],     \
                                            DIRECTION,              \
                                            FFTW_ESTIMATE);         \
        PREFIX##_execute(plan);                                     \
        PREFIX##_destroy_plan(plan);                                \
    }

template<typename T>
void fftw_common(Array<T> &arr, int rank)
{
    CPU_NOT_SUPPORTED();
}

template<typename T>
void ifftw_common(Array<T> &arr, int rank)
{
    CPU_NOT_SUPPORTED();
}

TRANSFORM( fft,  cfloat, fftwf_complex, fftwf,  FFTW_FORWARD);
TRANSFORM( fft, cdouble, fftw_complex , fftw ,  FFTW_FORWARD);
TRANSFORM(ifft,  cfloat, fftwf_complex, fftwf, FFTW_BACKWARD);
TRANSFORM(ifft, cdouble, fftw_complex , fftw , FFTW_BACKWARD);

template<int rank>
void computePaddedDims(dim4 &pdims, dim_type const * const pad)
{
    if (rank==1) {
        pdims[0] = pad[0];
    } else if (rank==2) {
        pdims[0] = pad[0];
        pdims[1] = pad[1];
    } else if (rank==3) {
        pdims[0] = pad[0];
        pdims[1] = pad[1];
        pdims[2] = pad[2];
    }
}

template<typename inType, typename outType, int rank, bool isR2C>
Array<outType> * fft(Array<inType> const &in, double norm_factor, dim_type const npad, dim_type const * const pad)
{
    ARG_ASSERT(1, ((in.isOwner()==true) && "fft: Sub-Arrays not supported yet."));

    dim4 pdims(1);

    switch(rank) {
        case 1 : computePaddedDims<1>(pdims, pad); break;
        case 2 : computePaddedDims<2>(pdims, pad); break;
        case 3 : computePaddedDims<3>(pdims, pad); break;
        default: AF_ERROR("invalid rank", AF_ERR_SIZE);
    }

    pdims[rank] = in.dims()[rank];

    Array<outType> *ret = createPaddedArray<inType, outType>(in, (npad>0 ? pdims : in.dims()));

    fftw_common<outType>(*ret, rank);

    scaleArray(*ret, norm_factor);

    return ret;
}

template<typename T, int rank>
Array<T> * ifft(Array<T> const &in, double norm_factor, dim_type const npad, dim_type const * const pad)
{
    ARG_ASSERT(1, ((in.isOwner()==true) && "ifft: Sub-Arrays not supported yet."));

    dim4 pdims(1);

    switch(rank) {
        case 1 : computePaddedDims<1>(pdims, pad); break;
        case 2 : computePaddedDims<2>(pdims, pad); break;
        case 3 : computePaddedDims<3>(pdims, pad); break;
        default: AF_ERROR("invalid rank", AF_ERR_SIZE);
    }

    pdims[rank] = in.dims()[rank];

    Array<T> *ret = createPaddedArray<T, T>(in, (npad>0 ? pdims : in.dims()));

    ifftw_common<T>(*ret, rank);

    scaleArray(*ret, norm_factor);

    return ret;
}

#define INSTANTIATE1(T1, T2)\
    template Array<T2> * fft <T1, T2, 1, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 2, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 3, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad);

INSTANTIATE1(float  , cfloat )
INSTANTIATE1(double , cdouble)

#define INSTANTIATE2(T)\
    template Array<T> * fft <T, T, 1, false>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * fft <T, T, 2, false>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * fft <T, T, 3, false>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * ifft<T, 1>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * ifft<T, 2>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * ifft<T, 3>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad);

INSTANTIATE2(cfloat )
INSTANTIATE2(cdouble)

}
