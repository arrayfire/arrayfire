#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <fft.hpp>
#include <err_cpu.hpp>
#include <fftw3.h>
#include <algorithm>

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

#define TRANSFORM(FUNC, IN_T, OUT_T, IN_C, OUT_C, PREFIX, CALL, ...)\
    template<>                                                      \
    Array<OUT_T>* FUNC##w_common(Array<IN_T> const &in, int rank)   \
    {                                                               \
        int rank_dims[3];                                           \
        const dim4 dims = in.dims();                                \
        switch(rank) {                                              \
            case 1: computeDims<1>(rank_dims, dims); break;         \
            case 2: computeDims<2>(rank_dims, dims); break;         \
            case 3: computeDims<3>(rank_dims, dims); break;         \
        }                                                           \
        Array<OUT_T> *out = createEmptyArray<OUT_T>(dims);          \
        const dim4 istrides = in.strides();                         \
        const dim4 ostrides = out->strides();                       \
        PREFIX##_plan plan = PREFIX##_plan_many_##CALL  (           \
                                            rank,                   \
                                            rank_dims,              \
                                            (int)dims[rank],        \
                                            (IN_C*)in.get(),        \
                                            NULL, (int)istrides[0], \
                                            (int)istrides[rank],    \
                                            (OUT_C*)out->get(),     \
                                            NULL, (int)ostrides[0], \
                                            (int)ostrides[rank],    \
                                            __VA_ARGS__             \
                                                        );          \
        PREFIX##_execute(plan);                                     \
        PREFIX##_destroy_plan(plan);                                \
        return out;                                                 \
    }

template<typename inType, typename outType>
Array<outType>* fftw_common(Array<inType> const &in, int rank)
{
    CPU_NOT_SUPPORTED();
}

template<typename inType, typename outType>
Array<outType>* ifftw_common(Array<inType> const &in, int rank)
{
    CPU_NOT_SUPPORTED();
}

TRANSFORM(fft, float  , cfloat, float        , fftwf_complex, fftwf, dft_r2c, FFTW_ESTIMATE);
TRANSFORM(fft, cfloat , cfloat, fftwf_complex, fftwf_complex, fftwf, dft, FFTW_FORWARD, FFTW_ESTIMATE);
TRANSFORM(fft, double ,cdouble, double       , fftw_complex , fftw , dft_r2c, FFTW_ESTIMATE);
TRANSFORM(fft, cdouble,cdouble, fftw_complex , fftw_complex , fftw , dft, FFTW_FORWARD, FFTW_ESTIMATE);

TRANSFORM(ifft, cfloat ,  float, fftwf_complex, float        , fftwf, dft_c2r, FFTW_ESTIMATE);
TRANSFORM(ifft, cfloat , cfloat, fftwf_complex, fftwf_complex, fftwf, dft, FFTW_BACKWARD, FFTW_ESTIMATE);
TRANSFORM(ifft, cdouble, double, fftw_complex , double       , fftw , dft_c2r, FFTW_ESTIMATE);
TRANSFORM(ifft, cdouble,cdouble, fftw_complex , fftw_complex , fftw , dft, FFTW_BACKWARD, FFTW_ESTIMATE);

template<typename T, unsigned ndims>
void copyToPaddedArray(Array<T> &dst, const Array<T> &src)
{
    dim4 src_dims       = src.dims();
    dim4 dst_dims       = dst.dims();
    dim4 src_strides    = src.strides();
    dim4 dst_strides    = dst.strides();

    const T * src_ptr   = src.get();
    T * dst_ptr         = dst.get();

    dim_type trgt_l = std::min(dst_dims[3], src_dims[3]);
    dim_type trgt_k = std::min(dst_dims[2], src_dims[2]);
    dim_type trgt_j = std::min(dst_dims[1], src_dims[1]);
    dim_type trgt_i = std::min(dst_dims[0], src_dims[0]);

    for(dim_type l=0; l<trgt_l; ++l) {

        dim_type src_loff = l*src_strides[3];
        dim_type dst_loff = l*dst_strides[3];

        for(dim_type k=0; k<trgt_k; ++k) {

            dim_type src_koff = k*src_strides[2];
            dim_type dst_koff = k*dst_strides[2];

            for(dim_type j=0; j<trgt_j; ++j) {

                dim_type src_joff = j*src_strides[1];
                dim_type dst_joff = j*dst_strides[1];

                for(dim_type i=0; i<trgt_i; ++i) {
                    dim_type src_idx = i*src_strides[0] + src_joff + src_koff + src_loff;
                    dim_type dst_idx = i*dst_strides[0] + dst_joff + dst_koff + dst_loff;

                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            }
        }
    }
}

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

template<typename T, int rank>
Array<T> * createPaddedArray(Array<T> const &in, dim_type const npad, dim_type const * const pad)
{
    dim4 pdims(1);

    switch(rank) {
        case 1 : computePaddedDims<1>(pdims, pad); break;
        case 2 : computePaddedDims<2>(pdims, pad); break;
        case 3 : computePaddedDims<3>(pdims, pad); break;
        default: AF_ERROR("invalid rank", AF_ERR_SIZE);
    }
    pdims[rank] = in.dims()[rank];

    Array<T> *ret = createValueArray<T>(pdims,T(0));

    switch(rank) {
        case 1 : copyToPaddedArray<T, 1>(*ret, in); break;
        case 2 : copyToPaddedArray<T, 2>(*ret, in); break;
        case 3 : copyToPaddedArray<T, 3>(*ret, in); break;
        default: AF_ERROR("invalid rank", AF_ERR_SIZE);
    }

    return ret;
}

template<typename T>
void normalizeArray(Array<T> &src, double factor)
{
    T * src_ptr = src.get();
    for(dim_type i=0; i<src.elements(); ++i)
        src_ptr[i] *= factor;
}

template<typename inType, typename outType, int rank, bool isR2C>
Array<outType>* fft_helper(Array<inType> const &in)
{
    Array<outType> *ret = nullptr;
    if (isR2C) {
        Array<outType> *paddedCmplxArray = createComplexFromReal<outType, inType>(in);

        ret = fftw_common<outType, outType>(*paddedCmplxArray, rank);

        destroyArray(*paddedCmplxArray);
    } else {
        ret = fftw_common<inType, outType>(in, rank);
    }
    return ret;
}

template<typename inType, typename outType, int rank, bool isR2C>
Array<outType> * fft(Array<inType> const &in, double norm_factor, dim_type const npad, dim_type const * const pad)
{
    Array<outType> *ret = nullptr;
    if (npad>0) {
        Array<inType> *paddedArray = createPaddedArray<inType, rank>(in, npad, pad);

        ret = fft_helper<inType, outType, rank, isR2C>(*paddedArray);

        destroyArray(*paddedArray);
    } else {
        ret = fft_helper<inType, outType, rank, isR2C>(in);
    }
    normalizeArray(*ret, norm_factor);
    return ret;
}

template<typename inType, typename outType, int rank>
Array<outType> * ifft(Array<inType> const &in, double norm_factor, dim_type const npad, dim_type const * const pad)
{
    Array<outType> *ret = nullptr;
    if (npad>0) {
        Array<inType> *paddedArray = createPaddedArray<inType, rank>(in, npad, pad);

        ret = ifftw_common<inType, outType>(*paddedArray, rank);

        destroyArray(*paddedArray);
    } else {
        ret = ifftw_common<inType, outType>(in, rank);
    }
    normalizeArray(*ret, norm_factor);
    return ret;
}

#define INSTANTIATE1(T1, T2)\
    template Array<T2> * fft <T1, T2, 1, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 2, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 3, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 1, false>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 2, false>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 3, false>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad);

INSTANTIATE1(float  , cfloat )
INSTANTIATE1(double , cdouble)

#define INSTANTIATE2(T1, T2)\
    template Array<T2> * fft <T1, T2, 1, false>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 2, false>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 3, false>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * ifft<T1, T2, 1>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * ifft<T1, T2, 2>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad);\
    template Array<T2> * ifft<T1, T2, 3>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad);

INSTANTIATE2(cfloat , cfloat )
INSTANTIATE2(cdouble, cdouble)

#define INSTANTIATE3(T1, T2)\
    template Array<T2> * ifft<T1, T2, 1>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * ifft<T1, T2, 2>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * ifft<T1, T2, 3>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad);

INSTANTIATE3(cfloat , float )
INSTANTIATE3(cdouble, double)

}
