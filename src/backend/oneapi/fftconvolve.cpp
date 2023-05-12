/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fftconvolve.hpp>

#include <Array.hpp>
#include <common/dispatch.hpp>
#include <err_oneapi.hpp>
#include <fft.hpp>
#include <af/dim4.hpp>

#include <kernel/fftconvolve_common.hpp>
#include <kernel/fftconvolve_pack.hpp>
#include <kernel/fftconvolve_pad.hpp>
#include <kernel/fftconvolve_multiply.hpp>
#include <kernel/fftconvolve_reorder.hpp>

#include <cmath>
#include <type_traits>
#include <vector>

#include <handle.hpp>

using af::dim4;
using std::ceil;
using std::conditional;
using std::is_integral;
using std::is_same;
using std::vector;

namespace arrayfire {
namespace oneapi {

template<typename T>
dim4 calcPackedSize(Array<T> const& i1, Array<T> const& i2, const dim_t rank) {
    const dim4& i1d = i1.dims();
    const dim4& i2d = i2.dims();

    dim_t pd[4] = {1, 1, 1, 1};

    // Pack both signal and filter on same memory array, this will ensure
    // better use of batched cuFFT capabilities
    pd[0] = nextpow2(static_cast<unsigned>(
        static_cast<int>(ceil(i1d[0] / 2.f)) + i2d[0] - 1));

    for (dim_t k = 1; k < rank; k++) {
        pd[k] = nextpow2(static_cast<unsigned>(i1d[k] + i2d[k] - 1));
    }

    dim_t i1batch = 1;
    dim_t i2batch = 1;
    for (int k = rank; k < 4; k++) {
        i1batch *= i1d[k];
        i2batch *= i2d[k];
    }
    pd[rank] = (i1batch + i2batch);

    return dim4(pd[0], pd[1], pd[2], pd[3]);
}

#include <stdio.h>

af_err serializeArrayToFile(af_array arr, const char* filePath) {
    printf("writing %s\n", filePath);

    af_dtype ty; af_get_type(&ty, arr);
    size_t sizeOfTy; af_get_size_of(&sizeOfTy, ty);
    unsigned ndims; af_get_numdims(&ndims, arr);
    dim_t dims_dim_t[4]; af_get_dims(&(dims_dim_t[0]), &(dims_dim_t[1]),
                                     &(dims_dim_t[2]), &(dims_dim_t[3]), arr);
    unsigned dims[4];
    for (unsigned i = 0; i < 4; i++) dims[i] = dims_dim_t[i];

    unsigned elements = 1;
    for (unsigned i = 0; i < ndims; i++) elements *= dims[i];

    void* hostPtr = malloc(elements * sizeOfTy);
    af_get_data_ptr(hostPtr, arr);

    printf("%d (%d bytes)\n", ty, sizeOfTy);
    printf("%d\n", ndims);
    printf("%d %d %d %d\n", dims[0], dims[1], dims[2], dims[3]);

    FILE* outFile = fopen(filePath, "wb");
    fwrite(&ty, sizeof(af_dtype), 1, outFile);
    fwrite(&ndims, sizeof(unsigned), 1, outFile);
    fwrite(dims, sizeof(unsigned), 4, outFile);
    fwrite(hostPtr, sizeOfTy, elements, outFile);
    fclose(outFile);

    free(hostPtr);

    return AF_SUCCESS;
}

template<typename T>
Array<T> fftconvolve(Array<T> const& signal, Array<T> const& filter,
                    const bool expand, AF_BATCH_KIND kind, const int rank) {

      using convT = typename conditional<is_integral<T>::value ||
                                         is_same<T, float>::value ||
                                         is_same<T, cfloat>::value,
                                         float, double>::type;
      using cT    = typename conditional<is_same<convT, float>::value, cfloat,
                                         cdouble>::type;

      const dim4& sDims = signal.dims();
      const dim4& fDims = filter.dims();

      dim4 oDims(1);
      if (expand) {
        for (int d = 0; d < AF_MAX_DIMS; ++d) {
          if (kind == AF_BATCH_NONE || kind == AF_BATCH_RHS) {
            oDims[d] = sDims[d] + fDims[d] - 1;
          } else {
            oDims[d] = (d < rank ? sDims[d] + fDims[d] - 1 : sDims[d]);
          }
        }
      } else {
        oDims = sDims;
        if (kind == AF_BATCH_RHS) {
          for (int i = rank; i < AF_MAX_DIMS; ++i) { oDims[i] = fDims[i]; }
        }
      }

      const dim4 pDims = calcPackedSize<T>(signal, filter, rank);
      Array<cT> packed = createEmptyArray<cT>(pDims);

      serializeArrayToFile(getHandle(packed), "/tmp/oneapi-signal");
      serializeArrayToFile(getHandle(packed), "/tmp/oneapi-filter");
      printf("rank %d kind %d expand %d\n", rank, kind, expand);
      serializeArrayToFile(getHandle(packed), "/tmp/oneapi-packed-00");

      kernel::packDataHelper<cT, T>(packed, signal, filter, rank, kind);
      kernel::padDataHelper<cT, T>(packed, signal, filter, rank, kind);

      serializeArrayToFile(getHandle(packed), "/tmp/oneapi-packed-01");

      fft_inplace<cT>(packed, rank, true);
      serializeArrayToFile(getHandle(packed), "/tmp/oneapi-packed-02");
      kernel::complexMultiplyHelper<cT, T>(packed, signal, filter, rank, kind);

      serializeArrayToFile(getHandle(packed), "/tmp/oneapi-packed-03");

      // Compute inverse FFT only on complex-multiplied data
      if (kind == AF_BATCH_RHS) {
        vector<af_seq> seqs;
        for (int k = 0; k < AF_MAX_DIMS; k++) {
          if (k < rank) {
            seqs.push_back({0., static_cast<double>(pDims[k] - 1), 1.});
          } else if (k == rank) {
            seqs.push_back({1., static_cast<double>(pDims[k] - 1), 1.});
          } else {
            seqs.push_back({0., 0., 1.});
          }
        }

        Array<cT> subPacked = createSubArray<cT>(packed, seqs);
        fft_inplace<cT>(subPacked, rank, false);
      } else {
        vector<af_seq> seqs;
        for (int k = 0; k < AF_MAX_DIMS; k++) {
          if (k < rank) {
            seqs.push_back({0., static_cast<double>(pDims[k]) - 1, 1.});
          } else if (k == rank) {
            seqs.push_back({0., static_cast<double>(pDims[k] - 2), 1.});
          } else {
            seqs.push_back({0., 0., 1.});
          }
        }

        Array<cT> subPacked = createSubArray<cT>(packed, seqs);
        fft_inplace<cT>(subPacked, rank, false);
      }

      Array<T> out = createEmptyArray<T>(oDims);

      kernel::reorderOutputHelper<T, cT>(out, packed, signal, filter, rank, kind,
                                         expand);
      serializeArrayToFile(getHandle(out), "/tmp/oneapi-out-04");

      return out;
}

#define INSTANTIATE(T)                                                 \
    template Array<T> fftconvolve<T>(Array<T> const&, Array<T> const&, \
                                     const bool, AF_BATCH_KIND, const int);

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(uint)
INSTANTIATE(int)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(uintl)
INSTANTIATE(intl)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace oneapi
}  // namespace arrayfire
