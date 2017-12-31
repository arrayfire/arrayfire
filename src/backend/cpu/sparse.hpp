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
#include <common/SparseArray.hpp>

#ifdef USE_MKL
#include <mkl_spblas.h>
#endif

namespace cpu
{

template<typename T, af_storage stype>
common::SparseArray<T> sparseConvertDenseToStorage(const Array<T> &in);

template<typename T, af_storage stype>
Array<T> sparseConvertStorageToDense(const common::SparseArray<T> &in);

template<typename T, af_storage src, af_storage dest>
common::SparseArray<T> sparseConvertStorageToStorage(const common::SparseArray<T> &in);

template<typename T, class Enable = void>
struct sparse_blas_base {
    using type = T;
};

template<typename T>
struct sparse_blas_base <T, typename std::enable_if<is_complex<T>::value>::type> {
  using type = typename std::conditional<std::is_same<T, cdouble>::value,
                                      cdouble, cfloat>
                                     ::type;
};

template<typename T>
using csparse_ptr_type     =   typename std::conditional< is_complex<T>::value,
                                                          const typename sparse_blas_base<T>::type *,
                                                          const T*>::type;
template<typename T>
using sparse_ptr_type     =    typename std::conditional<   is_complex<T>::value,
                                                            typename sparse_blas_base<T>::type *,
                                                            T*>::type;
template<typename T>
using sparse_scale_type   =    typename std::conditional<   is_complex<T>::value,
                                                            const typename sparse_blas_base<T>::type,
                                                            const T>::type;
}
