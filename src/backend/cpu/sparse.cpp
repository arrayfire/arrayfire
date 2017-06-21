/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse.hpp>
#include <kernel/sparse.hpp>

#include <stdexcept>
#include <string>

#include <arith.hpp>
#include <cast.hpp>
#include <complex.hpp>
#include <copy.hpp>
#include <err_common.hpp>
#include <lookup.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <reduce.hpp>
#include <where.hpp>

namespace cpu
{

using namespace common;

using std::add_const;
using std::add_pointer;
using std::enable_if;
using std::is_floating_point;
using std::remove_const;
using std::conditional;
using std::is_same;

template<typename T, class Enable = void>
struct blas_base {
    using type = T;
};

template<typename T>
struct blas_base <T, typename enable_if<is_complex<T>::value>::type> {
    using type = typename conditional<is_same<T, cdouble>::value,
                                      sp_cdouble, sp_cfloat>
                                     ::type;
};

template<typename T>
using cptr_type     =   typename conditional<   is_complex<T>::value,
                                                const typename blas_base<T>::type *,
                                                const T*>::type;
template<typename T>
using ptr_type     =    typename conditional<   is_complex<T>::value,
                                                typename blas_base<T>::type *,
                                                T*>::type;
template<typename T>
using scale_type   =    typename conditional<   is_complex<T>::value,
                                                const typename blas_base<T>::type *,
                                                const T *>::type;

#ifdef USE_MKL

// void mkl_zdnscsr (const MKL_INT *job ,
//                   const MKL_INT *m , const MKL_INT *n ,
//                   MKL_Complex16 *adns , const MKL_INT *lda ,
//                   MKL_Complex16 *acsr ,
//                   MKL_INT *ja , MKL_INT *ia ,
//                   MKL_INT *info );
template<typename T>
using dnscsr_func_def = void (*)(const int *,
                                 const int *, const int *,
                                 ptr_type<T>, const int *,
                                 ptr_type<T>,
                                 int *, int *,
                                 int *);

//void mkl_zcsrcsc (const MKL_INT *job ,
//                  const MKL_INT *n ,
//                  MKL_Complex16 *acsr ,
//                  MKL_INT *ja , MKL_INT *ia ,
//                  MKL_Complex16 *acsc ,
//                  MKL_INT *ja1 , MKL_INT *ia1 ,
//                  MKL_INT *info );
template<typename T>
using csrcsc_func_def = void (*)(const int *,
                                 const int *,
                                 ptr_type<T>, int *, int *,
                                 ptr_type<T>,
                                 int *, int *,
                                 int *);

#define SPARSE_FUNC_DEF( FUNC )                         \
template<typename T> FUNC##_func_def<T> FUNC##_func();

#define SPARSE_FUNC( FUNC, TYPE, PREFIX )               \
  template<> FUNC##_func_def<TYPE> FUNC##_func<TYPE>()  \
{ return &mkl_##PREFIX##FUNC; }

SPARSE_FUNC_DEF(dnscsr)
SPARSE_FUNC(dnscsr, float,  s)
SPARSE_FUNC(dnscsr, double, d)
SPARSE_FUNC(dnscsr, cfloat, c)
SPARSE_FUNC(dnscsr, cdouble,z)

SPARSE_FUNC_DEF(csrcsc)
SPARSE_FUNC(csrcsc, float,  s)
SPARSE_FUNC(csrcsc, double, d)
SPARSE_FUNC(csrcsc, cfloat, c)
SPARSE_FUNC(csrcsc, cdouble,z)

#undef SPARSE_FUNC
#undef SPARSE_FUNC_DEF

#endif // USE_MKL

////////////////////////////////////////////////////////////////////////////////
// Common Funcs for MKL and Non-MKL Code Paths
////////////////////////////////////////////////////////////////////////////////

// Partial template specialization of sparseConvertDenseToStorage for COO
// However, template specialization is not allowed
template<typename T>
SparseArray<T> sparseConvertDenseToCOO(const Array<T> &in)
{
    in.eval();

    Array<uint> nonZeroIdx_ = where<T>(in);
    Array<int> nonZeroIdx = cast<int, uint>(nonZeroIdx_);

    dim_t nNZ = nonZeroIdx.elements();

    Array<int> constDim = createValueArray<int>(dim4(nNZ), in.dims()[0]);
    constDim.eval();

    Array<int> rowIdx = arithOp<int, af_mod_t>(nonZeroIdx, constDim, nonZeroIdx.dims());
    Array<int> colIdx = arithOp<int, af_div_t>(nonZeroIdx, constDim, nonZeroIdx.dims());

    Array<T> values = copyArray<T>(in);
    values.modDims(dim4(values.elements()));
    values = lookup<T, int>(values, nonZeroIdx, 0);

    return createArrayDataSparseArray<T>(in.dims(), values, rowIdx, colIdx, AF_STORAGE_COO);
}

// Partial template specialization of sparseConvertStorageToDense for COO
// However, template specialization is not allowed
template<typename T>
Array<T> sparseConvertCOOToDense(const SparseArray<T> &in)
{
    in.eval();

    Array<T> dense = createValueArray<T>(in.dims(), scalar<T>(0));
    dense.eval();

    const Array<T>   values = in.getValues();
    const Array<int> rowIdx = in.getRowIdx();
    const Array<int> colIdx = in.getColIdx();

    getQueue().enqueue(kernel::coo2dense<T>, dense, values, rowIdx, colIdx);

    return dense;
}

////////////////////////////////////////////////////////////////////////////////
#ifdef USE_MKL // Implementation using MKL
////////////////////////////////////////////////////////////////////////////////

template<typename T, af_storage stype>
SparseArray<T> sparseConvertDenseToStorage(const Array<T> &in_)
{
    in_.eval();

    // MKL only has dns->csr.
    // CSR <-> CSC is only supported if input is square
    uint nNZ = reduce_all<af_notzero_t, T, uint>(in_);

    SparseArray<T> sparse_ = createEmptySparseArray<T>(in_.dims(), nNZ, AF_STORAGE_CSR);
    sparse_.eval();

    auto func = [=] (Param<T> values, Param<int> rowIdx, Param<int> colIdx, int num, CParam<T> in) {
        // Read: https://software.intel.com/en-us/node/520848
        // But job description is incorrect with regards to job[1]
        // 0 implies row major and 1 implies column major
        int j1 = 1, j2 = 0;
        const int job[] = {0, j1, j2, 2, num, 1};

        const int M = in.dims[0];
        const int N = in.dims[1];

        int ldd = in.strides[1];

        int info = 0;

        // Have to mess up all const correctness because MKL dnscsr function
        // is bidirectional and has input/output on all pointers
        dnscsr_func<T>()(
                job, &M, &N,
                reinterpret_cast<ptr_type<T>>(const_cast<T*>(in.get())), &ldd,
                reinterpret_cast<ptr_type<T>>(values.get()),
                colIdx.get(),
                rowIdx.get(),
                &info);
    };


    Array<T  > &values = sparse_.getValues();
    Array<int> &rowIdx = sparse_.getRowIdx();
    Array<int> &colIdx = sparse_.getColIdx();

    getQueue().enqueue(func, values, rowIdx, colIdx, (int)sparse_.elements(), in_);

    if(stype == AF_STORAGE_CSR)
        return sparse_;
    else
        AF_ERROR("CPU Backend only supports Dense to CSR or COO", AF_ERR_NOT_SUPPORTED);

    return sparse_;
}

template<typename T, af_storage stype>
Array<T> sparseConvertStorageToDense(const SparseArray<T> &in_)
{
    // MKL only has dns<->csr.
    // CSR <-> CSC is only supported if input is square

    if(stype == AF_STORAGE_CSC)
        AF_ERROR("CPU Backend only supports Dense to CSR or COO", AF_ERR_NOT_SUPPORTED);

    in_.eval();

    Array<T> dense_ = createValueArray<T>(in_.dims(), scalar<T>(0));
    dense_.eval();

    auto func = [=] (Param<T> dense,
                     CParam<T> values, CParam<int> rowIdx,
                     CParam<int> colIdx, int num) {
        // Read: https://software.intel.com/en-us/node/520848
        // But job description is incorrect with regards to job[1]
        // 0 implies row major and 1 implies column major
        int j1 = 1, j2 = 0;
        const int job[] = {1, j1, j2, 2, num, 1};

        const int M = dense.dims()[0];
        const int N = dense.dims()[1];

        int ldd = dense.strides()[1];

        int info = 0;

        // Have to mess up all const correctness because MKL dnscsr function
        // is bidirectional and has input/output on all pointers
        dnscsr_func<T>()(
                job, &M, &N,
                reinterpret_cast<ptr_type<T>>(dense.get()), &ldd,
                reinterpret_cast<ptr_type<T>>(const_cast<T*>(values.get())),
                const_cast<int*>(colIdx.get()),
                const_cast<int*>(rowIdx.get()),
                &info);
    };

    Array<T  > values = in_.getValues();
    Array<int> rowIdx = in_.getRowIdx();
    Array<int> colIdx = in_.getColIdx();

    getQueue().enqueue(func, dense_, values, rowIdx, colIdx, (int)in_.elements());

    if(stype == AF_STORAGE_CSR)
        return dense_;
    else
        AF_ERROR("CPU Backend only supports Dense to CSR or COO", AF_ERR_NOT_SUPPORTED);

    return dense_;
}

////////////////////////////////////////////////////////////////////////////////
#else // Implementation without using MKL
////////////////////////////////////////////////////////////////////////////////

template<typename T, af_storage stype>
SparseArray<T> sparseConvertDenseToStorage(const Array<T> &in_)
{
    in_.eval();

    uint nNZ = reduce_all<af_notzero_t, T, uint>(in_);

    SparseArray<T> sparse_ = createEmptySparseArray<T>(in_.dims(), nNZ, AF_STORAGE_CSR);
    sparse_.eval();

    Array<T  > values = sparse_.getValues();
    Array<int> rowIdx = sparse_.getRowIdx();
    Array<int> colIdx = sparse_.getColIdx();

    if(stype == AF_STORAGE_CSR)
        getQueue().enqueue(kernel::dense_csr<T>, values, rowIdx, colIdx, in_);
    else
        AF_ERROR("CPU Backend only supports Dense to CSR or COO", AF_ERR_NOT_SUPPORTED);


    return sparse_;
}

template<typename T, af_storage stype>
Array<T> sparseConvertStorageToDense(const SparseArray<T> &in_)
{
    in_.eval();

    Array<T> dense_ = createValueArray<T>(in_.dims(), scalar<T>(0));
    dense_.eval();

    Array<T  > values = in_.getValues();
    Array<int> rowIdx = in_.getRowIdx();
    Array<int> colIdx = in_.getColIdx();

    if(stype == AF_STORAGE_CSR)
        getQueue().enqueue(kernel::csr_dense<T>, dense_, values, rowIdx, colIdx);
    else
        AF_ERROR("CPU Backend only supports CSR or COO to Dense", AF_ERR_NOT_SUPPORTED);

    return dense_;
}

////////////////////////////////////////////////////////////////////////////////
#endif //USE_MKL
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Common Funcs for MKL and Non-MKL Code Paths
////////////////////////////////////////////////////////////////////////////////
template<typename T, af_storage dest, af_storage src>
SparseArray<T> sparseConvertStorageToStorage(const SparseArray<T> &in)
{
    in.eval();

    SparseArray<T> converted = createEmptySparseArray<T>(in.dims(), (int)in.getNNZ(), dest);
    converted.eval();

    function<void (Param<T>, Param<int>, Param<int>,
                   CParam<T>, CParam<int>, CParam<int>)
            > converter;

    if(src == AF_STORAGE_CSR && dest == AF_STORAGE_COO) {
        converter = kernel::csr_coo<T>;
    } else if (src == AF_STORAGE_COO && dest == AF_STORAGE_CSR) {
        converter = kernel::coo_csr<T>;
    } else {
        // Should never come here
        AF_ERROR("CPU Backend invalid conversion combination", AF_ERR_NOT_SUPPORTED);
    }

    getQueue().enqueue(converter,
                       converted.getValues(), converted.getRowIdx(), converted.getColIdx(),
                       in.getValues(), in.getRowIdx(), in.getColIdx());

    return converted;
}

#define INSTANTIATE_TO_STORAGE(T, S)                                                                        \
    template SparseArray<T> sparseConvertStorageToStorage<T, S, AF_STORAGE_CSR>(const SparseArray<T> &in);  \
    template SparseArray<T> sparseConvertStorageToStorage<T, S, AF_STORAGE_CSC>(const SparseArray<T> &in);  \
    template SparseArray<T> sparseConvertStorageToStorage<T, S, AF_STORAGE_COO>(const SparseArray<T> &in);  \

#define INSTANTIATE_COO_SPECIAL(T)                                                                      \
    template<> SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_COO>(const Array<T> &in)        \
    { return sparseConvertDenseToCOO<T>(in); }                                                          \
    template<> Array<T> sparseConvertStorageToDense<T, AF_STORAGE_COO>(const SparseArray<T> &in)        \
    { return sparseConvertCOOToDense<T>(in); }                                                          \

#define INSTANTIATE_SPARSE(T)                                                                           \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_CSR>(const Array<T> &in);         \
    template SparseArray<T> sparseConvertDenseToStorage<T, AF_STORAGE_CSC>(const Array<T> &in);         \
                                                                                                        \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_CSR>(const SparseArray<T> &in);         \
    template Array<T> sparseConvertStorageToDense<T, AF_STORAGE_CSC>(const SparseArray<T> &in);         \
                                                                                                        \
    INSTANTIATE_COO_SPECIAL(T)                                                                          \
                                                                                                        \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_CSR)                                                           \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_CSC)                                                           \
    INSTANTIATE_TO_STORAGE(T, AF_STORAGE_COO)                                                           \


INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

#undef INSTANTIATE_TO_STORAGE
#undef INSTANTIATE_COO_SPECIAL
#undef INSTANTIATE_SPARSE

}
