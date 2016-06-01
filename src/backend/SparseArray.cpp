/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <SparseArray.hpp>
#include <backend.hpp>
#include <copy.hpp>
#include <math.hpp>
#include <platform.hpp>

namespace common
{

using namespace detail;

////////////////////////////////////////////////////////////////////////////
// Sparse Array Base Implementations
////////////////////////////////////////////////////////////////////////////

// ROW_LENGTH and column length expect standard variable names of
// SparseArraBase::storage
// _nNZ  -> Constructor Argument
// _dims -> Constructor Argument
#define ROW_LENGTH ((storage == AF_SPARSE_COO || storage == AF_SPARSE_CSC) ? _nNZ : (_dims[0] + 1))
#define COL_LENGTH ((storage == AF_SPARSE_COO || storage == AF_SPARSE_CSR) ? _nNZ : (_dims[1] + 1))

SparseArrayBase::SparseArrayBase(af::dim4 _dims, dim_t _nNZ, af::sparseStorage _storage, af_dtype _type):
    info(getActiveDeviceId(), _dims, 0, calcStrides(_dims), _type, true),
    storage(_storage),
    rowIdx(createValueArray<int>(dim4(ROW_LENGTH), 0)),
    colIdx(createValueArray<int>(dim4(COL_LENGTH), 0))
{
#if __cplusplus > 199711l
    static_assert(offsetof(SparseArrayBase, info) == 0,
                  "SparseArrayBase::info must be the first member variable of SparseArrayBase.");
#endif
}

SparseArrayBase::SparseArrayBase(af::dim4 _dims, dim_t _nNZ,
                    const int * const _rowIdx, const int * const _colIdx,
                    const af::sparseStorage _storage, af_dtype _type,
                    bool _is_device, bool _copy_device):
    info(getActiveDeviceId(), _dims, 0, calcStrides(_dims), _type, true),
    storage(_storage),
    rowIdx(_is_device ?
          (!_copy_device ? createDeviceDataArray<int>(dim4(ROW_LENGTH), _rowIdx)
                         : createValueArray<int>(dim4(ROW_LENGTH), 0))
        : createHostDataArray<int>(dim4(ROW_LENGTH), _rowIdx)),
    colIdx(_is_device ?
          (!_copy_device ? createDeviceDataArray<int>(dim4(COL_LENGTH), _colIdx)
                         : createValueArray<int>(dim4(COL_LENGTH), 0))
        : createHostDataArray<int>(dim4(COL_LENGTH), _colIdx))
{
#if __cplusplus > 199711L
    static_assert(offsetof(SparseArrayBase, info) == 0,
                  "SparseArrayBase::info must be the first member variable of SparseArrayBase.");
#endif
    if(_is_device && _copy_device) {
        writeDeviceDataArray<int>(rowIdx, _rowIdx, ROW_LENGTH * sizeof(int));
        writeDeviceDataArray<int>(colIdx, _colIdx, COL_LENGTH * sizeof(int));
    }
}

SparseArrayBase::SparseArrayBase(af::dim4 _dims,
                    const Array<int> &_rowIdx, const Array<int> &_colIdx,
                    const af::sparseStorage _storage, af_dtype _type,
                    bool _copy):
    info(getActiveDeviceId(), _dims, 0, calcStrides(_dims), _type, true),
    storage(_storage),
    rowIdx(_copy ? copyArray<int>(_rowIdx): _rowIdx),
    colIdx(_copy ? copyArray<int>(_colIdx): _colIdx)
{
#if __cplusplus > 199711L
    static_assert(offsetof(SparseArrayBase, info) == 0,
                  "SparseArrayBase::info must be the first member variable of SparseArrayBase.");
#endif
}

SparseArrayBase::~SparseArrayBase()
{
}

dim_t SparseArrayBase::getNNZ() const
{
    if(storage == AF_SPARSE_COO || storage == AF_SPARSE_CSC)
        return rowIdx.elements();
    else if(storage == AF_SPARSE_CSR)
        return colIdx.elements();

    // This is to ensure future storages are properly configured
    return 0;
}

#undef ROW_LENGTH
#undef COL_LENGTH

////////////////////////////////////////////////////////////////////////////
// Friend functions for Sparse Array Creation Implementations
////////////////////////////////////////////////////////////////////////////
template<typename T>
SparseArray<T> createEmptySparseArray(
        const af::dim4 &_dims, dim_t _nNZ, const af::sparseStorage _storage)
{
    return SparseArray<T>(_dims, _nNZ, _storage);
}

template<typename T>
SparseArray<T> createHostDataSparseArray(
        const af::dim4 &_dims, const dim_t nNZ,
        const T * const _values,
        const int * const _rowIdx, const int * const _colIdx,
        const af::sparseStorage _storage)
{
    return SparseArray<T>(_dims, nNZ, _values, _rowIdx, _colIdx, _storage, false);
}

template<typename T>
SparseArray<T> createDeviceDataSparseArray(
        const af::dim4 &_dims, const dim_t nNZ,
        const T * const _values,
        const int * const _rowIdx, const int * const _colIdx,
        const af::sparseStorage _storage)
{
    return SparseArray<T>(_dims, nNZ, _values, _rowIdx, _colIdx, _storage, false);
}

template<typename T>
SparseArray<T> createArrayDataSparseArray(
        const af::dim4 &_dims,
        const Array<T> &_values,
        const Array<int> &_rowIdx, const Array<int> &_colIdx,
        const af::sparseStorage _storage)
{
    return SparseArray<T>(_dims, _values, _rowIdx, _colIdx, _storage, false);
}

template<typename T>
SparseArray<T> *initSparseArray()
{
    return new SparseArray<T>(dim4(), 0,  (af::sparseStorage)0);
}

template<typename T>
void destroySparseArray(SparseArray<T> *sparse)
{
    delete sparse;
}

////////////////////////////////////////////////////////////////////////////
// Sparse Array Class Implementations
////////////////////////////////////////////////////////////////////////////
template<typename T>
SparseArray<T>::SparseArray(dim4 _dims, dim_t _nNZ, af::sparseStorage _storage):
    base(_dims, _nNZ, _storage, (af_dtype)dtype_traits<T>::af_type),
    values(createValueArray<T>(dim4(_nNZ), scalar<T>(0)))
{
#if __cplusplus > 199711L
        static_assert(std::is_standard_layout<SparseArray<T>>::value,
                      "SparseArray<T> must be a standard layout type");
        static_assert(offsetof(SparseArray<T>, base) == 0,
                      "SparseArray<T>::base must be the first member variable of SparseArray<T>");
#endif
}

template<typename T>
SparseArray<T>::SparseArray(af::dim4 _dims, dim_t _nNZ,
                     const T * const _values,
                     const int * const _rowIdx, const int * const _colIdx,
                     const af::sparseStorage _storage,
                     bool _is_device, bool _copy_device):
    base(_dims, _nNZ, _rowIdx, _colIdx, _storage, (af_dtype)dtype_traits<T>::af_type, _is_device, _copy_device),
    values(_is_device ?
          (!_copy_device ? createDeviceDataArray<T>(dim4(_nNZ), _values)
                         : createValueArray<T>(dim4(_nNZ), scalar<T>(0)))
        : createHostDataArray<T>(dim4(_nNZ), _values))
{
#if __cplusplus > 199711L
        static_assert(std::is_standard_layout<SparseArray<T>>::value,
                      "SparseArray<T> must be a standard layout type");
        static_assert(offsetof(SparseArray<T>, base) == 0,
                      "SparseArray<T>::base must be the first member variable of SparseArray<T>");
#endif
    if(_is_device && _copy_device) {
        writeDeviceDataArray<T>(values, _values, _nNZ * sizeof(T));
    }
}

template<typename T>
SparseArray<T>::SparseArray(af::dim4 _dims,
            const Array<T> &_values,
            const Array<int> &_rowIdx, const Array<int> &_colIdx,
            const af::sparseStorage _storage, bool _copy):
    base(_dims, _rowIdx, _colIdx, _storage, (af_dtype)dtype_traits<T>::af_type, _copy),
    values(_copy ? copyArray<T>(_values): _values)
{
#if __cplusplus > 199711L
        static_assert(std::is_standard_layout<SparseArray<T>>::value,
                      "SparseArray<T> must be a standard layout type");
        static_assert(offsetof(SparseArray<T>, base) == 0,
                      "SparseArray<T>::base must be the first member variable of SparseArray<T>");
#endif
}

template<typename T>
SparseArray<T>::~SparseArray()
{
}

#define INSTANTIATE(T)                                                                              \
    template SparseArray<T> createEmptySparseArray<T>(                                              \
            const af::dim4 &_dims, dim_t _nNZ, const af::sparseStorage _storage);                   \
    template SparseArray<T> createHostDataSparseArray<T>(                                           \
            const af::dim4 &_dims, const dim_t _nNZ,                                                \
            const T * const _values,                                                                \
            const int * const _rowIdx, const int * const _colIdx,                                   \
            const af::sparseStorage _storage);                                                      \
    template SparseArray<T> createDeviceDataSparseArray<T>(                                         \
            const af::dim4 &_dims, const dim_t _nNZ,                                                \
            const T * const _values,                                                                \
            const int * const _rowIdx, const int * const _colIdx,                                   \
            const af::sparseStorage _storage);                                                      \
    template SparseArray<T> createArrayDataSparseArray<T>(                                          \
            const af::dim4 &_dims,                                                                  \
            const Array<T> &_values,                                                                \
            const Array<int> &_rowIdx, const Array<int> &_colIdx,                                   \
            const af::sparseStorage _storage);                                                      \
    template SparseArray<T> *initSparseArray<T>();                                                  \
    template void destroySparseArray<T>(SparseArray<T> *sparse);                                    \
                                                                                                    \
    template SparseArray<T>::SparseArray(af::dim4 _dims, dim_t _nNZ, af::sparseStorage _storage);   \
    template SparseArray<T>::SparseArray(af::dim4 _dims, dim_t _nNZ,                                \
                        const T * const _values,                                                    \
                        const int * const _rowIdx, const int * const _colIdx,                       \
                        const af::sparseStorage _storage,                                           \
                        bool _is_device, bool _copy_device);                                        \
    template SparseArray<T>::SparseArray(af::dim4 _dims,                                            \
                        const Array<T> &_values,                                                    \
                        const Array<int> &_rowIdx, const Array<int> &_colIdx,                       \
                        const af::sparseStorage _storage, bool _copy);                              \
    template SparseArray<T>::~SparseArray();

// Instantiate only floating types
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(cfloat);
INSTANTIATE(cdouble);

#undef INSTANTIATE

} // namespace common
