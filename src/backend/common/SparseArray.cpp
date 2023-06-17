/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/SparseArray.hpp>
#include <copy.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <af/traits.hpp>

using af::dim4;
using af::dtype_traits;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::copyArray;
using detail::createDeviceDataArray;
using detail::createHostDataArray;
using detail::createValueArray;
using detail::getActiveDeviceId;
using detail::scalar;
using detail::writeDeviceDataArray;

namespace arrayfire {
namespace common {
////////////////////////////////////////////////////////////////////////////
// Sparse Array Base Implementations
////////////////////////////////////////////////////////////////////////////

// ROW_LENGTH and column length expect standard variable names of
// SparseArrayBase::stype
// _nNZ  -> Constructor Argument
// _dims -> Constructor Argument
#define ROW_LENGTH                                               \
    ((stype == AF_STORAGE_COO || stype == AF_STORAGE_CSC) ? _nNZ \
                                                          : (_dims[0] + 1))
#define COL_LENGTH                                               \
    ((stype == AF_STORAGE_COO || stype == AF_STORAGE_CSR) ? _nNZ \
                                                          : (_dims[1] + 1))

SparseArrayBase::SparseArrayBase(const af::dim4 &_dims, dim_t _nNZ,
                                 af::storage _storage, af_dtype _type)
    : info(getActiveDeviceId(), _dims, 0, calcStrides(_dims), _type, true)
    , stype(_storage)
    , rowIdx(createValueArray<int>(dim4(ROW_LENGTH), 0))
    , colIdx(createValueArray<int>(dim4(COL_LENGTH), 0)) {
    static_assert(offsetof(SparseArrayBase, info) == 0,
                  "SparseArrayBase::info must be the first member variable of "
                  "SparseArrayBase.");
    static_assert(std::is_nothrow_move_assignable<SparseArrayBase>::value,
                  "SparseArrayBase is not move assignable");
    static_assert(std::is_nothrow_move_constructible<SparseArrayBase>::value,
                  "SparseArrayBase is not move constructible");
}

SparseArrayBase::SparseArrayBase(const af::dim4 &_dims, dim_t _nNZ,
                                 int *const _rowIdx, int *const _colIdx,
                                 const af::storage _storage, af_dtype _type,
                                 bool _is_device, bool _copy_device)
    : info(getActiveDeviceId(), _dims, 0, calcStrides(_dims), _type, true)
    , stype(_storage)
    , rowIdx(_is_device
                 ? (!_copy_device
                        ? createDeviceDataArray<int>(dim4(ROW_LENGTH), _rowIdx)
                        : createValueArray<int>(dim4(ROW_LENGTH), 0))
                 : createHostDataArray<int>(dim4(ROW_LENGTH), _rowIdx))
    , colIdx(_is_device
                 ? (!_copy_device
                        ? createDeviceDataArray<int>(dim4(COL_LENGTH), _colIdx)
                        : createValueArray<int>(dim4(COL_LENGTH), 0))
                 : createHostDataArray<int>(dim4(COL_LENGTH), _colIdx)) {
    static_assert(offsetof(SparseArrayBase, info) == 0,
                  "SparseArrayBase::info must be the first member variable of "
                  "SparseArrayBase.");
    if (_is_device && _copy_device) {
        writeDeviceDataArray<int>(rowIdx, _rowIdx, ROW_LENGTH * sizeof(int));
        writeDeviceDataArray<int>(colIdx, _colIdx, COL_LENGTH * sizeof(int));
    }
}

SparseArrayBase::SparseArrayBase(const af::dim4 &_dims,
                                 const Array<int> &_rowIdx,
                                 const Array<int> &_colIdx,
                                 const af::storage _storage, af_dtype _type,
                                 bool _copy)
    : info(getActiveDeviceId(), _dims, 0, calcStrides(_dims), _type, true)
    , stype(_storage)
    , rowIdx(_copy ? copyArray<int>(_rowIdx) : _rowIdx)
    , colIdx(_copy ? copyArray<int>(_colIdx) : _colIdx) {
    static_assert(offsetof(SparseArrayBase, info) == 0,
                  "SparseArrayBase::info must be the first member variable of "
                  "SparseArrayBase.");
}

SparseArrayBase::SparseArrayBase(const SparseArrayBase &base, bool copy)
    : info(base.info)
    , stype(base.stype)
    , rowIdx(copy ? copyArray<int>(base.rowIdx) : base.rowIdx)
    , colIdx(copy ? copyArray<int>(base.colIdx) : base.colIdx) {}

SparseArrayBase::~SparseArrayBase() = default;

dim_t SparseArrayBase::getNNZ() const {
    if (stype == AF_STORAGE_COO || stype == AF_STORAGE_CSC) {
        return rowIdx.elements();
    }
    if (stype == AF_STORAGE_CSR) { return colIdx.elements(); }

    // This is to ensure future storages are properly configured
    return 0;
}

#undef ROW_LENGTH
#undef COL_LENGTH

////////////////////////////////////////////////////////////////////////////
// Friend functions for Sparse Array Creation Implementations
////////////////////////////////////////////////////////////////////////////
template<typename T>
SparseArray<T> createEmptySparseArray(const af::dim4 &_dims, dim_t _nNZ,
                                      const af::storage _storage) {
    return SparseArray<T>(_dims, _nNZ, _storage);
}

template<typename T>
SparseArray<T> createHostDataSparseArray(const af::dim4 &_dims, const dim_t nNZ,
                                         const T *const _values,
                                         const int *const _rowIdx,
                                         const int *const _colIdx,
                                         const af::storage _storage) {
    return SparseArray<T>(_dims, nNZ, const_cast<T *>(_values),
                          const_cast<int *>(_rowIdx),
                          const_cast<int *>(_colIdx), _storage, false);
}

template<typename T>
SparseArray<T> createDeviceDataSparseArray(
    const af::dim4 &_dims, const dim_t nNZ, T *const _values,
    int *const _rowIdx,  // NOLINT(readability-non-const-parameter)
    int *const _colIdx,  // NOLINT(readability-non-const-parameter)
    const af::storage _storage, const bool _copy) {
    return SparseArray<T>(_dims, nNZ, _values, _rowIdx, _colIdx, _storage, true,
                          _copy);
}

template<typename T>
SparseArray<T> createArrayDataSparseArray(
    const af::dim4 &_dims, const Array<T> &_values, const Array<int> &_rowIdx,
    const Array<int> &_colIdx, const af::storage _storage, const bool _copy) {
    return SparseArray<T>(_dims, _values, _rowIdx, _colIdx, _storage, _copy);
}

template<typename T>
SparseArray<T> copySparseArray(const SparseArray<T> &other) {
    return SparseArray<T>(other, true);
}

template<typename T>
SparseArray<T> *initSparseArray() {
    return new SparseArray<T>(dim4(), 0, (af::storage)0);
}

template<typename T>
void destroySparseArray(SparseArray<T> *sparse) {
    delete sparse;
}

template<typename T>
void checkAndMigrate(const SparseArray<T> &arr) {
    checkAndMigrate(const_cast<Array<int> &>(arr.getColIdx()));
    checkAndMigrate(const_cast<Array<int> &>(arr.getRowIdx()));
    checkAndMigrate(const_cast<Array<T> &>(arr.getValues()));
}

////////////////////////////////////////////////////////////////////////////
// Sparse Array Class Implementations
////////////////////////////////////////////////////////////////////////////
template<typename T>
SparseArray<T>::SparseArray(const dim4 &_dims, dim_t _nNZ, af::storage _storage)
    : base(_dims, _nNZ, _storage,
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , values(createValueArray<T>(dim4(_nNZ), scalar<T>(0))) {
    static_assert(std::is_standard_layout<SparseArray<T>>::value,
                  "SparseArray<T> must be a standard layout type");
    static_assert(std::is_nothrow_move_assignable<SparseArray<T>>::value,
                  "SparseArray<T> is not move assignable");
    static_assert(std::is_nothrow_move_constructible<SparseArray<T>>::value,
                  "SparseArray<T> is not move constructible");
    static_assert(offsetof(SparseArray<T>, base) == 0,
                  "SparseArray<T>::base must be the first member variable of "
                  "SparseArray<T>");
}

template<typename T>
SparseArray<T>::SparseArray(const af::dim4 &_dims, dim_t _nNZ, T *const _values,
                            int *const _rowIdx, int *const _colIdx,
                            const af::storage _storage, bool _is_device,
                            bool _copy_device)
    : base(_dims, _nNZ, _rowIdx, _colIdx, _storage,
           static_cast<af_dtype>(dtype_traits<T>::af_type), _is_device,
           _copy_device)
    , values(_is_device ? (!_copy_device
                               ? createDeviceDataArray<T>(dim4(_nNZ), _values)
                               : createValueArray<T>(dim4(_nNZ), scalar<T>(0)))
                        : createHostDataArray<T>(dim4(_nNZ), _values)) {
    if (_is_device && _copy_device) {
        writeDeviceDataArray<T>(values, _values, _nNZ * sizeof(T));
    }
}

template<typename T>
SparseArray<T>::SparseArray(const af::dim4 &_dims, const Array<T> &_values,
                            const Array<int> &_rowIdx,
                            const Array<int> &_colIdx,
                            const af::storage _storage, bool _copy)
    : base(_dims, _rowIdx, _colIdx, _storage,
           static_cast<af_dtype>(dtype_traits<T>::af_type), _copy)
    , values(_copy ? copyArray<T>(_values) : _values) {}

template<typename T>
SparseArray<T>::SparseArray(const SparseArray<T> &other, bool copy)
    : base(other.base, copy)
    , values(copy ? copyArray<T>(other.values) : other.values) {}

#define INSTANTIATE(T)                                                       \
    template SparseArray<T> createEmptySparseArray<T>(                       \
        const af::dim4 &_dims, dim_t _nNZ, const af::storage _storage);      \
    template SparseArray<T> createHostDataSparseArray<T>(                    \
        const af::dim4 &_dims, const dim_t _nNZ, const T *const _values,     \
        const int *const _rowIdx, const int *const _colIdx,                  \
        const af::storage _storage);                                         \
    template SparseArray<T> createDeviceDataSparseArray<T>(                  \
        const af::dim4 &_dims, const dim_t _nNZ,                             \
        T *const _values, /*  NOLINT */                                      \
        int *const _rowIdx, int *const _colIdx, const af::storage _storage,  \
        const bool _copy);                                                   \
    template SparseArray<T> createArrayDataSparseArray<T>(                   \
        const af::dim4 &_dims, const Array<T> &_values,                      \
        const Array<int> &_rowIdx, const Array<int> &_colIdx,                \
        const af::storage _storage, const bool _copy);                       \
    template SparseArray<T> *initSparseArray<T>();                           \
    template SparseArray<T> copySparseArray<T>(const SparseArray<T> &other); \
    template void destroySparseArray<T>(SparseArray<T> * sparse);            \
                                                                             \
    template SparseArray<T>::SparseArray(const af::dim4 &_dims, dim_t _nNZ,  \
                                         af::storage _storage);              \
    template SparseArray<T>::SparseArray(                                    \
        const af::dim4 &_dims, dim_t _nNZ, T *const _values, /* NOLINT */    \
        int *const _rowIdx, int *const _colIdx, const af::storage _storage,  \
        bool _is_device, bool _copy_device);                                 \
    template SparseArray<T>::SparseArray(                                    \
        const af::dim4 &_dims, const Array<T> &_values,                      \
        const Array<int> &_rowIdx, const Array<int> &_colIdx,                \
        const af::storage _storage, bool _copy);                             \
    template void checkAndMigrate(const SparseArray<T> &arr)

// Instantiate only floating types
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(cfloat);
INSTANTIATE(cdouble);

#undef INSTANTIATE

}  // namespace common
}  // namespace arrayfire
