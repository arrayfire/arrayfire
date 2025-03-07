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
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/sparse_helpers.hpp>

#include <cstddef>
#include <vector>

namespace arrayfire {
namespace common {

template<typename T>
class SparseArray;

/// SparseArray Array Info class
///
/// This class is the base class to all SparseArray objects. The purpose of this
/// class was to have a way to retrieve basic information of an Array object
/// without specifying what type the object is at compile time.
///
/// NOTE: This is not a template class to allow the frontend to determine the
/// af_array type at runtime
class SparseArrayBase {
   private:
    ArrayInfo
        info;  ///< NOTE: This must be the first element of SparseArray<T>.
    af::storage stype;          ///< Storage format: CSR, CSC, COO
    detail::Array<int> rowIdx;  ///< Linear array containing row indices
    detail::Array<int> colIdx;  ///< Linear array containing col indices

   public:
    SparseArrayBase(SparseArrayBase &&other) noexcept = default;
    SparseArrayBase(const af::dim4 &_dims, dim_t _nNZ, af::storage _storage,
                    af_dtype _type);

    SparseArrayBase(const af::dim4 &_dims, dim_t _nNZ, int *const _rowIdx,
                    int *const _colIdx, const af::storage _storage,
                    af_dtype _type, bool _is_device = false,
                    bool _copy_device = false);

    SparseArrayBase(const af::dim4 &_dims, const detail::Array<int> &_rowIdx,
                    const detail::Array<int> &_colIdx,
                    const af::storage _storage, af_dtype _type,
                    bool _copy = false);

    SparseArrayBase &operator=(SparseArrayBase other) noexcept {
        std::swap(*this, other);
        return *this;
    }

    /// A copy constructor for SparseArray
    ///
    /// This constructor copies the \p in SparseArray and creates a new object
    /// from it. It can also perform a deep copy if the second argument is true.
    ///
    /// \param[in] in         The array that will be copied
    /// \param[in] deep_copy  If true a deep copy is performed
    SparseArrayBase(const SparseArrayBase &base, bool deep_copy = false);

    ~SparseArrayBase();

    ////////////////////////////////////////////////////////////////////////////
    // Functions that call ArrayInfo object's functions
    ////////////////////////////////////////////////////////////////////////////
#define INSTANTIATE_INFO(return_type, func) \
    return_type func() const { return info.func(); }

    INSTANTIATE_INFO(const af_dtype &, getType)
    INSTANTIATE_INFO(size_t, elements)
    INSTANTIATE_INFO(size_t, ndims)
    INSTANTIATE_INFO(const af::dim4 &, dims)
    INSTANTIATE_INFO(size_t, total)
    INSTANTIATE_INFO(int, getDevId)
    INSTANTIATE_INFO(af_backend, getBackendId)
    INSTANTIATE_INFO(bool, isEmpty)
    INSTANTIATE_INFO(bool, isScalar)
    INSTANTIATE_INFO(bool, isRow)
    INSTANTIATE_INFO(bool, isColumn)
    INSTANTIATE_INFO(bool, isVector)
    INSTANTIATE_INFO(bool, isComplex)
    INSTANTIATE_INFO(bool, isReal)
    INSTANTIATE_INFO(bool, isDouble)
    INSTANTIATE_INFO(bool, isSingle)
    INSTANTIATE_INFO(bool, isHalf)
    INSTANTIATE_INFO(bool, isRealFloating)
    INSTANTIATE_INFO(bool, isFloating)
    INSTANTIATE_INFO(bool, isInteger)
    INSTANTIATE_INFO(bool, isBool)
    INSTANTIATE_INFO(bool, isLinear)
    INSTANTIATE_INFO(bool, isSparse)

#undef INSTANTIATE_INFO

    // setId of info, values, rowIdx, colIdx
    void setId(int id) {
        info.setId(id);
        rowIdx.setId(id);
        colIdx.setId(id);
    }

    /// Returns the row indices for the corresponding values in the SparseArray
    detail::Array<int> &getRowIdx() { return rowIdx; }
    const detail::Array<int> &getRowIdx() const { return rowIdx; }

    /// Returns the column indices for the corresponding values in the
    /// SparseArray
    detail::Array<int> &getColIdx() { return colIdx; }
    const detail::Array<int> &getColIdx() const { return colIdx; }

    /// Returns the number of non-zero elements in the array.
    dim_t getNNZ() const;

    /// Returns the storage format of the SparseArray
    af::storage getStorage() const { return stype; }
};
static_assert(std::is_standard_layout<SparseArrayBase>::value,
              "SparseArrayBase must be a standard layout type");

////////////////////////////////////////////////////////////////////////////
// Sparse Array Class
////////////////////////////////////////////////////////////////////////////
template<typename T>
class SparseArray {
   private:
    SparseArrayBase
        base;  ///< This must be the first element of SparseArray<T>.
    detail::Array<T> values;  ///< Linear array containing actual values

    SparseArray(const af::dim4 &_dims, dim_t _nNZ, af::storage _storage);

    explicit SparseArray(const af::dim4 &_dims, dim_t _nNZ, T *const _values,
                         int *const _rowIdx, int *const _colIdx,
                         const af::storage _storage, bool _is_device = false,
                         bool _copy_device = false);

    SparseArray(const af::dim4 &_dims, const detail::Array<T> &_values,
                const detail::Array<int> &_rowIdx,
                const detail::Array<int> &_colIdx, const af::storage _storage,
                bool _copy = false);

    /// A copy constructor for SparseArray
    ///
    /// This constructor copies the \p in SparseArray and creates a new object
    /// from it. It can also perform a deep copy if the second argument is true.
    ///
    /// \param[in] other      The array that will be copied
    /// \param[in] deep_copy  If true a deep copy is performed
    SparseArray(const SparseArray<T> &other, bool deep_copy);

   public:
    SparseArray(const SparseArray<T> &other)     = default;
    SparseArray(SparseArray<T> &&other) noexcept = default;

    ~SparseArray() noexcept = default;

    SparseArray<T> &operator=(SparseArray<T> other) noexcept {
        std::swap(*this, other);
        return *this;
    }
// Functions that call ArrayInfo object's functions
#define INSTANTIATE_INFO(return_type, func) \
    return_type func() const { return base.func(); }

    INSTANTIATE_INFO(const af_dtype &, getType)
    INSTANTIATE_INFO(size_t, elements)
    INSTANTIATE_INFO(size_t, ndims)
    INSTANTIATE_INFO(const af::dim4 &, dims)
    INSTANTIATE_INFO(size_t, total)
    INSTANTIATE_INFO(int, getDevId)
    INSTANTIATE_INFO(af_backend, getBackendId)
    INSTANTIATE_INFO(bool, isEmpty)
    INSTANTIATE_INFO(bool, isScalar)
    INSTANTIATE_INFO(bool, isRow)
    INSTANTIATE_INFO(bool, isColumn)
    INSTANTIATE_INFO(bool, isVector)
    INSTANTIATE_INFO(bool, isComplex)
    INSTANTIATE_INFO(bool, isReal)
    INSTANTIATE_INFO(bool, isDouble)
    INSTANTIATE_INFO(bool, isSingle)
    INSTANTIATE_INFO(bool, isHalf)
    INSTANTIATE_INFO(bool, isRealFloating)
    INSTANTIATE_INFO(bool, isFloating)
    INSTANTIATE_INFO(bool, isInteger)
    INSTANTIATE_INFO(bool, isBool)
    INSTANTIATE_INFO(bool, isLinear)
    INSTANTIATE_INFO(bool, isSparse)

    // Function from Base but not in ArrayInfo
    INSTANTIATE_INFO(dim_t, getNNZ)
    INSTANTIATE_INFO(af::storage, getStorage)

    detail::Array<int> &getRowIdx() { return base.getRowIdx(); }
    detail::Array<int> &getColIdx() { return base.getColIdx(); }
    const detail::Array<int> &getRowIdx() const { return base.getRowIdx(); }
    const detail::Array<int> &getColIdx() const { return base.getColIdx(); }

#undef INSTANTIATE_INFO

    void setId(int id) {
        base.setId(id);
        values.setId(id);
    }

    // Return the values array
    detail::Array<T> &getValues() { return values; }
    const detail::Array<T> &getValues() const { return values; }

    void eval() const {
        getValues().eval();
        getRowIdx().eval();
        getColIdx().eval();
    }

    // Friend functions for Sparse Array Creation
    friend SparseArray<T> createEmptySparseArray<T>(const af::dim4 &_dims,
                                                    dim_t _nNZ,
                                                    const af::storage _storage);

    friend SparseArray<T> createHostDataSparseArray<T>(
        const af::dim4 &_dims, const dim_t nNZ, const T *const _values,
        const int *const _rowIdx, const int *const _colIdx,
        const af::storage _storage);

    friend SparseArray<T> createDeviceDataSparseArray<T>(
        const af::dim4 &_dims, const dim_t nNZ, T *const _values,
        int *const _rowIdx, int *const _colIdx, const af::storage _storage,
        const bool _copy);

    friend SparseArray<T> createArrayDataSparseArray<T>(
        const af::dim4 &_dims, const detail::Array<T> &_values,
        const detail::Array<int> &_rowIdx, const detail::Array<int> &_colIdx,
        const af::storage _storage, const bool _copy);

    friend SparseArray<T> *initSparseArray<T>();

    friend SparseArray<T> copySparseArray<T>(const SparseArray<T> &input);

    friend void destroySparseArray<T>(SparseArray<T> *sparse);
};

/// Checks if the Array object can be migrated to the current device and if not,
/// an error is thrown
///
/// \param[in] arr The Array that will be checked.
template<typename T>
void checkAndMigrate(const SparseArray<T> &arr);

}  // namespace common
}  // namespace arrayfire
