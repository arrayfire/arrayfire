/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <af/dim4.hpp>

#include <array>
#include <cstddef>
#include <iterator>
#include <vector>

namespace cpu {

/// Calculates the iterator offsets.
///
/// These are different from the original offsets because they define
/// the stride from the end of the last element in the previous dimension
/// to the first element on the next dimension.
static dim4 calcIteratorStrides(const dim4& dims, const dim4& stride) noexcept {
    return dim4(stride[0], stride[1] - (stride[0] * dims[0]),
                stride[2] - (stride[1] * dims[1]),
                stride[3] - (stride[2] * dims[2]));
}

/// A Param iterator that iterates through a Param object
template<typename T>
class ParamIterator {
   public:
    using difference_type   = ptrdiff_t;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;
    using iterator_category = std::forward_iterator_tag;

    /// Creates a sentinel iterator. This is equivalent to the end iterator
    ParamIterator() noexcept
        : ptr(nullptr)
        , dims(1)
        , stride(1)
        , dim_index{dims[0], dims[1], dims[2], dims[3]} {}

    /// ParamIterator Constructor
    ParamIterator(cpu::Param<T>& in) noexcept
        : ptr(in.get())
        , dims(in.dims())
        , stride(calcIteratorStrides(dims, in.strides()))
        , dim_index{in.dims()[0], in.dims()[1], in.dims()[2], in.dims()[3]} {}

    ParamIterator(cpu::CParam<typename std::remove_const<T>::type>& in) noexcept
        : ptr(in.get())
        , dims(in.dims())
        , stride(calcIteratorStrides(dims, in.strides()))
        , dim_index{in.dims()[0], in.dims()[1], in.dims()[2], in.dims()[3]} {}

    /// The equality operator
    bool operator==(const ParamIterator& other) const noexcept {
        return ptr == other.ptr;
    }

    /// The inequality operator
    bool operator!=(const ParamIterator& other) const noexcept {
        return ptr != other.ptr;
    }

    /// Advances the iterator, pre increment operator
    ParamIterator& operator++() noexcept {
        for (int i = 0; i < AF_MAX_DIMS; i++) {
            dim_index[i]--;
            ptr += stride[i];
            if (dim_index[i]) { return *this; }
            dim_index[i] = dims[i];
        }
        ptr = nullptr;
        return *this;
    }

    /// Advances the iterator by count elements
    ParamIterator& operator+=(std::size_t count) noexcept {
        while (count-- > 0) { operator++(); }
        return *this;
    }

    const reference operator*() const noexcept { return *ptr; }

    const pointer operator->() const noexcept { return ptr; }

    ParamIterator(const ParamIterator<T>& other) = default;
    ParamIterator(ParamIterator<T>&& other)      = default;
    ~ParamIterator() noexcept                    = default;
    ParamIterator<T>& operator=(const ParamIterator<T>& other) noexcept =
        default;
    ParamIterator<T>& operator=(ParamIterator<T>&& other) noexcept = default;

   private:
    T* ptr;

    // The dimension of the array
    const af::dim4 dims;

    // The iterator's stride
    const af::dim4 stride;

    // NOTE: This is not really the true coordinate of the iteration. It's
    // values will go down as you move through the array.
    std::array<dim_t, AF_MAX_DIMS> dim_index;
};

template<typename T>
ParamIterator<T> begin(Param<T>& param) {
    return ParamIterator<T>(param);
}

template<typename T>
ParamIterator<T> end(Param<T>& param) {
    return ParamIterator<T>();
}

template<typename T>
ParamIterator<const T> begin(CParam<T>& param) {
    return ParamIterator<const T>(param);
}

template<typename T>
ParamIterator<const T> end(CParam<T>& param) {
    return ParamIterator<const T>();
}

/// Neighborhood iterator for Param data
template<typename T>
class NeighborhoodIterator {
   public:
    using difference_type   = ptrdiff_t;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;
    using iterator_category = std::forward_iterator_tag;

    using Self = NeighborhoodIterator;

    /// Creates a sentinel iterator. This is equivalent to the end iterator
    NeighborhoodIterator() noexcept
        : nhoodRadius(0, 0, 0, 0)
        , origDims(1)
        , origStrides(1)
        , iterDims(1)
        , iterStrides(1)
        , origPtr(nullptr)
        , ptr(origPtr)
        , nhoodIndex(0) {
        calcOffsets();
    }

    /// NeighborhoodIterator Constructor
    NeighborhoodIterator(cpu::Param<T>& in, const af::dim4 _radius) noexcept
        : nhoodRadius(_radius)
        , origDims(nhoodSize(nhoodRadius))
        , origStrides(in.strides())
        , iterDims(origDims)
        , iterStrides(calcIteratorStrides(origDims, in.strides()))
        , origPtr(in.get())
        , ptr(origPtr)
        , nhoodIndex(0) {
        calcOffsets();
    }

    /// NeighborhoodIterator Constructor
    NeighborhoodIterator(cpu::CParam<typename std::remove_const<T>::type>& in,
                         const af::dim4 _radius) noexcept
        : nhoodRadius(_radius)
        , origDims(nhoodSize(nhoodRadius))
        , origStrides(in.strides())
        , iterDims(origDims)
        , iterStrides(calcIteratorStrides(origDims, in.strides()))
        , origPtr(const_cast<T*>(in.get()))
        , ptr(origPtr)
        , nhoodIndex(0) {
        calcOffsets();
    }

    /// The equality operator
    bool operator==(const Self& other) const noexcept {
        return ptr == other.ptr;
    }

    /// The inequality operator
    bool operator!=(const Self& other) const noexcept {
        return ptr != other.ptr;
    }

    /// Set neighborhood center
    ///
    /// This method automatically resets iterator to starting point
    /// of the neighborhood around the set center point
    void setCenter(const af::dim4 center) noexcept {
        ptr = origPtr;
        for (dim_t d = 0; d < AF_MAX_DIMS; ++d) {
            ptr += ((center[d] - nhoodRadius[d]) * origStrides[d]);
        }
        nhoodIndex = 0;
    }

    /// Advances the iterator, pre increment operator
    Self& operator++() noexcept {
        nhoodIndex++;
        for (dim_t i = 0; i < AF_MAX_DIMS; i++) {
            iterDims[i]--;
            ptr += iterStrides[i];
            if (iterDims[i]) { return *this; }
            iterDims[i] = origDims[i];
        }
        ptr = nullptr;
        return *this;
    }

    /// @copydoc operator++()
    Self operator++(int) noexcept {
        Self before(*this);
        operator++();
        return before;
    }

    reference operator*() const noexcept { return *ptr; }
    pointer operator->() const noexcept { return ptr; }

    /// Gets offsets of current position from center
    const af::dim4 offset() const noexcept {
        if (ptr) {
            // Branch predictor almost always is a hit since,
            // NeighborhoodIterator::offset is called only when iterator is
            // valid i.e. it is not equal to END iterator
            return offsets[nhoodIndex];
        } else {
            return af::dim4(0, 0, 0, 0);
        }
    }

    NeighborhoodIterator(const NeighborhoodIterator<T>& other) = default;
    NeighborhoodIterator(NeighborhoodIterator<T>&& other) = default;
    ~NeighborhoodIterator() noexcept = default;
    NeighborhoodIterator<T>& operator=(const Self& other) = default;
    NeighborhoodIterator<T>& operator=(Self&& other) = default;

   private:
    const af::dim4 nhoodRadius;
    const af::dim4 origDims;
    const af::dim4 origStrides;
    af::dim4 iterDims;
    af::dim4 iterStrides;
    pointer origPtr;
    pointer ptr;
    dim_t nhoodIndex;
    std::vector<af::dim4> offsets;

    af::dim4 nhoodSize(const af::dim4& radius) const noexcept {
        return af::dim4(2 * radius[0] + 1, 2 * radius[1] + 1, 2 * radius[2] + 1,
                        2 * radius[3] + 1);
    }

    void calcOffsets() noexcept {
        auto linear2Coords = [this](const dim_t index) -> af::dim4 {
            af::dim4 coords(0, 0, 0, 0);
            for (dim_t i = 0, idx = index; i < AF_MAX_DIMS;
                 ++i, idx /= origDims[i]) {
                coords[i] = idx % origDims[i];
            }
            return coords;
        };

        offsets.clear();
        size_t nElems = (2 * nhoodRadius[0] + 1) * (2 * nhoodRadius[1] + 1) *
                        (2 * nhoodRadius[2] + 1) * (2 * nhoodRadius[3] + 1);
        offsets.reserve(nElems);
        for (size_t i = 0; i < nElems; ++i) {
            auto coords = linear2Coords(i);
            offsets.emplace_back(
                coords[0] - nhoodRadius[0], coords[1] - nhoodRadius[1],
                coords[2] - nhoodRadius[2], coords[3] - nhoodRadius[3]);
        }
    }
};

}  // namespace cpu
