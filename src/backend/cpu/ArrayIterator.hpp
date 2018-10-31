/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/dim4.hpp>

#include <cstddef>
#include <iterator>
#include <vector>

namespace cpu {

/// A node iterator that performs a breadth first traversal of the node tree
template<typename T>
class ArrayIterator : public std::iterator<std::input_iterator_tag, T> {
    T* ptr;
    af::dim4 dims;
    af::dim4 stride;
    int d0, d1, d2, d3;
  public:
    using pointer   = T*;
    using reference = T&;

    /// ArrayIterator Constructor
    ArrayIterator(detail::Param<T>& in, bool end)
        : ptr(in.get())
        , dims(in.dims())
        , stride(in.strides())
        , d0(dims[0]), d1(dims[1]), d2(dims[2]), d3(dims[3]) {
          if(end) {
              ptr += (dims[0] - 1) * stride[0] +
                    (dims[1] - 1) * stride[1] +
                    (dims[2] - 1) * stride[2] +
                    (dims[3] - 1) * stride[3] + 1;
          }
    }

    ArrayIterator(detail::CParam<T>& in, bool end)
        : ptr(in.get())
        , dims(in.dims())
        , stride(in.strides())
        , d0(dims[0]), d1(dims[1]), d2(dims[2]), d3(dims[3]) {
        if(end) {
            ptr += (dims[0] - 1) * stride[0] +
                   (dims[1] - 1) * stride[1] +
                   (dims[2] - 1) * stride[2] +
                   (dims[3] - 1) * stride[3] + 1;
        }
    }

    /// The equality operator
    ///
    /// \param[in] other the rhs of the node
  bool operator==(const ArrayIterator& other) const noexcept {
      return other.ptr == ptr;

  }
  bool operator!=(const ArrayIterator& other) const noexcept {
      return other.ptr != ptr;
  }

    /// Advances the iterator by one node in the tree
    ArrayIterator& operator++() noexcept {
        d0--;
        ptr += stride[0];
        if(dims[0]) return *this;
        d0 = dims[0];
        d1--;
        ptr += stride[1];
        if(dims[1]) return *this;
        d1 = dims[1];
        d2--;
        ptr += stride[2];
        if(dims[2]) return *this;
        d2 = dims[2];
        d3--;
        ptr += stride[3];
        if(dims[3]) return *this;
        return *this;
    }

    /// @copydoc operator++()
    ArrayIterator operator++(int) noexcept {
        ArrayIterator before(*this);
        operator++();
        return before;
    }

    /// Advances the iterator by count nodes
    ArrayIterator& operator+=(std::size_t count) noexcept {
        while (count-- > 0) {
            operator++();
        }
        return *this;
    }

    reference operator*() const noexcept {
        return *ptr;
    }

    pointer operator->() const noexcept {
        return ptr;
    }

    /// Creates a sentinel iterator. This is equivalent to the end iterator
    ArrayIterator() = default;
    ArrayIterator(const ArrayIterator& other) = default;
    ArrayIterator(ArrayIterator&& other) noexcept = default;
    ~ArrayIterator() noexcept = default;
    ArrayIterator& operator=(const ArrayIterator& other) noexcept = default;
    ArrayIterator& operator=(ArrayIterator&& other) noexcept = default;
};

  template<typename T>
  ArrayIterator<T> begin(Param<T>& param) {
      return ArrayIterator<T>(param, false);
  }

  template<typename T>
  ArrayIterator<T> end(Param<T>& param) {
      return ArrayIterator<T>(param, true);
  }

  template<typename T>
  ArrayIterator<const T> begin(CParam<T>& param) {
      return ArrayIterator<const T>(param, false);
  }

  template<typename T>
  ArrayIterator<const T> end(CParam<T>& param) {
      return ArrayIterator<const T>(param, true);
  }

}
