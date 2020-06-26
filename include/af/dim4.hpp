/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifdef __cplusplus

#include <ostream>
#include <istream>
#include <vector>
#include <af/defines.h>
#include <af/seq.h>


namespace af
{
/// \brief Generic object that represents size and shape
/// \ingroup arrayfire_class
class AFAPI dim4
{
public:
    dim_t dims[4];
    /// Default constructor. Creates an invalid dim4 object
    dim4();

    /// Creates an new dim4 given a set of dimension
    dim4(   dim_t first,
            dim_t second = 1,
            dim_t third = 1,
            dim_t fourth = 1);

    /// Copy constructor
    ///
    /// \param[in] other The dim4 that will be copied
    dim4(const dim4& other);

#if AF_API_VERSION >= 38
#if AF_COMPILER_CXX_RVALUE_REFERENCES
    /// Default move constructor
    ///
    /// \param[in] other The dim4 that will be moved
    dim4(dim4 &&other) AF_NOEXCEPT = default;

    /// Default move assignment operator
    ///
    /// \param[in] other The dim4 that will be moved
    dim4 &operator=(dim4 other) AF_NOEXCEPT;
#endif
#endif

    /// Constructs a dim4 object from a C array of dim_t objects
    ///
    /// Creates a new dim4 from a C array. If the C array is less than 4, all
    /// values past \p ndims will be assigned the value 1.
    ///
    /// \param[in] ndims The number of elements in the C array. Must be less
    ///                  than 4
    /// \param[in] dims  The values to assign to each element of dim4
    dim4(const unsigned ndims, const dim_t *const dims);

    /// Returns the number of elements represented by this dim4
    dim_t elements();

    /// Returns the number of elements represented by this dim4
    dim_t elements() const;

    /// Returns the number of axis whose values are greater than one
    dim_t ndims();

    /// Returns the number of axis whose values are greater than one
    dim_t ndims() const;

    /// Returns true if the two dim4 represent the same shape
    bool operator==(const dim4 &other) const;

    /// Returns true if two dim4s store different values
    bool operator!=(const dim4 &other) const;

    /// Element-wise multiplication of the dim4 objects
    dim4 &operator*=(const dim4 &other);

    /// Element-wise addition of the dim4 objects
    dim4 &operator+=(const dim4 &other);

    /// Element-wise subtraction of the dim4 objects
    dim4 &operator-=(const dim4 &other);

    /// Returns the reference to the element at a give index. (Must be less than
    /// 4)
    dim_t &operator[](const unsigned dim);

    /// Returns the reference to the element at a give index. (Must be less than
    /// 4)
    const dim_t &operator[](const unsigned dim) const;

    /// Returns the underlying pointer to the dim4 object
    dim_t *get() { return dims; }

    /// Returns the underlying pointer to the dim4 object
    const dim_t *get() const { return dims; }
};

/// Performs an element-wise addition of two dim4 objects
AFAPI dim4 operator+(const dim4& first, const dim4& second);

/// Performs an element-wise subtraction of two dim4 objects
AFAPI dim4 operator-(const dim4& first, const dim4& second);

/// Performs an element-wise multiplication of two dim4 objects
AFAPI dim4 operator*(const dim4& first, const dim4& second);

/// Prints the elements of the dim4 array separated by spaces
///
/// \param[inout] ostr An ostream object
/// \param[in] dims The dim4 object to be printed
/// \returns the reference to the \p ostr after the dim4 string as been streamed in
static inline
std::ostream&
operator<<(std::ostream& ostr, const dim4& dims)
{
    ostr << dims[0] << " "
         << dims[1] << " "
         << dims[2] << " "
         << dims[3];
    return ostr;
}

/// Reads 4 dim_t values from an input stream and stores the results in a dim4
///
/// \param[inout] istr An istream object
/// \param[in] dims The dim4 object that will store the values
/// \return The \p istr object after 4 dim_t values have been read from the input
static inline
std::istream&
operator>>(std::istream& istr, dim4& dims)
{
    istr >> dims[0]
         >> dims[1]
         >> dims[2]
         >> dims[3];
    return istr;
}

/// Returns true if the af_seq object represents the entire range of an axis
AFAPI bool isSpan(const af_seq &seq);

/// Returns the number of elements that the af_seq object represents
AFAPI size_t seqElements(const af_seq &seq);

/// Returns the number of elements that will be represented by seq if applied on an array
AFAPI dim_t calcDim(const af_seq &seq, const dim_t &parentDim);
}

#endif
