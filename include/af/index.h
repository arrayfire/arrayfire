/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <af/seq.h>

///
/// \brief Struct used while indexing af_array
///
/// This struct represents objects which can be used to index into an af_array
/// Object. It contains a union object which can be an \ref af_seq or an
/// \ref af_array. Indexing with an int can be represented using a \ref af_seq
/// object with the same \ref af_seq::begin and \ref af_seq::end with an
/// af_seq::step of 1
///
typedef struct af_index_t{
    union {
        af_array arr;   ///< The af_array used for indexing
        af_seq   seq;   ///< The af_seq used for indexing
    } idx;

    bool     isSeq;     ///< If true the idx value represents a seq
    bool     isBatch;   ///< If true the seq object is a batch parameter
} af_index_t;


#if __cplusplus
namespace af
{

class dim4;
class array;
class seq;

///
/// \brief Wrapper for af_index.
///
/// This class is a wrapper for the af_index struct in the C interface. It
/// allows implicit type conversion from valid indexing types like int,
/// \ref af::seq, \ref af_seq, and \ref af::array.
///
/// \note This is a helper class and does not necessarily need to be created
/// explicitly. It is used in the operator() overloads to simplify the API.
///
class AFAPI index {

    af_index_t impl;
    public:
    ///
    /// \brief Default constructor. Equivalent to \ref af::span
    ///
    index();
    ~index();

    ///
    /// \brief Implicit int converter
    ///
    /// Indexes the af::array at index \p idx
    ///
    /// \param[in] idx is the id of the index
    ///
    /// \sa indexing
    ///
    index(const int idx);

    ///
    /// \brief Implicit seq converter
    ///
    /// Indexes the af::array using an \ref af::seq object
    ///
    /// \param[in] s0 is the set of indices to parse
    ///
    /// \sa indexing
    ///
    index(const af::seq& s0);

    ///
    /// \brief Implicit seq converter
    ///
    /// Indexes the af::array using an \ref af_seq object
    ///
    /// \param[in] s0 is the set of indices to parse
    ///
    /// \sa indexing
    ///
    index(const af_seq& s0);

    ///
    /// \brief Implicit int converter
    ///
    /// Indexes the af::array using an \ref af::array object
    ///
    /// \param[in] idx0 is the set of indices to parse
    ///
    /// \sa indexing
    ///
    index(const af::array& idx0);

#if AF_API_VERSION >= 31
    ///
    /// \brief Copy constructor
    ///
    /// \param[in] idx0 is index to copy.
    ///
    /// \sa indexing
    ///
    index(const index& idx0);
#endif

    ///
    /// \brief Returns true if the \ref af::index represents a af::span object
    ///
    /// \returns true if the af::index is an af::span
    ///
    bool isspan() const;

    ///
    /// \brief Gets the underlying af_index_t object
    ///
    /// \returns the af_index_t represented by this object
    ///
    const af_index_t& get() const;

#if AF_API_VERSION >= 31
    ///
    /// \brief Assigns idx0 to this index
    ///
    /// \param[in] idx0 is the index to be assigned to the /ref af::index
    /// \returns the reference to this
    ///
    ///
    index & operator=(const index& idx0);

#if __cplusplus > 199711L
    ///
    /// \brief Move constructor
    ///
    /// \param[in] idx0 is index to copy.
    ///
    index(index &&idx0);
    ///
    /// \brief Move assignment operator
    ///
    /// \param[in] idx0 is the index to be assigned to the /ref af::index
    /// \returns a reference to this
    ///
    index& operator=(index &&idx0);
#endif
#endif // AF_API_VERSION
};

///
/// Lookup the values of input array based on index
///
/// \param[in] in is input lookup array
/// \param[in] idx is lookup indices
/// \param[in] dim specifies the dimension for indexing
/// \returns an array containing values at locations specified by \p index
///
/// \ingroup index_func_index
///

AFAPI array lookup(const array &in, const array &idx, const int dim = -1);

#if AF_API_VERSION >= 31
///
/// Copy the values of an input array based on index
///
/// \param[out] dst The destination array
/// \param[in] src The source array
/// \param[in] idx0 The first index
/// \param[in] idx1 The second index (defaults to \ref af::span)
/// \param[in] idx2 The third index (defaults to \ref af::span)
/// \param[in] idx3 The fourth index (defaults to \ref af::span)
/// \ingroup index_func_index
///

AFAPI void copy(array &dst, const array &src,
                const index &idx0,
                const index &idx1 = span,
                const index &idx2 = span,
                const index &idx3 = span);
#endif

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    ///
    /// Lookup the values of input array based on sequences
    ///
    /// \param[out] out  output array containing values indexed by the
    ///                  sequences
    /// \param[in] in    is the input array
    /// \param[in] ndims is the number of sequences provided
    /// \param[in] index is an array of sequences
    ///
    /// \ingroup index_func_index

    AFAPI af_err af_index(  af_array *out,
                            const af_array in,
                            const unsigned ndims, const af_seq* const index);


    ///
    /// Lookup the values of input array based on index
    ///
    /// \param[out] out      output array containing values at locations
    ///                      specified by \p index
    /// \param[in] in        is input lookup array
    /// \param[in] indices   is lookup indices
    /// \param[in] dim       specifies the dimension for indexing
    ///
    /// \ingroup index_func_index
    ///

    AFAPI af_err af_lookup( af_array *out,
                            const af_array in, const af_array indices,
                            const unsigned dim);

    ///
    /// Copy and write values in the locations specified by the sequences
    ///
    /// \param[out] out     output array with values of \p rhs copied to
    ///                     locations specified by \p index and values from
    ///                     \p lhs in all other locations.
    /// \param[in] lhs      is array whose values are used for indices NOT
    ///                     specified by \p index
    /// \param[in] ndims    is the number of sequences provided
    /// \param[in] indices  is an array of sequences
    /// \param[in] rhs      is the array whose values are used for indices
    ///                     specified by \p index
    ///
    /// \ingroup index_func_assign
    ///

    AFAPI af_err af_assign_seq( af_array *out,
                                const af_array lhs,
                                const unsigned ndims, const af_seq* const indices,
                                const af_array rhs);

    ///
    /// \brief Indexing an array using \ref af_seq, or \ref af_array
    ///
    /// Generalized indexing function that accepts either af_array or af_seq
    /// along a dimension to index the input array and create the corresponding
    /// output array
    ///
    /// \param[out] out     output array containing values at indexed by
    ///                     the sequences
    /// \param[in] in       is the input array
    /// \param[in] ndims    is the number of \ref af_index_t provided
    /// \param[in] indices  is an array of \ref af_index_t objects
    ///
    /// \ingroup index_func_index
    ///
    AFAPI af_err af_index_gen(  af_array *out,
                                const af_array in,
                                const dim_t ndims, const af_index_t* indices);

    ///
    /// \brief Assignment of an array using \ref af_seq, or \ref af_array
    ///
    /// Generalized assignment function that accepts either af_array or af_seq
    /// along a dimension to assign elements form an input array to an output
    /// array
    ///
    /// \param[out] out     output array containing values at indexed by
    ///                     the sequences
    /// \param[in] lhs      is the input array
    /// \param[in] ndims    is the number of \ref af_index_t provided
    /// \param[in] indices  is an af_array of \ref af_index_t objects
    /// \param[in] rhs      is the array whose values will be assigned to \p lhs
    ///
    /// \ingroup index_func_assign
    ///
    AFAPI af_err af_assign_gen( af_array *out,
                                const af_array lhs,
                                const dim_t ndims, const af_index_t* indices,
                                const af_array rhs);

#ifdef __cplusplus
}
#endif
