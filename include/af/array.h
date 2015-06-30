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
#include <af/util.h>
#include <af/index.h>

#ifdef __cplusplus
#include <af/traits.hpp>
#include <vector>
namespace af
{

    class dim4;

    ///
    /// \brief A multi dimensional data container
    ///
    class AFAPI array {
        af_array   arr;


    public:
        ///
        /// \brief Updates the internal \ref af_array object
        ///
        /// /note This function will reduce the reference of the previous
        ///       \ref af_array object
        ///
        void set(af_array tmp);

        ///
        /// \brief Intermediate data class. Used for assignment and indexing
        ///
        class AFAPI array_proxy
        {
            struct array_proxy_impl;    //forward declaration
            array_proxy_impl *impl;     // implementation

        public:
            array_proxy(array& par, af_index_t *ssss, bool linear = false);
            array_proxy(const array_proxy &other);
#if __cplusplus > 199711L
            array_proxy(array_proxy &&other);
            array_proxy & operator=(array_proxy &&other);
#endif
            ~array_proxy();

            // Implicit conversion operators
            operator array() const;
            operator array();

#define ASSIGN(OP)                                                  \
            array_proxy& operator OP(const array_proxy &a);         \
            array_proxy& operator OP(const array &a);               \
            array_proxy& operator OP(const double &a);              \
            array_proxy& operator OP(const cdouble &a);             \
            array_proxy& operator OP(const cfloat &a);              \
            array_proxy& operator OP(const float &a);               \
            array_proxy& operator OP(const int &a);                 \
            array_proxy& operator OP(const unsigned &a);            \
            array_proxy& operator OP(const bool &a);                \
            array_proxy& operator OP(const char &a);                \
            array_proxy& operator OP(const unsigned char &a);       \
            array_proxy& operator OP(const long  &a);               \
            array_proxy& operator OP(const unsigned long &a);       \
            array_proxy& operator OP(const long long  &a);          \
            array_proxy& operator OP(const unsigned long long &a);  \

            ASSIGN(=)
            ASSIGN(+=)
            ASSIGN(-=)
            ASSIGN(*=)
            ASSIGN(/=)
#undef ASSIGN

            // af::array member functions. same behavior as those below
            af_array get();
            af_array get() const;
            dim_t elements() const;
            template<typename T> T* host() const;
            void host(void *ptr) const;
            dtype type() const;
            dim4 dims() const;
            dim_t dims(unsigned dim) const;
            unsigned numdims() const;
            size_t bytes() const;
            array copy() const;
            bool isempty() const;
            bool isscalar() const;
            bool isvector() const;
            bool isrow() const;
            bool iscolumn() const;
            bool iscomplex() const;
            inline bool isreal() const { return !iscomplex(); }
            bool isdouble() const;
            bool issingle() const;
            bool isrealfloating() const;
            bool isfloating() const;
            bool isinteger() const;
            bool isbool() const;
            void eval() const;
            array as(dtype type) const;
            array T() const;
            array H() const;
            template<typename T> T scalar() const;
            template<typename T> T* device() const;
            void unlock() const;
            void lock() const;

                  array::array_proxy row(int index);
            const array::array_proxy row(int index) const;

                  array::array_proxy rows(int first, int last);
            const array::array_proxy rows(int first, int last) const;

                  array::array_proxy col(int index);
            const array::array_proxy col(int index) const;
                  array::array_proxy cols(int first, int last);
            const array::array_proxy cols(int first, int last) const;

                  array::array_proxy slice(int index);
            const array::array_proxy slice(int index) const;

                  array::array_proxy slices(int first, int last);
            const array::array_proxy slices(int first, int last) const;
        };

        //array(af_array in, const array *par, af_index_t seqs[4]);
        /**
            \ingroup construct_mat
            @{
        */
        /**
            Create undimensioned array (no data, undefined size)

            \code
            array A, B, C;   // creates three arrays called A, B and C
            \endcode
        */
        array();

        /**
            Creates an array from an \ref af_array handle
            \param handle the af_array object.
         */
        explicit
        array(const af_array handle);

        /**
            Creates a copy to the \p in array.

            \param in The input \ref array
         */
        array(const array& in);

        /**
            Allocate a one-dimensional array of a specified size with undefined
            contents

            Declare a two-dimensional array by passing the
            number of rows and the number of columns as the first two parameters.
            The (optional) second parameter is the type of the array. The default
            type is f32 or 4-byte single-precision floating-point numbers.

            \code
            // allocate space for an array with 10 rows
            array A(10);          // type is the default f32

            // allocate space for a column vector with 100 rows
            array A(100, f64);    // f64 = double precision
            \endcode

            \param[in] dim0 number of columns in the array
            \param[in] ty   optional label describing the data type
                       (default is f32)

        */
        array(dim_t dim0, dtype ty = f32);

        /**
            Allocate a two-dimensional array of a specified size with undefined
            contents

            Declare a two-dimensional array by passing the
            number of rows and the number of columns as the first two parameters.
            The (optional) third parameter is the type of the array. The default
            type is f32 or 4-byte single-precision floating-point numbers.

            \code
            // allocate space for an array with 10 rows and 8 columns
            array A(10, 8);          // type is the default f32

            // allocate space for a column vector with 100 rows (and 1 column)
            array A(100, 1, f64);    // f64 = double precision
            \endcode

            \param[in] dim0 number of columns in the array
            \param[in] dim1 number of rows in the array
            \param[in] ty optional label describing the data type
                       (default is f32)

        */
        array(dim_t dim0, dim_t dim1, dtype ty = f32);

        /**
            Allocate a three-dimensional (3D) array of a specified size with
            undefined contents

            This is useful to quickly declare a three-dimensional array by
            passing the size as the first three parameters. The (optional)
            fourth parameter is the type of the array. The default type is f32
            or 4-byte single-precision floating point numbers.

            \code
            // allocate space for a 10 x 10 x 10 array
            array A(10, 10, 10);          // type is the default f32

            // allocate space for a 3D, double precision array
            array A(10, 10, 10, f64);     // f64 = double precision
            \endcode

            \param[in] dim0 first dimension of the array
            \param[in] dim1 second dimension of the array
            \param[in] dim2 third dimension of the array
            \param[in] ty optional label describing the data type
                       (default is f32)

        */
        array(dim_t dim0, dim_t dim1, dim_t dim2, dtype ty = f32);

        /**
            Allocate a four-dimensional (4D) array of a specified size with
            undefined contents

            This is useful to quickly declare a four-dimensional array by
            passing the size as the first four parameters. The (optional) fifth
            parameter is the type of the array. The default type is f32 or
            4-byte floating point numbers.

            \code
            // allocate space for a 10 x 10 x 10 x 20 array
            array A(10, 10, 10, 20);          // type is the default f32

            // allocate space for a 4D, double precision array
            array A(10, 10, 10, 20, f64);     // f64 = double precision
            \endcode

            \param[in] dim0 first dimension of the array
            \param[in] dim1 second dimension of the array
            \param[in] dim2 third dimension of the array
            \param[in] dim3 fourth dimension of the array
            \param[in] ty optional label describing the data type
                       (default is f32)

        */
        array(dim_t dim0, dim_t dim1, dim_t dim2, dim_t dim3, dtype ty = f32);

        /**
            Allocate an array of a specified size with undefined contents

            This can be useful when the dimensions of the array are calculated
            somewhere else within the code. The first parameter specifies the
            size of the array via dim4(). The second parameter is the type of
            the array. The default type is f32 or 4-byte
            single-precision floating point numbers.

            \code

            // create a two-dimensional 10 x 10 array
            dim4 dims(10, 10);       // converted to (10, 10, 1, 1)
            array a1(dims);          // create the array (type is f32, the default)

            // create a three-dimensional 10 x 10 x 20 array
            dim4 dims(10, 10, 20);   // converted to (10, 10, 20, 1)
            array a2(dims,f64);      // f64 = double precision

            \endcode

            \param[in] dims size of the array
            \param[in] ty optional label describing the data type
                       (default is f32)
        */
        explicit
        array(const dim4& dims, dtype ty = f32);

        /**
            Create a column vector on the device using a host/device pointer

            This function can be used to transfer data from a host or device
            pointer to an array object on the device with one column. The type
            of the array is automatically matched to the type of the data.

            Depending on the specified size of the column vector, the data will
            be copied partially or completely. However, the user needs to be
            careful to ensure that the array size is not larger than the number
            of elements in the input buffer.

            \param[in] dim0     number of elements in the column vector
            \param[in] pointer  pointer (points to a buffer on the host/device)
            \param[in] src      source of the data (default is afHost, can also
                                be afDevice)

            \code
            // allocate data on the host
            int h_buffer[] = {23, 34, 18, 99, 34};

            array A(4, h_buffer);   // copy host data to device
                                    //
                                    // A = 23
                                    //   = 34
                                    //   = 18
                                    //   = 99

            \endcode
        */
        template<typename T>
        array(dim_t dim0,
              const T *pointer, af::source src=afHost);


        /**
            Create a 2D array on the device using a host/device pointer

            This function copies data from the location specified by the
            pointer to a 2D array on the device. The data is arranged in
            "column-major" format (similar to that used by FORTRAN).

            Note that this is a synchronous copy. The elements are not
            actually filled until this array is evaluated or used in the
            evaluation of some other expression that uses this array object.

            \param[in] dim0     number of rows
            \param[in] dim1     number of columns
            \param[in] pointer  pointer (points to a buffer on the host/device)
            \param[in] src      source of the data (default is afHost, can also
                                be \ref afDevice)

            \code
            int h_buffer[] = {0, 1, 2, 3, 4, 5};  // host array
            array A(2, 3, h_buffer);              // copy host data to device
            \endcode

            \image html 2dArray.png
        */
        template<typename T>
        array(dim_t dim0, dim_t dim1,
              const T *pointer, af::source src=afHost);


        /**
            Create a 3D array on the device using a host/device pointer

            This function copies data from the location specified by the pointer
            to a 3D array on the device. The data is arranged in "column-major"
            format (similar to that used by FORTRAN).

            \param[in] dim0     first dimension
            \param[in] dim1     second dimension
            \param[in] dim2     third dimension
            \param[in] pointer  pointer (points to a buffer on the host/device)
            \param[in] src      source of the data (default is \ref afHost, can
                                also be \ref afDevice)

            \code
            int h_buffer[] = {0, 1, 2, 3, 4, 5, 6, 7, 8
                              9, 0, 1, 2, 3, 4, 5, 6, 7};   // host array

            array A(3, 3, 2,  h_buffer);   // copy host data to 3D device array
            \endcode

            \image html 3dArray.png
        */
        template<typename T>
        array(dim_t dim0, dim_t dim1, dim_t dim2,
              const T *pointer, af::source src=afHost);


        /**
            Create a 4D array on the device using a host/device pointer

            This function copies data from the location specified by the pointer
            to a 4D array on the device. The data is arranged in "column-major"
            format (similar to that used by FORTRAN).

            \param[in] dim0     first dimension
            \param[in] dim1     second dimension
            \param[in] dim2     third dimension
            \param[in] dim3     fourth dimension
            \param[in] pointer  pointer (points to a buffer on the host/device)
            \param[in] src      source of the data (default is afHost, can also
                                be \ref afDevice)

            \code
            int h_buffer[] = {0, 1, 2, 3,
                              4, 5, 6, 7,
                              8, 9, 0, 1,
                              2, 3, 4, 5};   // host array with 16 elements

            array A(2, 2, 2, 2, h_buffer);   // copy host data to 4D device array
            \endcode
        */
        template<typename T>
        array(dim_t dim0, dim_t dim1, dim_t dim2, dim_t dim3,
              const T *pointer, af::source src=afHost);

        /**
            Create an array of specified size on the device using a host/device
            pointer

            This function copies data from the location specified by the
            pointer to a 1D/2D/3D/4D array on the device. The data is arranged
            in "column-major" format (similar to that used by FORTRAN).

            \param[in] dims    vector data type containing the dimension of the
                               \ref array
            \param[in] pointer pointer (points to a buffer on the host/device)
            \param[in] src     source of the data (default is afHost, can also
                                be \ref afDevice)

            \code
            int h_buffer[] = {0, 1, 2, 3,    // host array with 16 elements
                              4, 5, 6, 7,    // written in "row-major" format
                              8, 9, 0, 1,
                              2, 3, 4, 5};

            dim4 dims(4, 4);

            array A(dims, h_buffer);         // A  =  0  4  8  2
                                             //       1  5  9  3
                                             //       2  6  0  4
                                             //       3  7  1  5

                                             // Note the "column-major" ordering
                                             // used in ArrayFire
            \endcode
        */
        template<typename T>
        explicit
        array(const dim4& dims,
              const T *pointer, af::source src=afHost);

        /**
           Adjust the dimensions of an N-D array (fast).

           This operation simply rearranges the description of the array.
           No memory transfers or transformations are  performed. The total
           number of elements must not change.

           \code
           float f[] = {1,2,3,4};
           array a(2,2,f);
           //a=[1 3]
           //  [2 4]

           array b = array(a, dim4(4));
           //b=[1]
           //  [2]
           //  [3]
           //  [4]

           array c = array(a, b.T().dims() );
           //c=[1 2 3 4]
           \endcode

           \param[in] input
           \param[in] dims total number of elements must not change.
           \return same underlying array data with different dimensions
        */
        array(const array& input, const dim4& dims);

        /**
           Adjust the dimensions of an N-D array (fast).

           This operation simply rearranges the description of the array.
           No memory transfers or transformations are  performed. The total
           number of elements must not change.

           \code

           float f[] = {1,2,3,4};
           array a(2,2,f);
           //a=[1 3]
           //  [2 4]

           array b = array(a, 4);
           //b=[1]
           //  [2]
           //  [3]
           //  [4]

           array c = array(a, 1, 4);
           //c=[1 2 3 4]
           \endcode

           \param[in] input
           \param[in] dim0 first dimension
           \param[in] dim1 second dimension
           \param[in] dim2 third dimension
           \param[in] dim3 fourth dimension
           \return same underlying array data with different dimensions
        */
        array(  const array& input,
                const dim_t dim0, const dim_t dim1 = 1,
                const dim_t dim2 = 1, const dim_t dim3 = 1);

        /**
            @}
        */

        /**
           \ingroup method_mat
           @{
        */

        /**
           get the \ref af_array handle
        */
        af_array get();

        /**
           get the \ref af_array handle
        */
        af_array get() const;

        /**
           get the number of elements in array
        */
        dim_t elements() const;

        /**
           Copy array data to host and return host pointer
        */
        template<typename T> T* host() const;

        /**
           Copy array data to existing host pointer
        */
        void host(void *ptr) const;

        /**
           Perform deep copy from host/device pointer to an existing array
        */
        template<typename T> void write(const T *ptr, const size_t bytes, af::source src = afHost);

        /**
           Get array data type
        */
        dtype type() const;

        /**
           Get dimensions of the array
        */
        dim4 dims() const;

        /**
           Get dimensions of the array
        */
        dim_t dims(unsigned dim) const;

        /**
           Get the number of dimensions of the array
        */
        unsigned numdims() const;

        /**
           Get the size of the array in bytes
        */
        size_t bytes() const;

        /**
           Perform deep copy of the array
        */
        array copy() const;

        /**
           \brief Returns true of the array is empty
         */
        bool isempty() const;

        /**
           \brief Returns true of the array contains only one value
         */
        bool isscalar() const;

        /**
           \brief Returns true if only one of the array dimensions has more than one element
        */
        bool isvector() const;

        /**
           \brief Returns true if only the second dimension has more than one element
        */
        bool isrow() const;

        /**
           \brief Returns true if only the first dimension has more than one element
        */
        bool iscolumn() const;

        /**
           \brief Returns true if the array type is \ref c32 or \ref c64
        */
        bool iscomplex() const;

        /**
           \brief Returns true if the array type is neither \ref c32 nor \ref c64
        */
        inline bool isreal() const { return !iscomplex(); }

        /**
           \brief Returns true if the array type is \ref f64 or \ref c64
        */
        bool isdouble() const;

        /**
           \brief Returns true if the array type is neither \ref f64 nor \ref c64
        */
        bool issingle() const;

        /**
           \brief Returns true if the array type is \ref f32 or \ref f64
        */
        bool isrealfloating() const;

        /**
           \brief Returns true if the array type is \ref f32, \ref f64, \ref c32 or \ref c64
        */
        bool isfloating() const;

        /**
           \brief Returns true if the array type is \ref u8, \ref b8, \ref s32 \ref u32, \ref s64, \ref u64
        */
        bool isinteger() const;

        /**
           \brief Returns true if the array type is \ref b8
        */
        bool isbool() const;

        /**
           \brief Evaluate any JIT expressions to generate data for the array
        */
        void eval() const;

        /**
           \brief Get the first element of the array as a scalar

           \note This is recommended for use while debugging. Calling this method constantly reduces performance.
        */
        template<typename T> T scalar() const;

        /**
           @}
        */


        /**
           \defgroup device_func_device array::device<T>

           Get the device pointer from the array
           @{

           \ingroup arrayfire_func
           \ingroup device_mat
        */
        template<typename T> T* device() const;
        /**
           @}
        */

        // INDEXING
        // Single arguments


        /// \ingroup array_mem_operator_paren
        /// @{
        ///
        /// \brief Gets a reference to a set of linear elements
        ///
        /// \copydetails array_mem_operator_paren_one
        ///
        /// \param[in] s0   is sequence of linear indices
        ///
        /// \returns A reference to the array at the given index
        ///
              array::array_proxy operator()(const index &s0);

        /// \copydoc operator()(const index &)
        const array::array_proxy operator()(const index &s0) const;


        ///
        /// \brief Gets a reference to a sub array
        ///
        /// \copydetails array_mem_operator_paren_many
        ///
        /// \param[in] s0   is sequence of indices along the first dimension
        /// \param[in] s1   is sequence of indices along the second dimension
        /// \param[in] s2   is sequence of indices along the third dimension
        /// \param[in] s3   is sequence of indices along the fourth dimension
        ///
        /// \returns A reference to the array at the given index
        ///
              array::array_proxy operator()(const index &s0,
                                            const index &s1,
                                            const index &s2 = span,
                                            const index &s3 = span);

        /// \copydoc operator()(const index &, const index &, const index &, const index &)
        const array::array_proxy operator()(const index &s0,
                                            const index &s1,
                                            const index &s2 = span,
                                            const index &s3 = span) const;
        /// @}

        /// \ingroup array_mem_row
        /// @{
        ///
        /// \brief Returns a reference to a row
        ///
        /// \copydetails array_mem_row
        ///
        /// \param[in]  index is the index of the row to be returned
        ///
        /// \returns a reference to a row defined by \p index
        ///
              array::array_proxy row(int index);
        const array::array_proxy row(int index) const; ///< \copydoc row

        ///
        /// \brief Returns a reference to sequence of rows
        ///
        /// \copydetails array_mem_row
        ///
        /// \param[in]  first is the index of the row to be returned
        /// \param[in]  last is the index of the row to be returned
        ///
        /// \returns a reference to a set of rows
              array::array_proxy rows(int first, int last);
        const array::array_proxy rows(int first, int last) const; ///< \copydoc rows
        /// @}

        /// \ingroup array_mem_col
        /// @{
        ///
        /// \brief Returns a reference to a col
        ///
        /// \copydetails array_mem_col
        ///
        /// \param[in]  index is the index of the col to be returned
        ///
        /// \returns a reference to a col defined by \p index
        ///
              array::array_proxy col(int index);
        const array::array_proxy col(int index) const; ///< \copydoc col

        ///
        /// \brief Returns a reference to sequence of columns
        ///
        /// \copydetails array_mem_col
        ///
        /// \param[in]  first is the index of the columns to be returned
        /// \param[in]  last is the index of the columns to be returned
        ///
        /// \returns a reference to a set of columns
              array::array_proxy cols(int first, int last);
        const array::array_proxy cols(int first, int last) const; ///< \copydoc cols
        /// @}

        /// \ingroup array_mem_slice
        /// @{
        ///
        /// \brief Returns a reference to a matrix in a volume
        ///
        /// \copydetails array_mem_slice
        ///
        /// \param[in]  index is the index of the slice to be returned
        ///
        /// \returns a reference to a col
        ///
              array::array_proxy slice(int index);
        const array::array_proxy slice(int index) const; ///< \copydoc slice

        /// \brief Returns a reference to a matrix in a volume
        ///
        /// \copydetails array_mem_slice
        ///
        /// \param[in]  first is the index of the slices to be returned
        /// \param[in]  last is the index of the slices to be returned
        ///
        /// \returns a reference to a set of slice
              array::array_proxy slices(int first, int last);
        const array::array_proxy slices(int first, int last) const; ///< \copydoc slices
        /// @}

        /// \brief Converts the array into another type
        ///
        ///  \param[in] type is the desired type(f32, s64, etc.)
        /// \returns an array with the type specified by \p type
        /// \ingroup method_mat
        const array as(dtype type) const;


        ~array();

        // Transpose and Conjugate Tranpose
        array T() const;
        array H() const;

#define ASSIGN(OP)                                                                      \
        array& OP(const array &val);                                                    \
        array& OP(const double &val);              /**< \copydoc OP (const array &) */  \
        array& OP(const cdouble &val);             /**< \copydoc OP (const array &) */  \
        array& OP(const cfloat &val);              /**< \copydoc OP (const array &) */  \
        array& OP(const float &val);               /**< \copydoc OP (const array &) */  \
        array& OP(const int &val);                 /**< \copydoc OP (const array &) */  \
        array& OP(const unsigned &val);            /**< \copydoc OP (const array &) */  \
        array& OP(const bool &val);                /**< \copydoc OP (const array &) */  \
        array& OP(const char &val);                /**< \copydoc OP (const array &) */  \
        array& OP(const unsigned char &val);       /**< \copydoc OP (const array &) */  \
        array& OP(const long  &val);               /**< \copydoc OP (const array &) */  \
        array& OP(const unsigned long &val);       /**< \copydoc OP (const array &) */  \
        array& OP(const long long  &val);          /**< \copydoc OP (const array &) */  \
        array& OP(const unsigned long long &val);  /**< \copydoc OP (const array &) */  \

        /// \ingroup array_mem_operator_eq
        /// @{
        /// \brief Assignes the value(s) of val to the elements of the array.
        ///
        /// \param[in] val  is the value to be assigned to the /ref af::array
        /// \returns the reference to this
        ///
        /// \note   This is a copy on write operation. The copy only occurs when the
        ///          operator() is used on the left hand side.
        ASSIGN(operator=)
        /// @}

        /// \ingroup array_mem_operator_plus_eq
        /// @{
        /// \brief Adds the value(s) of val to the elements of the array.
        ///
        /// \param[in] val  is the value to be assigned to the /ref af::array
        /// \returns the reference to this
        ///
        /// \note   This is a copy on write operation. The copy only occurs when the
        ///          operator() is used on the left hand side.
        ASSIGN(operator+=)
        /// @}

        /// \ingroup array_mem_operator_minus_eq
        /// @{
        /// \brief Subtracts the value(s) of val to the elements of the array.
        ///
        /// \param[in] val  is the value to be assigned to the /ref af::array
        /// \returns the reference to this
        ///
        /// \note   This is a copy on write operation. The copy only occurs when the
        ///          operator() is used on the left hand side.
        ASSIGN(operator-=)
        /// @}

        /// \ingroup array_mem_operator_multiply_eq
        /// @{
        /// \brief Multiplies the value(s) of val to the elements of the array.
        ///
        /// \param[in] val  is the value to be assigned to the /ref af::array
        /// \returns the reference to this
        ///
        /// \note   This is a copy on write operation. The copy only occurs when the
        ///          operator() is used on the left hand side.
        ASSIGN(operator*=)
        /// @}

        /// \ingroup array_mem_operator_divide_eq
        /// @{
        /// \brief Divides the value(s) of val to the elements of the array.
        ///
        /// \param[in] val  is the value to be assigned to the /ref af::array
        /// \returns the reference to this
        ///
        /// \note   This is a copy on write operation. The copy only occurs when the
        ///          operator() is used on the left hand side.
        /// \ingroup array_mem_operator_divide_eq
        ASSIGN(operator/=)
        /// @}


#undef ASSIGN

        ///
        /// \brief Negates the values of the array
        /// \ingroup arith_func_neg
        ///
        /// \returns an \ref array with negated values
        array operator -() const;

        ///
        /// \brief Performs a not operation on the values of the array
        /// \ingroup arith_func_not
        ///
        /// \returns an \ref array with negated values
        array operator !() const;

        ///
        /// \brief Get the count of non-zero elements in the array
        ///
        /// For dense matrix, this is the same as count<int>(arr);
        int nonzeros() const;


        ///
        /// \brief Locks the device buffer in the memory manager.
        ///
        /// This method can be called to take control of the device pointer from the memory manager.
        /// While a buffer is locked, the memory manager does not free the memory.
        void lock() const;

        ///
        /// \brief Unlocks the device buffer in the memory manager.
        ///
        /// This method can be called after called after calling \ref array::lock()
        /// Calling this method gives back the control of the device pointer to the memory manager.
        void unlock() const;
    };
    // end of class array

#define BIN_OP(OP)                                                                                                       \
    AFAPI array OP (const array& lhs, const array& rhs);                                                                 \
    AFAPI array OP (const bool& lhs, const array& rhs);                 /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const int& lhs, const array& rhs);                  /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const unsigned& lhs, const array& rhs);             /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const char& lhs, const array& rhs);                 /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const unsigned char& lhs, const array& rhs);        /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const long& lhs, const array& rhs);                 /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const unsigned long& lhs, const array& rhs);        /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const long long& lhs, const array& rhs);            /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const unsigned long long& lhs, const array& rhs);   /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const double& lhs, const array& rhs);               /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const float& lhs, const array& rhs);                /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const cfloat& lhs, const array& rhs);               /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const cdouble& lhs, const array& rhs);              /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const bool& rhs);                 /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const int& rhs);                  /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const unsigned& rhs);             /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const char& rhs);                 /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const unsigned char& rhs);        /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const long& rhs);                 /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const unsigned long& rhs);        /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const long long& rhs);            /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const unsigned long long& rhs);   /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const double& rhs);               /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const float& rhs);                /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const cfloat& rhs);               /**< \copydoc OP (const array&, const array&) */ \
    AFAPI array OP (const array& lhs, const cdouble& rhs);              /**< \copydoc OP (const array&, const array&) */ \

    /// \ingroup arith_func_add
    /// @{
    /// \brief Adds two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array which is the sum of the \p lhs and \p rhs
    BIN_OP(operator+ )
    /// @}

    /// \ingroup arith_func_sub
    /// @{
    /// \brief Subtracts two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array which is the subtraction of the \p lhs and \p rhs
    BIN_OP(operator- )
    /// @}

    /// \ingroup arith_func_mul
    /// @{
    /// \brief Multiplies two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array which is the product of the \p lhs and \p rhs
    BIN_OP(operator* )
    /// @}

    /// \ingroup arith_func_div
    /// @{
    /// \brief Divides two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array which is the quotient of the \p lhs and \p rhs
    BIN_OP(operator/ )
    /// @}

    /// \ingroup arith_func_eq
    /// @{
    /// \brief Performs an equality operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array with the equality operation performed on each element
    BIN_OP(operator==)
    /// @}

    /// \ingroup arith_func_neq
    /// @{
    /// \brief Performs an inequality operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with the != operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator!=)
    /// @}

    /// \ingroup arith_func_lt
    /// @{
    /// \brief Performs a lower than operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with the < operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator< )
    /// @}

    /// \ingroup arith_func_le
    /// @{
    /// \brief Performs an lower or equal operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with the <= operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator<=)
    /// @}

    /// \ingroup arith_func_gt
    /// @{
    /// \brief Performs an greater than operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with the > operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator> )
    /// @}

    /// \ingroup arith_func_ge
    /// @{
    /// \brief Performs an greater or equal operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with the >= operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator>=)
    /// @}

    /// \ingroup arith_func_and
    /// @{
    /// \brief  Performs a logical AND operation on two arrays or an array and a
    ///         value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a logical AND operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator&&)
    /// @}

    /// \ingroup arith_func_or
    /// @{
    /// \brief  Performs an logical OR operation on two arrays or an array and a
    ///         value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a logical OR operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator||)
    /// @}

    /// \ingroup numeric_func_mod
    /// @{
    /// \brief Performs an modulo operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a modulo operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator% )
    /// @}

    /// \ingroup arith_func_bitand
    /// @{
    /// \brief  Performs an bitwise AND operation on two arrays or an array and
    ///         a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a bitwise AND operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator& )
    /// @}

    /// \ingroup arith_func_bitor
    /// @{
    /// \brief  Performs an bitwise OR operation on two arrays or an array and
    ///         a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a bitwise OR operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator| )
    /// @}

    /// \ingroup arith_func_bitxor
    /// @{
    /// \brief  Performs an bitwise XOR operation on two arrays or an array and
    ///         a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a bitwise OR operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator^ )
    /// @}

    /// \ingroup arith_func_shiftl
    /// @{
    /// \brief  Performs an left shift operation on two arrays or an array and a
    ///          value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a left shift operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator<<)
    /// @}

    /// \ingroup arith_func_shiftr
    /// @{
    /// \brief  Performs an right shift operation on two arrays or an array and a
    ///          value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a right shift operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator>>)
    /// @}

#undef BIN_OP

    /// Evaluate an expression (nonblocking).
    /**
       \ingroup method_mat
       @{
    */
    inline array &eval(array &a) { a.eval(); return a; }
    inline void eval(array &a, array &b) { eval(a); b.eval(); }
    inline void eval(array &a, array &b, array &c) { eval(a, b); c.eval(); }
    inline void eval(array &a, array &b, array &c, array &d) { eval(a, b, c); d.eval(); }
    inline void eval(array &a, array &b, array &c, array &d, array &e) { eval(a, b, c, d); e.eval(); }
    inline void eval(array &a, array &b, array &c, array &d, array &e, array &f) { eval(a, b, c, d, e); f.eval(); }
    /**
       @}
    */

}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    /**
       \ingroup construct_mat
       @{
    */

    /**
       Create an \ref af_array handle initialized with user defined data

       This function will create an \ref af_array handle from the memory provided in \p data

       \param[out]  arr The pointer to the returned object.
       \param[in]   data The data which will be loaded into the array
       \param[in]   ndims The number of dimensions read from the \p dims parameter
       \param[in]   dims A C pointer with \p ndims elements. Each value represents the size of that dimension
       \param[in]   type The type of the \ref af_array object

       \returns \ref AF_SUCCESS if the operation was a success
    */
    AFAPI af_err af_create_array(af_array *arr, const void * const data, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
       Create af_array handle

       \param[out]  arr The pointer to the retured object.
       \param[in]   ndims The number of dimensions read from the \p dims parameter
       \param[in]   dims A C pointer with \p ndims elements. Each value represents the size of that dimension
       \param[in]   type The type of the \ref af_array object

       \returns \ref AF_SUCCESS if the operation was a success
    */
    AFAPI af_err af_create_handle(af_array *arr, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
    @}
    */

    /**
       \ingroup method_mat
       @{

       Deep copy an array to another
    */
    AFAPI af_err af_copy_array(af_array *arr, const af_array in);

    /**
       Copy data from a C pointer (host/device) to an existing array.
    */
    AFAPI af_err af_write_array(af_array arr, const void *data, const size_t bytes, af_source src);

    /**
       Copy data from an af_array to a C pointer.

       Needs to used in conjunction with the two functions above
    */
    AFAPI af_err af_get_data_ptr(void *data, const af_array arr);

    /**
       \brief Reduce the reference count of the \ref af_array
    */
    AFAPI af_err af_release_array(af_array arr);

    /**
       Increments an \ref af_array reference count
    */
    AFAPI af_err af_retain_array(af_array *out, const af_array in);

    /**
       \ingroup method_mat
       @{

       Get the use count of `af_array`
    */
    AFAPI af_err af_get_data_ref_count(int *use_count, const af_array in);


    /**
       Evaluate any expressions in the Array
    */
    AFAPI af_err af_eval(af_array in);

    /**
      @}
    */

#ifdef __cplusplus
}
#endif
