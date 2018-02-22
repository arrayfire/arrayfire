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
        /// \note This function will reduce the reference of the previous
        ///       \ref af_array object
        ///
        void set(af_array tmp);

        ///
        /// \brief Intermediate data class. Used for assignment and indexing.
        ///
        /// \note This class is for internal book keeping while indexing. This class is not intended for use in user code.
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

#if AF_API_VERSION >= 32
#define ASSIGN(OP)                                                  \
            array_proxy& operator OP(const short &a);               \
            array_proxy& operator OP(const unsigned short &a);      \

            ASSIGN(=)
            ASSIGN(+=)
            ASSIGN(-=)
            ASSIGN(*=)
            ASSIGN(/=)
#undef ASSIGN
#endif

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
            size_t allocated() const;
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
#if AF_API_VERSION >= 34
            bool issparse() const;
#endif
            void eval() const;
            array as(dtype type) const;
            array T() const;
            array H() const;
            template<typename T> T scalar() const;
            template<typename T> T* device() const;
            void unlock() const;
#if AF_API_VERSION >= 31
            void lock() const;
#endif

#if AF_API_VERSION >= 34
            bool isLocked() const;
#endif

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

            \note If \p src is \ref afHost, the first \p dim0 elements are copied. If \p src is \ref afDevice, no copy is done; the array object wraps the device pointer AND takes ownership of the underlying memory.

        */
        template<typename T>
        array(dim_t dim0,
              const T *pointer, af::source src=afHost);


        /**
            Create a 2D array on the device using a host/device pointer

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

            \note If \p src is \ref afHost, the first \p dim0 * \p dim1 elements are copied. If \p src is \ref afDevice, no copy is done; the array object wraps the device pointer AND takes ownership of the underlying memory. The data is treated as column major format when performing linear algebra operations.
        */
        template<typename T>
        array(dim_t dim0, dim_t dim1,
              const T *pointer, af::source src=afHost);


        /**
            Create a 3D array on the device using a host/device pointer

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

            \note If \p src is \ref afHost, the first \p dim0 * \p dim1 * \p dim2 elements are copied. If \p src is \ref afDevice, no copy is done; the array object just wraps the device pointer and does not take ownership of the underlying memory. The data is treated as column major format when performing linear algebra operations.

            \image html 3dArray.png
        */
        template<typename T>
        array(dim_t dim0, dim_t dim1, dim_t dim2,
              const T *pointer, af::source src=afHost);


        /**
            Create a 4D array on the device using a host/device pointer

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

            \note If \p src is \ref afHost, the first \p dim0 * \p dim1 * \p dim2 * \p dim3 elements are copied. If \p src is \ref afDevice, no copy is done; the array object just wraps the device pointer and does not take ownership of the underlying memory. The data is treated as column major format when performing linear algebra operations.
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

            \note If \p src is \ref afHost, the first dims.elements() elements are copied. If \p src is \ref afDevice, no copy is done; the array object just wraps the device pointer and does not take ownership of the underlying memory. The data is treated as column major format when performing linear algebra operations.
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
           Get the size of the array in memory. This will return the parent's
           bytes() if the array is indexed.
        */
        size_t allocated() const;

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
           \brief Returns true if the array type is \ref u8, \ref b8, \ref s32 \ref u32, \ref s64, \ref u64, \ref s16, \ref u16
        */
        bool isinteger() const;

        /**
           \brief Returns true if the array type is \ref b8
        */
        bool isbool() const;

#if AF_API_VERSION >= 34
        /**
           \brief Returns true if the array is a sparse array
        */
        bool issparse() const;
#endif

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

           Get the device pointer from the array and lock the buffer in memory manager.
           @{

           The device memory returned by this function is not freed until unlock() is called.

           \ingroup arrayfire_func
           \ingroup device_mat
        */
        template<typename T> T* device() const;
        /**
           @}
        */

        // INDEXING
        // Single arguments

        /**
            \brief This operator returns a reference of the original array at a given coordinate.

            You can pass \ref af::seq, \ref af::array, or an int as its parameters.
            These references can be used for assignment or returning references
            to \ref af::array objects.

            If the \ref af::array is a multi-dimensional array then this coordinate
            will treated as the data as a linear array.

            \param[in] s0   is sequence of linear indices

            \returns A reference to the array at the given index

            \ingroup array_mem_operator_paren

        */
        array::array_proxy operator()(const index &s0);

        /**
            \copydoc operator()(const index &)

            \ingroup array_mem_operator_paren
        */
        const array::array_proxy operator()(const index &s0) const;


        /**
            \brief This operator returns a reference of the original array at a
            given coordinate.

            You can pass \ref af::seq, \ref af::array, or an int as it's parameters.
            These references can be used for assignment or returning references
            to \ref af::array objects.

            \param[in] s0   is sequence of indices along the first dimension
            \param[in] s1   is sequence of indices along the second dimension
            \param[in] s2   is sequence of indices along the third dimension
            \param[in] s3   is sequence of indices along the fourth dimension

            \returns A reference to the array at the given index

            \ingroup array_mem_operator_paren
        */
        array::array_proxy operator()(const index &s0,
                                      const index &s1,
                                      const index &s2 = span,
                                      const index &s3 = span);

        /**
            \copydoc operator()(const index &, const index &, const index &, const index &)

            \ingroup array_mem_operator_paren
        */
        const array::array_proxy operator()(const index &s0,
                                            const index &s1,
                                            const index &s2 = span,
                                            const index &s3 = span) const;


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

        /// \brief Get the transposed the array
        ///
        /// \returns Transposed matrix
        /// \ingroup method_mat
        array T() const;
        /// \brief Get the conjugate-transpose of the current array
        ///
        /// \returns conjugate-transpose matrix
        /// \ingroup method_mat
        array H() const;

#define ASSIGN_(OP, DOXY_STRING)                               \
        array& OP(const array &val);                           \
        array& OP(const double &val);             DOXY_STRING  \
        array& OP(const cdouble &val);            DOXY_STRING  \
        array& OP(const cfloat &val);             DOXY_STRING  \
        array& OP(const float &val);              DOXY_STRING  \
        array& OP(const int &val);                DOXY_STRING  \
        array& OP(const unsigned &val);           DOXY_STRING  \
        array& OP(const bool &val);               DOXY_STRING  \
        array& OP(const char &val);               DOXY_STRING  \
        array& OP(const unsigned char &val);      DOXY_STRING  \
        array& OP(const long  &val);              DOXY_STRING  \
        array& OP(const unsigned long &val);      DOXY_STRING  \
        array& OP(const long long  &val);         DOXY_STRING  \
        array& OP(const unsigned long long &val); DOXY_STRING  \

#if AF_API_VERSION >= 32
#define ASSIGN(OP, DOXY_STRING)                                \
        ASSIGN_(OP, DOXY_STRING)                               \
        array& OP(const short  &val);             DOXY_STRING  \
        array& OP(const unsigned short &val);     DOXY_STRING  \

#else
#define ASSIGN(OP, DOXY_STRING) ASSIGN_(OP, DOXY_STRING)
#endif


        /// \ingroup array_mem_operator_eq
        /// @{
        /// \brief Assignes the value(s) of val to the elements of the array.
        ///
        /// \param[in] val  is the value to be assigned to the /ref af::array
        /// \returns the reference to this
        ///
        /// \note   This is a copy on write operation. The copy only occurs when the
        ///          operator() is used on the left hand side.
        ASSIGN(operator=, /**< \copydoc operator=(const array &) */)
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
        ASSIGN(operator+=, /**< \copydoc operator+=(const array &) */)
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
        ASSIGN(operator-=, /**< \copydoc operator-=(const array &) */)
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
        ASSIGN(operator*=, /**< \copydoc operator*=(const array &) */)
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
        ASSIGN(operator/=, /**< \copydoc operator/=(const array &) */)
        /// @}


#undef ASSIGN
#undef ASSIGN_

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
        /// While a buffer is locked, the memory manager doesn't free the memory until unlock() is invoked.
        void lock() const;


#if AF_API_VERSION >= 34
        ///
        /// \brief Query if the array has been locked by the user.
        ///
        /// An array can be locked by the user by calling `arry.lock` or `arr.device`
        /// or `getRawPtr` function.
        bool isLocked() const;
#endif


        ///
        /// \brief Unlocks the device buffer in the memory manager.
        ///
        /// This method can be called after called after calling \ref array::lock()
        /// Calling this method gives back the control of the device pointer to the memory manager.
        void unlock() const;
    };
    // end of class array

#define BIN_OP_(OP, DOXY_STRING)                                                   \
    AFAPI array OP (const array& lhs, const array& rhs);                           \
    AFAPI array OP (const bool& lhs, const array& rhs);                DOXY_STRING \
    AFAPI array OP (const int& lhs, const array& rhs);                 DOXY_STRING \
    AFAPI array OP (const unsigned& lhs, const array& rhs);            DOXY_STRING \
    AFAPI array OP (const char& lhs, const array& rhs);                DOXY_STRING \
    AFAPI array OP (const unsigned char& lhs, const array& rhs);       DOXY_STRING \
    AFAPI array OP (const long& lhs, const array& rhs);                DOXY_STRING \
    AFAPI array OP (const unsigned long& lhs, const array& rhs);       DOXY_STRING \
    AFAPI array OP (const long long& lhs, const array& rhs);           DOXY_STRING \
    AFAPI array OP (const unsigned long long& lhs, const array& rhs);  DOXY_STRING \
    AFAPI array OP (const double& lhs, const array& rhs);              DOXY_STRING \
    AFAPI array OP (const float& lhs, const array& rhs);               DOXY_STRING \
    AFAPI array OP (const cfloat& lhs, const array& rhs);              DOXY_STRING \
    AFAPI array OP (const cdouble& lhs, const array& rhs);             DOXY_STRING \
    AFAPI array OP (const array& lhs, const bool& rhs);                DOXY_STRING \
    AFAPI array OP (const array& lhs, const int& rhs);                 DOXY_STRING \
    AFAPI array OP (const array& lhs, const unsigned& rhs);            DOXY_STRING \
    AFAPI array OP (const array& lhs, const char& rhs);                DOXY_STRING \
    AFAPI array OP (const array& lhs, const unsigned char& rhs);       DOXY_STRING \
    AFAPI array OP (const array& lhs, const long& rhs);                DOXY_STRING \
    AFAPI array OP (const array& lhs, const unsigned long& rhs);       DOXY_STRING \
    AFAPI array OP (const array& lhs, const long long& rhs);           DOXY_STRING \
    AFAPI array OP (const array& lhs, const unsigned long long& rhs);  DOXY_STRING \
    AFAPI array OP (const array& lhs, const double& rhs);              DOXY_STRING \
    AFAPI array OP (const array& lhs, const float& rhs);               DOXY_STRING \
    AFAPI array OP (const array& lhs, const cfloat& rhs);              DOXY_STRING \
    AFAPI array OP (const array& lhs, const cdouble& rhs);             DOXY_STRING \

#if AF_API_VERSION >= 32
#define BIN_OP(OP, DOXY_STRING)                                                    \
        BIN_OP_(OP, DOXY_STRING)                                                   \
        AFAPI array OP (const short& lhs, const array& rhs);           DOXY_STRING \
        AFAPI array OP (const unsigned short& lhs, const array& rhs);  DOXY_STRING \
        AFAPI array OP (const array& lhs, const short& rhs);           DOXY_STRING \
        AFAPI array OP (const array& lhs, const unsigned short& rhs);  DOXY_STRING \

#else
#define BIN_OP(OP, DOXY_STRING) BIN_OP_(OP, DOXY_STRING)
#endif

    /// \ingroup arith_func_add
    /// @{
    /// \brief Adds two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array which is the sum of the \p lhs and \p rhs
    BIN_OP(operator+, /**< \copydoc operator+ (const array&, const array&) */)
    /// @}

    /// \ingroup arith_func_sub
    /// @{
    /// \brief Subtracts two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array which is the subtraction of the \p lhs and \p rhs
    BIN_OP(operator-, /**< \copydoc operator- (const array&, const array&) */)
    /// @}

    /// \ingroup arith_func_mul
    /// @{
    /// \brief Multiplies two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array which is the product of the \p lhs and \p rhs
    BIN_OP(operator*, /**< \copydoc operator* (const array&, const array&) */)
    /// @}

    /// \ingroup arith_func_div
    /// @{
    /// \brief Divides two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array which is the quotient of the \p lhs and \p rhs
    BIN_OP(operator/, /**< \copydoc operator/ (const array&, const array&) */)
    /// @}

    /// \ingroup arith_func_eq
    /// @{
    /// \brief Performs an equality operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns an array of type b8 with the equality operation performed on each element
    BIN_OP(operator==, /**< \copydoc operator== (const array&, const array&) */)
    /// @}

    /// \ingroup arith_func_neq
    /// @{
    /// \brief Performs an inequality operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with the != operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator!=, /**< \copydoc operator!= (const array&, const array&) */)
    /// @}

    /// \ingroup arith_func_lt
    /// @{
    /// \brief Performs a lower than operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with the < operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator<, /**< \copydoc operator< (const array&, const array&) */)
    /// @}

    /// \ingroup arith_func_le
    /// @{
    /// \brief Performs an lower or equal operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with the <= operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator<=, /**< \copydoc operator<= (const array&, const array&) */)
    /// @}

    /// \ingroup arith_func_gt
    /// @{
    /// \brief Performs an greater than operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with the > operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator>, /**< \copydoc operator> (const array&, const array&) */)
    /// @}

    /// \ingroup arith_func_ge
    /// @{
    /// \brief Performs an greater or equal operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with the >= operation performed on each element
    ///             of \p lhs and \p rhs
    BIN_OP(operator>=, /**< \copydoc operator>= (const array&, const array&) */)
    /// @}

    /// \ingroup arith_func_and
    /// @{
    /// \brief  Performs a logical AND operation on two arrays or an array and a
    ///         value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with a logical AND operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator&&, /**< \copydoc operator&& (const array&, const array&) */)
    /// @}

    /// \ingroup arith_func_or
    /// @{
    /// \brief  Performs an logical OR operation on two arrays or an array and a
    ///         value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array of type b8 with a logical OR operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator||, /**< \copydoc operator|| (const array&, const array&) */)
    /// @}

    /// \ingroup arith_func_mod
    /// @{
    /// \brief Performs an modulo operation on two arrays or an array and a value.
    ///
    /// \param[in] lhs the left hand side value of the operand
    /// \param[in] rhs the right hand side value of the operand
    ///
    /// \returns    an array with a modulo operation performed on each
    ///             element of \p lhs and \p rhs
    BIN_OP(operator%, /**< \copydoc operator% (const array&, const array&) */)
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
    BIN_OP(operator&, /**< \copydoc operator& (const array&, const array&) */)
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
    BIN_OP(operator|, /**< \copydoc operator| (const array&, const array&) */)
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
    BIN_OP(operator^, /**< \copydoc operator^ (const array&, const array&) */)
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
    BIN_OP(operator<<, /**< \copydoc operator<< (const array&, const array&) */)
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
    BIN_OP(operator>>, /**< \copydoc operator>> (const array&, const array&) */)
    /// @}

#undef BIN_OP
#undef BIN_OP_

    /// Evaluate an expression (nonblocking).
    /**
       \ingroup method_mat
       @{
    */
    inline array &eval(array &a) { a.eval(); return a; }

#if AF_API_VERSION >= 34
    ///
    /// Evaluate multiple arrays simultaneously
    ///
    AFAPI void eval(int num, array **arrays);
#endif

    inline void eval(array &a, array &b)
    {
#if AF_API_VERSION >= 34
        array *arrays[] = {&a, &b};
        return eval(2, arrays);
#else
        eval(a); b.eval();
#endif
    }

    inline void eval(array &a, array &b, array &c)
    {
#if AF_API_VERSION >= 34
        array *arrays[] = {&a, &b, &c};
        return eval(3, arrays);
#else
        eval(a, b); c.eval();
#endif
    }

    inline void eval(array &a, array &b, array &c, array &d)
    {
#if AF_API_VERSION >= 34
        array *arrays[] = {&a, &b, &c, &d};
        return eval(4, arrays);
#else
        eval(a, b, c); d.eval();
#endif

    }

    inline void eval(array &a, array &b, array &c, array &d, array &e)
    {
#if AF_API_VERSION >= 34
        array *arrays[] = {&a, &b, &c, &d, &e};
        return eval(5, arrays);
#else
        eval(a, b, c, d); e.eval();
#endif
    }

    inline void eval(array &a, array &b, array &c, array &d, array &e, array &f)
    {
#if AF_API_VERSION >= 34
        array *arrays[] = {&a, &b, &c, &d, &e, &f};
        return eval(6, arrays);
#else
        eval(a, b, c, d, e); f.eval();
#endif
    }

#if AF_API_VERSION >= 34
    ///
    /// Turn the manual eval flag on or off
    ///
    AFAPI void setManualEvalFlag(bool flag);
#endif

#if AF_API_VERSION >= 34
    /// Get the manual eval flag
    AFAPI bool getManualEvalFlag();
#endif

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

#if AF_API_VERSION >= 31
    /**
       \ingroup method_mat
       @{

       Get the use count of `af_array`
    */
    AFAPI af_err af_get_data_ref_count(int *use_count, const af_array in);
#endif


    /**
       Evaluate any expressions in the Array
    */
    AFAPI af_err af_eval(af_array in);

    /**
      @}
    */


#if AF_API_VERSION >= 34
    /**
       Evaluate multiple arrays together
    */
    AFAPI af_err af_eval_multiple(const int num, af_array *arrays);
    /**
      @}
    */
#endif

#if AF_API_VERSION >= 34
    /**
       Turn the manual eval flag on or off
    */
    AFAPI af_err af_set_manual_eval_flag(bool flag);
    /**
      @}
    */
#endif

#if AF_API_VERSION >= 34
    /**
       Get the manual eval flag
    */
    AFAPI af_err af_get_manual_eval_flag(bool *flag);
    /**
      @}
    */
#endif

    /**
        \ingroup method_mat
        @{
    */
    /**
        \brief Gets the number of elements in an array.

        \param[out] elems is the output that contains number of elements of \p arr
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_get_elements(dim_t *elems, const af_array arr);

    /**
        \brief Gets the type of an array.

        \param[out] type is the output that contains the type of \p arr
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_get_type(af_dtype *type, const af_array arr);

    /**
        \brief Gets the dimensions of an array.

        \param[out] d0 is the output that contains the size of first dimension of \p arr
        \param[out] d1 is the output that contains the size of second dimension of \p arr
        \param[out] d2 is the output that contains the size of third dimension of \p arr
        \param[out] d3 is the output that contains the size of fourth dimension of \p arr
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_get_dims(dim_t *d0, dim_t *d1, dim_t *d2, dim_t *d3,
                             const af_array arr);

    /**
        \brief Gets the number of dimensions of an array.

        \param[out] result is the output that contains the number of dims of \p arr
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_get_numdims(unsigned *result, const af_array arr);

    /**
        \brief Check if an array is empty.

        \param[out] result is true if elements of arr is 0, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_empty        (bool *result, const af_array arr);

    /**
        \brief Check if an array is scalar, ie. single element.

        \param[out] result is true if elements of arr is 1, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_scalar       (bool *result, const af_array arr);

    /**
        \brief Check if an array is row vector.

        \param[out] result is true if arr has dims [1 x 1 1], false otherwise
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_row          (bool *result, const af_array arr);

    /**
        \brief Check if an array is a column vector

        \param[out] result is true if arr has dims [x 1 1 1], false otherwise
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_column       (bool *result, const af_array arr);

    /**
        \brief Check if an array is a vector

        A vector is any array that has exactly 1 dimension not equal to 1.

        \param[out] result is true if arr is a vector, false otherwise
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_vector       (bool *result, const af_array arr);

    /**
        \brief Check if an array is complex type

        \param[out] result is true if arr is of type \ref c32 or \ref c64, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_complex      (bool *result, const af_array arr);

    /**
        \brief Check if an array is real type

        This is mutually exclusive to \ref af_is_complex

        \param[out] result is true if arr is NOT of type \ref c32 or \ref c64, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_real         (bool *result, const af_array arr);

    /**
        \brief Check if an array is double precision type

        \param[out] result is true if arr is of type \ref f64 or \ref c64, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_double       (bool *result, const af_array arr);

    /**
        \brief Check if an array is single precision type

        \param[out] result is true if arr is of type \ref f32 or \ref c32, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_single       (bool *result, const af_array arr);

    /**
        \brief Check if an array is real floating point type

        \param[out] result is true if arr is of type \ref f32 or \ref f64, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_realfloating (bool *result, const af_array arr);

    /**
        \brief Check if an array is floating precision type

        This is a combination of \ref af_is_realfloating and \ref af_is_complex

        \param[out] result is true if arr is of type \ref f32, \ref f64, \ref c32 or \ref c64, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_floating     (bool *result, const af_array arr);

    /**
        \brief Check if an array is integer type

        \param[out] result is true if arr is of integer types, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_integer      (bool *result, const af_array arr);

    /**
        \brief Check if an array is bool type

        \param[out] result is true if arr is of \ref b8 type, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_bool         (bool *result, const af_array arr);

#if AF_API_VERSION >= 34
    /**
        \brief Check if an array is sparse

        \param[out] result is true if arr is sparse, otherwise false
        \param[in] arr is the input array

        \returns error codes
    */
    AFAPI af_err af_is_sparse       (bool *result, const af_array arr);
#endif

#if AF_API_VERSION >= 35
    /**
        \brief Get first element from an array

        \param[out] output_value is the element requested
        \param[in] arr is the input array
        \return \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_get_scalar(void* output_value, const af_array arr);
#endif

    /**
        @}
    */

#ifdef __cplusplus
}
#endif
