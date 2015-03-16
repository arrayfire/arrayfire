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
#include <af/dim4.hpp>
#include <af/seq.h>
#include <af/traits.hpp>


#ifdef __cplusplus
#include <vector>
namespace af
{

    class dim4;

    /// Specify which address-space pointer belongs
    typedef enum {
        afDevice, ///< Device-memory pointer
        afHost    ///< Host-memory pointer
    } af_source_t;

    ///
    /// \brief A multi dimensional data container
    ///
    class AFAPI array {

    private:
        af_array   arr;

        //FIXME: Put the following in a different class
        const array   *parent;
        bool     isRef;
        std::vector<seq> s;
        void getSeq(af_seq* afs) const;
        array(af_array in, const array *par, seq *seqs);
        void set(af_array tmp);
        void set(af_array tmp) const;
        //END FIXME

    public:
        /**
            \ingroup construct_mat
            @{
        */
        /**
            Create non-dimensioned array (no data, undefined size)

            \code
            array A, B, C;   // creates three arrays called A, B and C
            \endcode
        */
        array();
        /**
            Creates an array from an \ref af_array handle
            \param handle the af_array object.
         */
        array(const af_array handle);
        /**
            Creates an arra
            TODO: Copy or reference semantics?
            \param in The input \ref array
         */
        array(const array& in);

        /**
            Allocate a one-dimensional array of a specified size with undefined contents

            Declare a two-dimensional array by passing the
            number of rows and the number of columns as the first two parameters.
            The (optional) third parameter is the type of the array. The default type is f32
            or 4-byte single-precision floating-point numbers.

            \code
            // allocate space for an array with 10 rows
            array A(10);          // type is the default f32

            // allocate space for a column vector with 100 rows
            array A(100, f64);    // f64 = double precision
            \endcode

            \param[in] dim0 number of columns in the array
            \param[in] ty optional label describing the data type
                       (default is f32)

        */
        array(dim_type dim0, dtype ty = f32);

        /**
            Allocate a two-dimensional array of a specified size with undefined contents

            Declare a two-dimensional array by passing the
            number of rows and the number of columns as the first two parameters.
            The (optional) third parameter is the type of the array. The default type is f32
            or 4-byte single-precision floating-point numbers.

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
        array(dim_type dim0, dim_type dim1, dtype ty = f32);

        /**
            Allocate a three-dimensional (3D) array of a specified size with undefined contents

            This is useful to quickly declare a three-dimensional array by passing
            the size as the first three parameters. The (optional) fourth parameter
            is the type of the array. The default type is f32 or 4-byte
            single-precision floating point numbers.

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
        array(dim_type dim0, dim_type dim1, dim_type dim2, dtype ty = f32);

        /**
            Allocate a four-dimensional (4D) array of a specified size with undefined contents

            This is useful to quickly declare a four-dimensional array by passing the
            size as the first four parameters. The (optional) fifth parameter is the
            type of the array. The default type is f32 or 4-byte floating point numbers.

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
        array(dim_type dim0, dim_type dim1, dim_type dim2, dim_type dim3, dtype ty = f32);

        /**
            Allocate an array of a specified size with undefined contents

            This can be useful when the dimensions of the array are calculated
            somewhere else within the code. The first parameter specifies the
            size of the array via dim4(). The second parameter is
            the type of the array. The default type is f32 or 4-byte
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
        array(const dim4& dims, dtype ty = f32);

        /**
            Create a column vector on the device using a host/device pointer

            This function can be used to transfer data from a host or device pointer
            to an array object on the device with one column. The type of the array
            is automatically matched to the type of the data.

            Depending on the specified size of the column vector, the data will be
            copied partially or completely. However, the user needs to be careful to
            ensure that the array size is not larger than the number of elements in
            the input buffer.

            \param[in] dim0    number of elements in the column vector
            \param[in] pointer pointer (points to a buffer on the host/device)
            \param[in] src     source of the data (default is afHost, can also be afDevice)
            \param[in] ngfor   number of gfor tiles (default is ZERO)

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
            array(dim_type dim0,
                  const T *pointer, af_source_t src=afHost, dim_type ngfor=0);


        /**
            Create a 2D array on the device using a host/device pointer

            This function copies data from the location specified by the pointer
            to a 2D array on the device. The data is arranged in "column-major"
            format (similar to that used by FORTRAN).

            Note that this is an synchronous copy. The elements are not actually
            filled until this array is evaluated or used in the evaluation of some
            other expression that uses this array object.

            \param[in] dim0    number of rows
            \param[in] dim1    number of columns
            \param[in] pointer pointer (points to a buffer on the host/device)
            \param[in] src     source of the data (default is afHost, can also be afDevice)
            \param[in] ngfor   number of gfor tiles (default is ZERO)

            \code
            int h_buffer[] = {0, 1, 2, 3, 4, 5};  // host array
            array A(2, 3, h_buffer);              // copy host data to device
            \endcode

            \image html 2dArray.png
        */
        template<typename T>
            array(dim_type dim0, dim_type dim1,
                  const T *pointer, af_source_t src=afHost, dim_type ngfor=0);


        /**
            Create a 3D array on the device using a host/device pointer

            This function copies data from the location specified by the pointer
            to a 3D array on the device. The data is arranged in "column-major"
            format (similar to that used by FORTRAN).

            \param[in] dim0    first dimension
            \param[in] dim1    second dimension
            \param[in] dim2    third dimension
            \param[in] pointer pointer (points to a buffer on the host/device)
            \param[in] src     source of the data (default is \ref afHost, can also be \ref afDevice)
            \param[in] ngfor   number of gfor tiles (default is ZERO)

            \code
            int h_buffer[] = {0, 1, 2, 3, 4, 5, 6, 7, 8
                              9, 0, 1, 2, 3, 4, 5, 6, 7};   // host array

            array A(3, 3, 2,  h_buffer);   // copy host data to 3D device array
            \endcode

            \image html 3dArray.png
        */
        template<typename T>
            array(dim_type dim0, dim_type dim1, dim_type dim2,
                  const T *pointer, af_source_t src=afHost, dim_type ngfor=0);


        /**
            Create a 4D array on the device using a host/device pointer

            This function copies data from the location specified by the pointer
            to a 4D array on the device. The data is arranged in "column-major"
            format (similar to that used by FORTRAN).

            \param[in] dim0    first dimension
            \param[in] dim1    second dimension
            \param[in] dim2    third dimension
            \param[in] dim3    fourth dimension
            \param[in] pointer pointer (points to a buffer on the host/device)
            \param[in] src     source of the data (default is afHost, can also be afDevice)
            \param[in] ngfor   number of gfor tiles (default is ZERO)

            \code
            int h_buffer[] = {0, 1, 2, 3,
                              4, 5, 6, 7,
                              8, 9, 0, 1,
                              2, 3, 4, 5};   // host array with 16 elements

            array A(2, 2, 2, 2, h_buffer);   // copy host data to 4D device array
            \endcode
        */
        template<typename T>
            array(dim_type dim0, dim_type dim1, dim_type dim2, dim_type dim3,
                  const T *pointer, af_source_t src=afHost, dim_type ngfor=0);

        /**
            Create an array of specified size on the device using a host/device pointer

            This function copies data from the location specified by the pointer
            to a 1D/2D/3D/4D array on the device. The data is arranged in "column-major"
            format (similar to that used by FORTRAN).

            \param[in] dims    vector data type containing the dimension of the array
            \param[in] pointer pointer (points to a buffer on the host/device)
            \param[in] src     source of the data (default is afHost, can also be afDevice)
            \param[in] ngfor   number of gfor tiles (default is ZERO)

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
            array(const dim4& dims,
                  const T *pointer, af_source_t src=afHost, dim_type ngfor=0);

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
        dim_type elements() const;

        /**
           Copy array data to host and return host pointer
        */
        template<typename T> T* host() const;

        /**
           Copy array data to existing host pointer
        */
        void host(void *ptr) const;

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
        dim_type dims(unsigned dim) const;

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
           \brief Returns true if only one of the array dimensions has more than one elment
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
           @}
        */
        template<typename T> T scalar() const;


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

        void unlock() const;


        // INDEXING
    public:
        // Single arguments
        array operator()(const array& idx) const;
        array operator()(const seq& s0) const;
        array operator()(const int& s0) const
                        { return this->operator()(seq(s0, s0)); }

        // Two arguments
        array operator()(const seq& s0, const seq& s1) const;

        array operator()(const int& s0, const int& s1) const
                        { return this->operator()(seq(s0, s0), seq(s1, s1)); }

        array operator()(const int& s0, const seq& s1) const
                        { return this->operator()(seq(s0, s0), s1); }

        array operator()(const seq& s0, const int& s1) const
                        { return this->operator()(s0, seq(s1, s1)); }

        // Three arguments
        array operator()(const seq& s0, const seq& s1, const seq& s2) const;

        array operator()(const int& s0, const int& s1, const int& s2) const
                        { return this->operator()(seq(s0, s0), seq(s1, s1), seq(s2, s2)); }

        array operator()(const int& s0, const seq& s1, const seq& s2) const
                        { return this->operator()(seq(s0, s0), s1, s2); }

        array operator()(const int& s0, const int& s1, const seq& s2) const
                        { return this->operator()(seq(s0, s0), seq(s1, s1), s2); }

        array operator()(const seq& s0, const int& s1, const seq& s2) const
        { return this->operator()(s0, seq(s1, s1), s2); }

        array operator()(const seq& s0, const int& s1, const int& s2) const
        { return this->operator()(s0, seq(s1, s1), seq(s2, s2)); }

        array operator()(const seq& s0, const seq& s1, const int& s2) const
        { return this->operator()(s0, s1, seq(s2, s2)); }

        // Four arguments
        array operator()(const seq& s0, const seq& s1, const seq& s2, const seq& s3) const;
        array operator()(const seq& s0, const seq& s1, const seq& s2, const int& s3) const
        { return this->operator()(s0, s1, s2, seq(s3, s3)); }
        array operator()(const seq& s0, const seq& s1, const int& s2, const seq& s3) const
        { return this->operator()(s0, s1, seq(s2, s2), s3); }
        array operator()(const seq& s0, const seq& s1, const int& s2, const int& s3) const
        { return this->operator()(s0, s1, seq(s2, s2), seq(s3, s3)); }
        array operator()(const seq& s0, const int& s1, const seq& s2, const seq& s3) const
        { return this->operator()(s0, seq(s1, s1), s2, s3); }
        array operator()(const seq& s0, const int& s1, const seq& s2, const int& s3) const
        { return this->operator()(s0, seq(s1, s1), s2, seq(s3, s3)); }
        array operator()(const seq& s0, const int& s1, const int& s2, const seq& s3) const
        { return this->operator()(s0, seq(s1, s1), seq(s2, s2), s3); }
        array operator()(const seq& s0, const int& s1, const int& s2, const int& s3) const
        { return this->operator()(s0, seq(s1, s1), seq(s2, s2), seq(s3, s3)); }

        array operator()(const int& s0, const seq& s1, const seq& s2, const int& s3) const
        { return this->operator()(seq(s0, s0), s1, s2, seq(s3, s3)); }
        array operator()(const int& s0, const seq& s1, const int& s2, const seq& s3) const
        { return this->operator()(seq(s0, s0), s1, seq(s2, s2), s3); }
        array operator()(const int& s0, const seq& s1, const int& s2, const int& s3) const
        { return this->operator()(seq(s0, s0), s1, seq(s2, s2), seq(s3, s3)); }
        array operator()(const int& s0, const int& s1, const seq& s2, const seq& s3) const
        { return this->operator()(seq(s0, s0), seq(s1, s1), s2, s3); }
        array operator()(const int& s0, const int& s1, const seq& s2, const int& s3) const
        { return this->operator()(seq(s0, s0), seq(s1, s1), s2, seq(s3, s3)); }
        array operator()(const int& s0, const int& s1, const int& s2, const seq& s3) const
        { return this->operator()(seq(s0, s0), seq(s1, s1), seq(s2, s2), s3); }
        array operator()(const int& s0, const int& s1, const int& s2, const int& s3) const
        { return this->operator()(seq(s0, s0), seq(s1, s1), seq(s2, s2), seq(s3, s3)); }

        array row(int index) const;

        array col(int index) const;

        array slice(int index) const;

        array rows(int first, int last) const;

        array cols(int first, int last) const;

        array slices(int first, int last) const;

        array as(dtype type) const;

        ~array();

        // Transpose and Conjugate Tranpose
        array T() const;
        array H() const;

#define ASSIGN(OP)                                          \
        array& operator OP(const array &a);                 \
        array& operator OP(const double &a);                \
        array& operator OP(const cdouble &a);               \
        array& operator OP(const cfloat &a);                \
        array& operator OP(const float &a);                 \
        array& operator OP(const int &a);                   \
        array& operator OP(const unsigned &a);              \
        array& operator OP(const bool &a);                  \
        array& operator OP(const char &a);                  \
        array& operator OP(const unsigned char &a);         \
        array& operator OP(const long  &a);                 \
        array& operator OP(const unsigned long &a);         \
        array& operator OP(const long long  &a);            \
        array& operator OP(const unsigned long long &a);    \

        ASSIGN(= )

        /**
           \ingroup arith_func_add
           @{
        */
        ASSIGN(+=)
        /**
           @}
        */

        /**
           \ingroup arith_func_sub
           @{
        */
        ASSIGN(-=)
        /**
           @}
        */

        /**
           \ingroup arith_func_mul
           @{
        */
        ASSIGN(*=)
        /**
           @}
        */

        /**
           \ingroup arith_func_div
           @{
        */
        ASSIGN(/=)
        /**
           @}
        */

#undef ASSIGN

#define OPERATOR(op)                                                \
            array operator op(const array &a) const;                \
            array operator op(const double &a) const;               \
            array operator op(const cdouble &a) const;              \
            array operator op(const cfloat &a) const;               \
            array operator op(const float &a) const;                \
            array operator op(const int &a) const;                  \
            array operator op(const unsigned &a) const;             \
            array operator op(const bool &a) const;                 \
            array operator op(const char &a) const;                 \
            array operator op(const unsigned char &a) const;        \
            array operator op(const long  &a) const;                \
            array operator op(const unsigned long &a) const;        \
            array operator op(const long long  &a) const;           \
            array operator op(const unsigned long long &a) const;   \

        /**
           \ingroup arith_func_add
           @{
        */
        OPERATOR(+ )
        /**
           @}
        */

        /**
           \ingroup arith_func_sub
           @{
        */
        OPERATOR(- )
        /**
           @}
        */

        /**
           \ingroup arith_func_mul
           @{
        */
        OPERATOR(* )
        /**
           @}
        */

        /**
           \ingroup arith_func_div
           @{
        */
        OPERATOR(/ )
        /**
           @}
        */

        /**
           \ingroup logic_func_eq
           @{
        */
        OPERATOR(==)
        /**
           @}
        */

        /**
           \ingroup logic_func_neq
           @{
        */
        OPERATOR(!=)
        /**
           @}
        */

        /**
           \ingroup logic_func_lt
           @{
        */
        OPERATOR(< )
        /**
           @}
        */

        /**
           \ingroup logic_func_le
           @{
        */
        OPERATOR(<=)
        /**
           @}
        */

        /**
           \ingroup logic_func_gt
           @{
        */
        OPERATOR(> )
        /**
           @}
        */

        /**
           \ingroup logic_func_ge
           @{
        */
        OPERATOR(>=)
        /**
           @}
        */

        /**
           \ingroup logic_func_and
           @{
        */
        OPERATOR(&&)
        /**
           @}
        */

        /**
           \ingroup logic_func_or
        */
        OPERATOR(||)
        /**
           @}
        */

        /**
           \ingroup numeric_func_rem
           @{
        */
        OPERATOR(% )
        /**
           @}
        */

        /**
           \ingroup logic_func_bitand
           @{
        */
        OPERATOR(& )
        /**
           @}
        */

        /**
           \ingroup logic_func_bitor
           @{
        */
        OPERATOR(| )
        /**
           @}
        */

        /**
           \ingroup logic_func_bitxor
           @{
        */
        OPERATOR(^ )
        /**
           @}
        */

        /**
           \ingroup arith_func_shiftl
           @{
        */
        OPERATOR(<<)
        /**
           @}
        */

        /**
           \ingroup logic_func_shiftr
           @{
        */
        OPERATOR(>>)
        /**
           @}
        */

#undef OPERATOR

#define FRIEND_OP(op)                                                   \
        AFAPI friend array operator op(const bool&, const array&);      \
        AFAPI friend array operator op(const int&, const array&);       \
        AFAPI friend array operator op(const unsigned&, const array&);  \
        AFAPI friend array operator op(const char&, const array&);      \
        AFAPI friend array operator op(const unsigned char&, const array&); \
        AFAPI friend array operator op(const long&, const array&);      \
        AFAPI friend array operator op(const unsigned long&, const array&); \
        AFAPI friend array operator op(const long long&, const array&); \
        AFAPI friend array operator op(const unsigned long long&, const array&); \
        AFAPI friend array operator op(const double&, const array&);    \
        AFAPI friend array operator op(const float&, const array&);     \
        AFAPI friend array operator op(const cfloat&, const array&);    \
        AFAPI friend array operator op(const cdouble&, const array&);   \

        FRIEND_OP(+ )
        FRIEND_OP(- )
        FRIEND_OP(* )
        FRIEND_OP(/ )
        FRIEND_OP(==)
        FRIEND_OP(!=)
        FRIEND_OP(< )
        FRIEND_OP(<=)
        FRIEND_OP(> )
        FRIEND_OP(>=)
        FRIEND_OP(&&)
        FRIEND_OP(||)
        FRIEND_OP(% )
        FRIEND_OP(& )
        FRIEND_OP(| )
        FRIEND_OP(^ )
        FRIEND_OP(<<)
        FRIEND_OP(>>)

#undef FRIEND_OP

        /**
           \ingroup arith_func_neg
        */
        array operator -() const;

        /**
           \ingroup arith_func_not
        */
        array operator !() const;
    };
    // end of class array

#define BIN_OP(op)                                                      \
    AFAPI array operator op(const bool&, const array&);                 \
    AFAPI array operator op(const int&, const array&);                  \
    AFAPI array operator op(const unsigned&, const array&);             \
    AFAPI array operator op(const char&, const array&);                 \
    AFAPI array operator op(const unsigned char&, const array&);        \
    AFAPI array operator op(const long&, const array&);                 \
    AFAPI array operator op(const unsigned long&, const array&);        \
    AFAPI array operator op(const long long&, const array&);            \
    AFAPI array operator op(const unsigned long long&, const array&);   \
    AFAPI array operator op(const double&, const array&);               \
    AFAPI array operator op(const float&, const array&);                \
    AFAPI array operator op(const cfloat&, const array&);               \
    AFAPI array operator op(const cdouble&, const array&);              \

    /**
       \ingroup arith_func_add
       @{
    */
    BIN_OP(+ )
    /**
       @}
    */

    /**
       \ingroup arith_func_sub
       @{
    */
    BIN_OP(- )
    /**
       @}
    */

    /**
       \ingroup arith_func_mul
       @{
    */
    BIN_OP(* )
    /**
       @}
    */

    /**
       \ingroup arith_func_div
       @{
    */
    BIN_OP(/ )
    /**
       @}
    */

    /**
       \ingroup logic_func_eq
       @{
    */
    BIN_OP(==)
    /**
       @}
    */

    /**
       \ingroup logic_func_neq
       @{
    */
    BIN_OP(!=)
    /**
       @}
    */

    /**
       \ingroup logic_func_lt
       @{
    */
    BIN_OP(< )
    /**
       @}
    */

    /**
       \ingroup logic_func_le
       @{
    */
    BIN_OP(<=)
    /**
       @}
    */

    /**
       \ingroup logic_func_gt
       @{
    */
    BIN_OP(> )
    /**
       @}
    */

    /**
       \ingroup logic_func_ge
       @{
    */
    BIN_OP(>=)
    /**
       @}
    */

    /**
       \ingroup logic_func_and
       @{
    */
    BIN_OP(&&)
    /**
       @}
    */

    /**
       \ingroup logic_func_or
       @{
    */
    BIN_OP(||)
    /**
       @}
    */

    /**
       \ingroup numeric_func_rem
       @{
    */
    BIN_OP(% )
    /**
       @}
    */

    /**
       \ingroup logic_func_bitand
       @{
    */
    BIN_OP(& )
    /**
       @}
    */

    /**
       \ingroup logic_func_bitor
       @{
    */
    BIN_OP(| )
    /**
       @}
    */

    /**
       \ingroup logic_func_bitxor
       @{
    */
    BIN_OP(^ )
    /**
       @}
    */

    /**
       \ingroup arith_func_shiftl
       @{
    */
    BIN_OP(<<)
    /**
       @}
    */

    /**
       \ingroup arith_func_shiftr
       @{
    */
    BIN_OP(>>)
    /**
       @}
    */

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

       \param[out]  arr The pointer to the retured object.
       \param[in]   data The data which will be loaded into the array
       \param[in]   ndims The number of dimensions read from the \p dims parameter
       \param[in]   dims A C pointer with \p ndims elements. Each value represents the size of that dimension
       \param[in]   type The type of the \ref af_array object

       \returns \ref AF_SUCCESS if the operation was a success
    */
    AFAPI af_err af_create_array(af_array *arr, const void * const data, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    /**
       Create af_array handle

       \param[out]  arr The pointer to the retured object.
       \param[in]   ndims The number of dimensions read from the \p dims parameter
       \param[in]   dims A C pointer with \p ndims elements. Each value represents the size of that dimension
       \param[in]   type The type of the \ref af_array object

       \returns \ref AF_SUCCESS if the operation was a success
    */
    AFAPI af_err af_create_handle(af_array *arr, const unsigned ndims, const dim_type * const dims, const af_dtype type);

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
       Copy data from an af_array to a C pointer.

       Needs to used in conjunction with the two functions above
    */
    AFAPI af_err af_get_data_ptr(void *data, const af_array arr);

    /**
       \brief Destroy af_array
    */
    AFAPI af_err af_destroy_array(af_array arr);

    /**
       weak copy array
    */
    AFAPI af_err af_weak_copy(af_array *out, const af_array in);

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
