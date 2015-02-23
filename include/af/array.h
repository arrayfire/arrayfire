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

    /// GPU array container
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
        array();
        array(const af_array handle);
        array(const array& in);

        array(dim_type d0, dtype ty = f32);
        array(dim_type d0, dim_type d1, dtype ty = f32);
        array(dim_type d0, dim_type d1, dim_type d2, dtype ty = f32);
        array(dim_type d0, dim_type d1, dim_type d2, dim_type d3, dtype ty = f32);
        array(const dim4& dims, dtype ty = f32);

        template<typename T>
            array(dim_type d0,
                  const T *pointer, af_source_t src=afHost, dim_type ngfor=0);


        template<typename T>
            array(dim_type d0, dim_type d1,
                  const T *pointer, af_source_t src=afHost, dim_type ngfor=0);


        template<typename T>
            array(dim_type d0, dim_type d1, dim_type d2,
                  const T *pointer, af_source_t src=afHost, dim_type ngfor=0);


        template<typename T>
            array(dim_type d0, dim_type d1, dim_type d2, dim_type d3,
                  const T *pointer, af_source_t src=afHost, dim_type ngfor=0);

        template<typename T>
            array(const dim4& dims,
                  const T *pointer, af_source_t src=afHost, dim_type ngfor=0);

        af_array get();
        af_array get() const;
        dim_type elements() const;

        template<typename T> T* host() const;
        void host(void *ptr) const;
        dtype type() const;

        // FIXME: Everything below this has not been implemented
        dim4 dims() const;
        dim_type dims(unsigned dim) const;
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

        void unlock() const;

        template<typename T> T scalar() const;
        template<typename T> T* device() const;

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
        ASSIGN(+=)
        ASSIGN(-=)
        ASSIGN(*=)
        ASSIGN(/=)

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

        OPERATOR(+ )
        OPERATOR(- )
        OPERATOR(* )
        OPERATOR(/ )
        OPERATOR(==)
        OPERATOR(!=)
        OPERATOR(< )
        OPERATOR(<=)
        OPERATOR(> )
        OPERATOR(>=)
        OPERATOR(&&)
        OPERATOR(||)
        OPERATOR(% )
        OPERATOR(& )
        OPERATOR(| )
        OPERATOR(^ )
        OPERATOR(<<)
        OPERATOR(>>)

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

        array operator -() const;
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

    BIN_OP(+ );
    BIN_OP(- );
    BIN_OP(* );
    BIN_OP(/ );
    BIN_OP(==);
    BIN_OP(!=);
    BIN_OP(< );
    BIN_OP(<=);
    BIN_OP(> );
    BIN_OP(>=);
    BIN_OP(&&);
    BIN_OP(||);
    BIN_OP(% );
    BIN_OP(& );
    BIN_OP(| );
    BIN_OP(^ );
    BIN_OP(<<);
    BIN_OP(>>);

#undef BIN_OP


    /// Evaluate an expression (nonblocking).
    inline array &eval(array &a) { a.eval(); return a; }
    inline void eval(array &a, array &b) { eval(a); b.eval(); }
    inline void eval(array &a, array &b, array &c) { eval(a, b); c.eval(); }
    inline void eval(array &a, array &b, array &c, array &d) { eval(a, b, c); d.eval(); }
    inline void eval(array &a, array &b, array &c, array &d, array &e) { eval(a, b, c, d); e.eval(); }
    inline void eval(array &a, array &b, array &c, array &d, array &e, array &f) { eval(a, b, c, d, e); f.eval(); }

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // Create af_array from memory
    AFAPI af_err af_create_array(af_array *arr, const void * const data, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Create af_array handle without initializing values
    AFAPI af_err af_create_handle(af_array *arr, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Deep copy an array to another
    AFAPI af_err af_copy_array(af_array *arr, const af_array in);

    // Copy data from an af_array to a C pointer.
    // Needs to used in conjunction with the two functions above
    AFAPI af_err af_get_data_ptr(void *data, const af_array arr);

    // Destroy af_array
    AFAPI af_err af_destroy_array(af_array arr);

    // weak copy array
    AFAPI af_err af_weak_copy(af_array *out, const af_array in);

    // Evaluate any expressions in the Array
    AFAPI af_err af_eval(af_array in);

#ifdef __cplusplus
}
#endif
