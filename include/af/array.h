#pragma once
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>


#ifdef __cplusplus
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
        bool     isRef;
        af_seq    s[4];

        array(af_array in, af_seq *seqs);

    public:
        array();
        array(const af_array handle);
        array(const array& in);

        array(dim_type d0, af_dtype ty = f32);
        array(dim_type d0, dim_type d1, af_dtype ty = f32);
        array(dim_type d0, dim_type d1, dim_type d2, af_dtype ty = f32);
        array(dim_type d0, dim_type d1, dim_type d2, dim_type d3, af_dtype ty = f32);
        array(const dim4& dims, af_dtype ty = f32);

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
        af_dtype type() const;

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
        void eval();
        void unlock() const;

        template<typename T> T scalar() const;
        template<typename T> T* device() const;

        template<typename T> static T* alloc(size_t elements);
        static void *alloc(size_t elements, af_dtype type);

        template<typename T> static T* pinned(size_t elements);
        static void *pinned(size_t elements, af_dtype type);

        static void free(const void *);

        array operator()(const af_seq& s0, const af_seq& s1=span, const af_seq& s2=span, const af_seq& s3=span);

        array as(af_dtype type) const;

        ~array();

        array& operator= (const array &a);  ///< array assignment
        array& operator= (const double &value);
        array& operator= (const af_cfloat &value);
        array& operator= (const af_cdouble &value);

#define SELF(op)                                                            \
        array& operator op(const array &a);                                  \
        array& operator op(const double &a);                                 \
        array& operator op(const af_cdouble &a);                             \
        array& operator op(const af_cfloat &a);                              \

        SELF(+=)
        SELF(-=)
        SELF(*=)
        SELF(/=)


#define BIN(op)                                                             \
        array operator op(const array&) const;                              \
        array operator op(const double&) const;                             \
        array operator op(const af_cfloat&) const;                          \
        array operator op(const af_cdouble&) const;                         \
        AFAPI friend array operator op(const double&, const array&);        \
        AFAPI friend array operator op(const af_cfloat&, const array&);     \
        AFAPI friend array operator op(const af_cdouble&, const array&);    \

        BIN(+)
        BIN(-)
        BIN(*)
        BIN(/)

/*
// FIXME
//#define LOGIC(op)                                                           \
//        array operator op(const array&) const;                              \
//        array operator op##op(const array&) const;                          \
//        array operator op##op(const bool&) const;                           \
//        array operator op##op(const int&) const;                            \
//        array operator op##op(const unsigned&) const;                       \
//        array operator op##op(const double&) const;                         \
//        array operator op##op(const af_cfloat&) const;                      \
//        array operator op##op(const af_cdouble&) const;                     \
//        AFAPI friend array operator op##op(const bool&, const array&);      \
//        AFAPI friend array operator op##op(const int&, const array&);       \
//        AFAPI friend array operator op##op(const unsigned&, const array&);  \
//        AFAPI friend array operator op##op(const af_cfloat&, const array&); \
//        AFAPI friend array operator op##op(const af_cdouble&, const array&);\
*/


#define COMP(op)                                                           \
        array operator op(const array&) const;                             \
        array operator op(const bool&) const;                              \
        array operator op(const int&) const;                               \
        array operator op(const double&) const;                            \
        array operator op(const af_cfloat&) const;                         \
        array operator op(const af_cdouble&) const;                        \
        AFAPI friend array operator op(const bool&, const array&);         \
        AFAPI friend array operator op(const int&, const array&);          \
        AFAPI friend array operator op(const double&, const array&);       \
        AFAPI friend array operator op(const af_cfloat&, const array&);    \
        AFAPI friend array operator op(const af_cdouble&, const array&);   \

        COMP(==)
        COMP(!=)
        COMP(< )
        COMP(<=)
        COMP(> )
        COMP(>=)

#undef SELF
#undef BIN
#undef COMP
    };
    // end of class array

#define BIN(op)                                                         \
    AFAPI array operator op(const double&, const array&);               \
    AFAPI array operator op(const af_cfloat&, const array&);            \
    AFAPI array operator op(const af_cdouble&, const array&);           \

    BIN(+)
    BIN(-)
    BIN(*)
    BIN(/)

/*
//#define LOGIC(op)                                                       \
//        AFAPI array& operator op##op(const bool&, const array&);         \
//        AFAPI array& operator op##op(const int&, const array&);          \
//        AFAPI array& operator op##op(const unsigned&, const array&);     \
//        AFAPI array& operator op##op(const af_cfloat&, const array&);    \
//        AFAPI array& operator op##op(const af_cdouble&, const array&);   \
 */


#define COMP(op)                                                        \
    AFAPI array operator op(const bool&, const array&);                 \
    AFAPI array operator op(const int&, const array&);                  \
    AFAPI array operator op(const double&, const array&);               \
    AFAPI array operator op(const af_cfloat&, const array&);            \
    AFAPI array operator op(const af_cdouble&, const array&);           \

    COMP(==)
    COMP(!=)
    COMP(< )
    COMP(<=)
    COMP(> )
    COMP(>=)

#undef SELF
#undef BIN
#undef LOGIC
#undef COMP

    AFAPI array constant(double val, const dim4 &dims, af_dtype ty=f32);
    AFAPI array constant(af_cdouble val, const dim4 &dims);
    AFAPI array constant(af_cfloat val, const dim4 &dims);

    AFAPI array constant(double val, const dim_type d0, af_dtype ty=f32);
    AFAPI array constant(double val, const dim_type d0,
                         const dim_type d1, af_dtype ty=f32);
    AFAPI array constant(double val, const dim_type d0,
                         const dim_type d1, const dim_type d2, af_dtype ty=f32);
    AFAPI array constant(double val, const dim_type d0,
                         const dim_type d1, const dim_type d2,
                         const dim_type d3, af_dtype ty=f32);


    AFAPI array randu(const dim4 &dims, af_dtype ty=f32);
    AFAPI array randu(const dim_type d0, af_dtype ty=f32);
    AFAPI array randu(const dim_type d0,
                      const dim_type d1, af_dtype ty=f32);
    AFAPI array randu(const dim_type d0,
                      const dim_type d1, const dim_type d2, af_dtype ty=f32);
    AFAPI array randu(const dim_type d0,
                      const dim_type d1, const dim_type d2,
                      const dim_type d3, af_dtype ty=f32);


    AFAPI array randn(const dim4 &dims, af_dtype ty=f32);
    AFAPI array randn(const dim_type d0, af_dtype ty=f32);
    AFAPI array randn(const dim_type d0,
                      const dim_type d1, af_dtype ty=f32);
    AFAPI array randn(const dim_type d0,
                      const dim_type d1, const dim_type d2, af_dtype ty=f32);
    AFAPI array randn(const dim_type d0,
                      const dim_type d1, const dim_type d2,
                      const dim_type d3, af_dtype ty=f32);

    AFAPI array iota(const dim4 &dims, const unsigned rep = 3, af_dtype ty=f32);
    AFAPI array iota(const dim_type d0, const unsigned rep = 0, af_dtype ty=f32);
    AFAPI array iota(const dim_type d0, const dim_type d1,
                     const unsigned rep = 1, af_dtype ty=f32);
    AFAPI array iota(const dim_type d0, const dim_type d1, const dim_type d2,
                     const unsigned rep = 2, af_dtype ty=f32);
    AFAPI array iota(const dim_type d0, const dim_type d1, const dim_type d2,
                     const dim_type d3, const unsigned rep = 3, af_dtype ty=f32);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // Create af_array from a constant value
    AFAPI af_err af_constant(af_array *arr, const double val, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    AFAPI af_err af_constant_c64(af_array *arr, const void* val,
                                 const unsigned ndims, const dim_type * const dims);

    AFAPI af_err af_constant_c32(af_array *arr, const void* val,
                                 const unsigned ndims, const dim_type * const dims);

    // Create sequence array
    AFAPI af_err af_iota(af_array *arr, const unsigned ndims, const dim_type * const dims,
                         const unsigned rep, const af_dtype type);

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

    // Generate Random Numbers using uniform distribution
    AFAPI af_err af_randu(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Generate Random Numbers using normal distribution
    AFAPI af_err af_randn(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // weak copy array
    AFAPI af_err af_weak_copy(af_array *out, const af_array in);

#ifdef __cplusplus
}
#endif
