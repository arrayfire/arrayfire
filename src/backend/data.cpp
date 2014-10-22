#include <complex>
#include <af/dim4.hpp>
#include <af/array.h>
#include <af/util.h>
#include <copy.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <handle.hpp>
#include <random.hpp>
#include <math.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static af_array createHandle(af::dim4 d)
{
    return getHandle(*createEmptyArray<T>(d));
}

template<typename T>
static af_array createHandle(af::dim4 d, double val)
{
    return getHandle(*createValueArray<T>(d, scalar<T>(val)));
}

template<typename T>
static af_array createHandle(af::dim4 d, const T * const data)
{
    return getHandle(*createDataArray<T>(d, data));
}

template<typename T>
static void copyData(T *data, const af_array &arr)
{
    return copyData(data, getArray<T>(arr));
}

template<typename T>
static void destroyHandle(const af_array arr)
{
    destroyArray(getWritableArray<T>(arr));
}

af_err af_get_data_ptr(void *data, const af_array arr)
{
    af_err ret = AF_SUCCESS;

    try {
        af_dtype type;
        af_get_type(&type, arr);
        switch(type) {
        case f32:   copyData(static_cast<float    *>(data), arr);  break;
        case c32:   copyData(static_cast<cfloat   *>(data), arr);  break;
        case f64:   copyData(static_cast<double   *>(data), arr);  break;
        case c64:   copyData(static_cast<cdouble  *>(data), arr);  break;
        case b8:    copyData(static_cast<char     *>(data), arr);  break;
        case s32:   copyData(static_cast<int      *>(data), arr);  break;
        case u32:   copyData(static_cast<unsigned *>(data), arr);  break;
        case u8:    copyData(static_cast<uchar    *>(data), arr);  break;
        case s8:    copyData(static_cast<char     *>(data), arr);  break;
        default:    ret  = AF_ERR_RUNTIME;                         break;
        }
    }
    CATCHALL
        return ret;
}

//Strong Exception Guarantee
af_err af_create_array(af_array *result, const void * const data,
                       const unsigned ndims, const dim_type * const dims,
                       const af_dtype type)
{
    af_err ret = AF_ERR_ARG;
    af_array out;
    try {
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
        }
        switch(type) {
        case f32:   out = createHandle(d, static_cast<const float   *>(data)); break;
        case c32:   out = createHandle(d, static_cast<const cfloat  *>(data)); break;
        case f64:   out = createHandle(d, static_cast<const double  *>(data)); break;
        case c64:   out = createHandle(d, static_cast<const cdouble *>(data)); break;
        case b8:    out = createHandle(d, static_cast<const char    *>(data)); break;
        case s32:   out = createHandle(d, static_cast<const int     *>(data)); break;
        case u32:   out = createHandle(d, static_cast<const uint    *>(data)); break;
        case u8:    out = createHandle(d, static_cast<const uchar   *>(data)); break;
        case s8:    out = createHandle(d, static_cast<const char    *>(data)); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;    break;
        }
        std::swap(*result, out);
        ret = AF_SUCCESS;
    }
    CATCHALL
        return ret;
}

//Strong Exception Guarantee
af_err af_constant(af_array *result, const double value,
                   const unsigned ndims, const dim_type * const dims,
                   const af_dtype type)
{
    af_err ret = AF_ERR_ARG;
    af_array out;
    try {
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
        }
        switch(type) {
        case f32:   out = createHandle<float  >(d, value); break;
        case c32:   out = createHandle<cfloat >(d, value); break;
        case f64:   out = createHandle<double >(d, value); break;
        case c64:   out = createHandle<cdouble>(d, value); break;
        case b8:    out = createHandle<char   >(d, value); break;
        case s32:   out = createHandle<int    >(d, value); break;
        case u32:   out = createHandle<uint   >(d, value); break;
        case u8:    out = createHandle<uchar  >(d, value); break;
        case s8:    out = createHandle<char   >(d, value); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;    break;
        }
        std::swap(*result, out);
        ret = AF_SUCCESS;
    }
    CATCHALL
        return ret;
}

//Strong Exception Guarantee
af_err af_create_handle(af_array *result, const unsigned ndims, const dim_type * const dims,
                        const af_dtype type)
{
    af_err ret = AF_ERR_ARG;
    af_array out;
    try {
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
        }
        switch(type) {
        case f32:   out = createHandle<float  >(d); break;
        case c32:   out = createHandle<cfloat >(d); break;
        case f64:   out = createHandle<double >(d); break;
        case c64:   out = createHandle<cdouble>(d); break;
        case b8:    out = createHandle<char   >(d); break;
        case s32:   out = createHandle<int    >(d); break;
        case u32:   out = createHandle<uint   >(d); break;
        case u8:    out = createHandle<uchar  >(d); break;
        case s8:    out = createHandle<char   >(d); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;    break;
        }
        std::swap(*result, out);
        ret = AF_SUCCESS;
    }
    CATCHALL
        return ret;
}

template<typename T>
static inline af_array randn_(const af::dim4 &dims)
{
    return getHandle(*randn<T>(dims));
}

template<typename T>
static inline af_array randu_(const af::dim4 &dims)
{
    return getHandle(*randu<T>(dims));
}

af_err af_randu(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type)
{
    af_err ret = AF_SUCCESS;
    af_array result;
    try {
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
            if(d[i] < 1) {
                return AF_ERR_ARG;
            }
        }
        switch(type) {
            case f32:   result = randu_<float  >(d);    break;
            case c32:   result = randu_<cfloat >(d);    break;
            case f64:   result = randu_<double >(d);    break;
            case c64:   result = randu_<cdouble>(d);    break;
            case s32:   result = randu_<int    >(d);    break;
            case u32:   result = randu_<uint   >(d);    break;
            case u8:    result = randu_<uchar  >(d);    break;
            // Removed because of bool type. Functions implementations exist.
            //case s8:    result = randu_<char   >(d);    break;
            //case b8:    result = randu_<char   >(d);    break;
            default:    ret    = AF_ERR_NOT_SUPPORTED; break;
        }
        if(ret == AF_SUCCESS)
            std::swap(*out, result);
    }
    CATCHALL
    return ret;
}

af_err af_randn(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type)
{
    af_err ret = AF_SUCCESS;
    af_array result;
    try {
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
            if(d[i] < 1) {
                return AF_ERR_ARG;
            }
        }
        switch(type) {
            case f32:   result = randn_<float  >(d);    break;
            case c32:   result = randn_<cfloat >(d);    break;
            case f64:   result = randn_<double >(d);    break;
            case c64:   result = randn_<cdouble>(d);    break;
            default:    ret    = AF_ERR_NOT_SUPPORTED; break;
        }
        if(ret == AF_SUCCESS)
            std::swap(*out, result);
    }
    CATCHALL
    return ret;
}

af_err af_destroy_array(af_array arr)
{
    af_err ret = AF_ERR_ARG;
    try {
        af_dtype type;
        af_get_type(&type, arr);

        switch(type) {
        case f32:   destroyHandle<float   >(arr); break;
        case c32:   destroyHandle<cfloat  >(arr); break;
        case f64:   destroyHandle<double  >(arr); break;
        case c64:   destroyHandle<cdouble >(arr); break;
        case b8:    destroyHandle<char    >(arr); break;
        case s32:   destroyHandle<int     >(arr); break;
        case u32:   destroyHandle<uint    >(arr); break;
        case u8:    destroyHandle<uchar   >(arr); break;
        case s8:    destroyHandle<char    >(arr); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;    break;
        }

        ret = AF_SUCCESS;
    }
    CATCHALL
        return ret;
}
