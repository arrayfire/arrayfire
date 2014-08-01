#include <complex>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/array.h>
#include <af/defines.h>
#include <generator.hpp>
#include <copy.hpp>
#include <helper.h>
#include <backend.h>
using af::dim4;

af_err af_get_data_ptr(void *data, const af_array arr)
{
    af_err ret = AF_SUCCESS;

    try {
        af_dtype type;
        af_get_type(&type, arr);
        switch(type) {
        case f32:   copyData(static_cast<float *>(data), arr);                   break;
        case c32:   copyData(static_cast<cfloat *>(data), arr);                  break;
        case f64:   copyData(static_cast<double *>(data), arr);                  break;
        case c64:   copyData(static_cast<cdouble *>(data), arr);                 break;
        case b8:    copyData(static_cast<char *>(data), arr);                    break;
        case s32:   copyData(static_cast<int *>(data), arr);                     break;
        case u32:   copyData(static_cast<unsigned *>(data), arr);                break;
        case u8:    copyData(static_cast<unsigned char *>(data), arr);           break;
        case s8:    copyData(static_cast<char *>(data), arr);                    break;
        default:    ret  = AF_ERR_RUNTIME;                         break;
        }
    }
    CATCHALL
        return ret;
}

//Strong Exception Guarantee
af_err af_create_array(af_array *result, const void * const data,
                       const unsigned ndims, const long * const dims,
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
        case f32:   out = createArrayHandle(d, static_cast<const float   *>(data)); break;
        case c32:   out = createArrayHandle(d, static_cast<const cfloat  *>(data)); break;
        case f64:   out = createArrayHandle(d, static_cast<const double  *>(data)); break;
        case c64:   out = createArrayHandle(d, static_cast<const cdouble *>(data)); break;
        case b8:    out = createArrayHandle(d, static_cast<const char    *>(data)); break;
        case s32:   out = createArrayHandle(d, static_cast<const int     *>(data)); break;
        case u32:   out = createArrayHandle(d, static_cast<const unsigned int  *>(data)); break;
        case u8:    out = createArrayHandle(d, static_cast<const unsigned char *>(data)); break;
        case s8:    out = createArrayHandle(d, static_cast<const char *>(data)); break;
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
                   const unsigned ndims, const long * const dims,
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
        case f32:   out = createArrayHandle<float  >(d, value); break;
        case c32:   out = createArrayHandle<cfloat >(d, value); break;
        case f64:   out = createArrayHandle<double >(d, value); break;
        case c64:   out = createArrayHandle<cdouble>(d, value); break;
        case b8:    out = createArrayHandle<char   >(d, value); break;
        case s32:   out = createArrayHandle<int    >(d, value); break;
        case u32:   out = createArrayHandle<unsigned int >(d, value); break;
        case u8:    out = createArrayHandle<unsigned char>(d, value); break;
        case s8:    out = createArrayHandle<char>(d, value); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;    break;
        }
        std::swap(*result, out);
        ret = AF_SUCCESS;
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
        case f32:   destroyArrayHandle<float   >(arr); break;
        case c32:   destroyArrayHandle<cfloat  >(arr); break;
        case f64:   destroyArrayHandle<double  >(arr); break;
        case c64:   destroyArrayHandle<cdouble >(arr); break;
        case b8:    destroyArrayHandle<char    >(arr); break;
        case s32:   destroyArrayHandle<int     >(arr); break;
        case u32:   destroyArrayHandle<unsigned int  >(arr); break;
        case u8:    destroyArrayHandle<unsigned char >(arr); break;
        case s8:    destroyArrayHandle<char >(arr); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;    break;
        }
        ret = AF_SUCCESS;
    }
    CATCHALL
        return ret;
}
