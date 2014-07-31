#include <complex>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/generator.h>
#include <af/array.h>
#include <af/defines.h>
#include <generator.hpp>
#include <helper.h>
#include <backend.h>

using af::dim4;

//Strong Exception Guarantee
af_err af_create_array(af_array *result,
                       const unsigned ndims, const long * const dims,
                       af_dtype type)
{
    af_err ret = AF_ERR_ARG;
    af_array out;
    try {
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
        }
        switch(type) {
        case f32:   out = createArrayHandle<float  >(d, 0); break;
        case c32:   out = createArrayHandle<cfloat >(d, 0); break;
        case f64:   out = createArrayHandle<double >(d, 0); break;
        case c64:   out = createArrayHandle<cdouble>(d, 0); break;
        case b8:    out = createArrayHandle<char   >(d, 0); break;
        case s32:   out = createArrayHandle<int    >(d, 0); break;
        case u32:   out = createArrayHandle<unsigned int >(d, 0); break;
        case u8:    out = createArrayHandle<unsigned char>(d, 0); break;
        case s8:    out = createArrayHandle<char>(d, 0); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;    break;
        }
        std::swap(*result, out);
        ret = AF_SUCCESS;
    }
    CATCHALL
    return ret;
}

//Strong Exception Guarantee
af_err af_constant(af_array *result, double value,
                   const unsigned ndims, const long * const dims,
                   af_dtype type)
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
