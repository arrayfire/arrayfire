#include <vector>
#include <cassert>

#include <af/array.h>
#include <ArrayInfo.hpp>
#include <index.hpp>
#include <helper.h>
#include <backend.h>

using std::vector;
using std::swap;

af_err af_index(af_array *result, const af_array in, const unsigned ndims, const af_seq* index)
{
    //TODO: Check ndims agains max_dims
    af_err ret = AF_ERR_INTERNAL;
    af_array out;
    try {
        if(result)  { out = *result; }

        switch(getInfo(in).getType()) {
            case f32:    indexArray<float>          (out, in, ndims, index);  break;
            case c32:    indexArray<cfloat>         (out, in, ndims, index);  break;
            case f64:    indexArray<double>         (out, in, ndims, index);  break;
            case c64:    indexArray<cdouble>        (out, in, ndims, index);  break;
            case b8:     indexArray<char>           (out, in, ndims, index);  break;
            case s32:    indexArray<int>            (out, in, ndims, index);  break;
            case u32:    indexArray<unsigned>       (out, in, ndims, index);  break;
            case u8:     indexArray<unsigned char>  (out, in, ndims, index);  break;
            case s8:     indexArray<char>           (out, in, ndims, index);  break;
            default:    ret = AF_ERR_NOT_SUPPORTED;                     break;
        }
        if(ret !=AF_ERR_NOT_SUPPORTED)
            ret = AF_SUCCESS;
    }
    CATCHALL

    swap(*result, out);
    return ret;
}
