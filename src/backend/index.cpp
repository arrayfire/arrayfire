#include <vector>
#include <cassert>

#include <af/array.h>
#include <af/index.h>
#include <ArrayInfo.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <Array.hpp>

using namespace detail;
using std::vector;
using std::swap;

template<typename T>
static void indexArray(af_array &dest, const af_array &src, const unsigned ndims, const af_seq *index)
{
    using af::toOffset;
    using af::toDims;
    using af::toStride;

    const Array<T> &parent = getArray<T>(src);
    vector<af_seq> index_(index, index+ndims);
    Array<T>* dst =  createSubArray(    parent,
                                        toDims(index_, parent.dims()),
                                        toOffset(index_),
                                        toStride(index_, parent.dims()) );
    dest = getHandle(*dst);
}

af_err af_index(af_array *result, const af_array in, const unsigned ndims, const af_seq* index)
{
    //TODO: Check ndims agains max_dims
    af_err ret = AF_ERR_INTERNAL;
    af_array out;
    try {
        if(result)  { out = *result; }

        switch(getInfo(in).getType()) {
            case f32:    indexArray<float>   (out, in, ndims, index);  break;
            case c32:    indexArray<cfloat>  (out, in, ndims, index);  break;
            case f64:    indexArray<double>  (out, in, ndims, index);  break;
            case c64:    indexArray<cdouble> (out, in, ndims, index);  break;
            case b8:     indexArray<char>    (out, in, ndims, index);  break;
            case s32:    indexArray<int>     (out, in, ndims, index);  break;
            case u32:    indexArray<unsigned>(out, in, ndims, index);  break;
            case u8:     indexArray<uchar>   (out, in, ndims, index);  break;
            case s8:     indexArray<char>    (out, in, ndims, index);  break;
            default:    ret = AF_ERR_NOT_SUPPORTED;                    break;
        }
        if(ret !=AF_ERR_NOT_SUPPORTED)
            ret = AF_SUCCESS;
    }
    CATCHALL

    swap(*result, out);
    return ret;
}
