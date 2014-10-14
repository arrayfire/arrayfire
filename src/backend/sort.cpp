#include <af/array.h>
#include <af/defines.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <sort.hpp>
#include <copy.hpp>

#include<cstdio>

using af::dim4;
using namespace detail;

template<typename T>
static inline void sort(af_array *sx, const af_array in, const bool dir, const unsigned dim)
{
    const Array<T> &inArray = getArray<T>(in);
    Array<T> *sxArray = copyArray<T>(inArray);
    if(dir) {
        sort<T, 1>(*sxArray, inArray, dim);
    } else {
        sort<T, 0>(*sxArray, inArray, dim);
    }
    *sx = getHandle(*sxArray);
}

template<typename T>
static inline void sort_index(af_array *sx, af_array *ix, const af_array in, const bool dir, const unsigned dim)
{
    const Array<T> &inArray = getArray<T>(in);
    Array<T> *sxArray = copyArray<T>(inArray);
    if(dir) {
        sort_index<T, 1>(*sxArray, getWritableArray<unsigned>(*ix), inArray, dim);
    } else {
        sort_index<T, 0>(*sxArray, getWritableArray<unsigned>(*ix), inArray, dim);
    }
    *sx = getHandle(*sxArray);
}

af_err af_sort(af_array *sorted, const af_array in, const bool dir, const unsigned dim)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        af::dim4 idims = info.dims();

        DIM_ASSERT(2, info.elements() > 0);
        // Only Dim 0 supported
        ARG_ASSERT(4, dim == 0);

        af_array sx;
        af_array ix;
        af_create_handle(&ix, idims.ndims(), idims.get(), u32);

        switch(type) {
            case f32: sort<float  >(&sx, in, dir, dim);  break;
            case f64: sort<double >(&sx, in, dir, dim);  break;
            case s32: sort<int    >(&sx, in, dir, dim);  break;
            case u32: sort<uint   >(&sx, in, dir, dim);  break;
            case u8:  sort<uchar  >(&sx, in, dir, dim);  break;
         // case s8:  sort<char   >(&sx, in, dir, dim);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*sorted , sx);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_sort_index(af_array *sorted, af_array *indices, const af_array in, const bool dir, const unsigned dim)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        af::dim4 idims = info.dims();

        DIM_ASSERT(2, info.elements() > 0);
        // Only Dim 0 supported
        ARG_ASSERT(4, dim == 0);

        af_array sx;
        af_array ix;
        af_create_handle(&ix, idims.ndims(), idims.get(), u32);

        switch(type) {
            case f32: sort_index<float  >(&sx, &ix, in, dir, dim);  break;
            case f64: sort_index<double >(&sx, &ix, in, dir, dim);  break;
            case s32: sort_index<int    >(&sx, &ix, in, dir, dim);  break;
            case u32: sort_index<uint   >(&sx, &ix, in, dir, dim);  break;
            case u8:  sort_index<uchar  >(&sx, &ix, in, dir, dim);  break;
         // case s8:  sort_index<char   >(&sx, &ix, in, dir, dim);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*sorted , sx);
        std::swap(*indices, ix);
    }
    CATCHALL;

    return AF_SUCCESS;
}
