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
static inline void sort(af_array *val, const af_array in, const unsigned dim, const bool dir)
{
    const Array<T> &inArray = getArray<T>(in);
    Array<T> *valArray = copyArray<T>(inArray);
    if(dir) {
        sort<T, 1>(*valArray, inArray, dim);
    } else {
        sort<T, 0>(*valArray, inArray, dim);
    }
    *val = getHandle(*valArray);
}

template<typename T>
static inline void sort_index(af_array *val, af_array *idx, const af_array in,
                              const unsigned dim, const bool dir)
{
    const Array<T> &inArray = getArray<T>(in);
    Array<T> *valArray = copyArray<T>(inArray);
    if(dir) {
        sort_index<T, 1>(*valArray, getWritableArray<unsigned>(*idx), inArray, dim);
    } else {
        sort_index<T, 0>(*valArray, getWritableArray<unsigned>(*idx), inArray, dim);
    }
    *val = getHandle(*valArray);
}

af_err af_sort(af_array *out, const af_array in, const unsigned dim, const bool dir)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();

        DIM_ASSERT(1, info.elements() > 0);
        // Only Dim 0 supported
        ARG_ASSERT(2, dim == 0);

        af_array val;

        switch(type) {
            case f32: sort<float  >(&val, in, dim, dir);  break;
            case f64: sort<double >(&val, in, dim, dir);  break;
            case s32: sort<int    >(&val, in, dim, dir);  break;
            case u32: sort<uint   >(&val, in, dim, dir);  break;
            case u8:  sort<uchar  >(&val, in, dim, dir);  break;
         // case s8:  sort<char   >(&val, in, dim, dir);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out, val);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_sort_index(af_array *out, af_array *indices, const af_array in, const unsigned dim, const bool dir)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        af::dim4 idims = info.dims();

        DIM_ASSERT(2, info.elements() > 0);
        // Only Dim 0 supported
        ARG_ASSERT(3, dim == 0);

        af_array val;
        af_array idx;
        af_create_handle(&idx, idims.ndims(), idims.get(), u32);

        switch(type) {
            case f32: sort_index<float  >(&val, &idx, in, dim, dir);  break;
            case f64: sort_index<double >(&val, &idx, in, dim, dir);  break;
            case s32: sort_index<int    >(&val, &idx, in, dim, dir);  break;
            case u32: sort_index<uint   >(&val, &idx, in, dim, dir);  break;
            case u8:  sort_index<uchar  >(&val, &idx, in, dim, dir);  break;
         // case s8:  sort_index<char   >(&val, &idx, in, dim, dir);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out , val);
        std::swap(*indices, idx);
    }
    CATCHALL;

    return AF_SUCCESS;
}
