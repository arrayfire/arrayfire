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

template<typename Tk, typename Tv>
static inline void sort_by_key(af_array *okey, af_array *oval, const af_array ikey, const af_array ival,
                               const unsigned dim, const bool dir)
{
    const Array<Tk> &ikeyArray = getArray<Tk>(ikey);
    const Array<Tv> &ivalArray = getArray<Tv>(ival);
    Array<Tk> *okeyArray = copyArray<Tk>(ikeyArray);
    Array<Tv> *ovalArray = copyArray<Tv>(ivalArray);
    if(dir) {
        sort_by_key<Tk, Tv, 1>(*okeyArray, *ovalArray, ikeyArray, ivalArray, dim);
    } else {
        sort_by_key<Tk, Tv, 0>(*okeyArray, *ovalArray, ikeyArray, ivalArray, dim);
    }
    *okey = getHandle(*okeyArray);
    *oval = getHandle(*ovalArray);
}

template<typename Tk>
af_err sort_by_key_tmplt(af_array *okey, af_array *oval, const af_array ikey, const af_array ival,
                         const unsigned dim, const bool dir)
{
    try {
        ArrayInfo info = getInfo(ival);
        af_dtype vtype = info.getType();

        switch(vtype) {
            case f32: sort_by_key<Tk, float  >(okey, oval, ikey, ival, dim, dir);  break;
            case f64: sort_by_key<Tk, double >(okey, oval, ikey, ival, dim, dir);  break;
            case s32: sort_by_key<Tk, int    >(okey, oval, ikey, ival, dim, dir);  break;
            case u32: sort_by_key<Tk, uint   >(okey, oval, ikey, ival, dim, dir);  break;
            case u8:  sort_by_key<Tk, uchar  >(okey, oval, ikey, ival, dim, dir);  break;
         // case s8:  sort_by_key<Tk, char   >(okey, oval, ikey, ival, dim, dir);  break;
            default:  TYPE_ERROR(1, vtype);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_sort_by_key(af_array *out_keys, af_array *out_values,
                      const af_array keys, const af_array values,
                      const unsigned dim, const bool dir)
{
    try {
        ArrayInfo info = getInfo(keys);
        af_dtype type = info.getType();

        ArrayInfo vinfo = getInfo(values);

        DIM_ASSERT(3, info.elements() > 0);
        DIM_ASSERT(4, info.dims() == vinfo.dims());
        // Only Dim 0 supported
        ARG_ASSERT(5, dim == 0);

        af_array oKey;
        af_array oVal;

        switch(type) {
            case f32: sort_by_key_tmplt<float  >(&oKey, &oVal, keys, values, dim, dir);  break;
            case f64: sort_by_key_tmplt<double >(&oKey, &oVal, keys, values, dim, dir);  break;
            case s32: sort_by_key_tmplt<int    >(&oKey, &oVal, keys, values, dim, dir);  break;
            case u32: sort_by_key_tmplt<uint   >(&oKey, &oVal, keys, values, dim, dir);  break;
            case u8:  sort_by_key_tmplt<uchar  >(&oKey, &oVal, keys, values, dim, dir);  break;
         // case s8:  sort_by_key_tmplt<char   >(&oKey, &oVal, keys, values, dim, dir);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*out_keys , oKey);
        std::swap(*out_values , oVal);
    }
    CATCHALL;

    return AF_SUCCESS;
}
