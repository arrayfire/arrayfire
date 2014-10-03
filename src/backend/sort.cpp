#include <af/array.h>
#include <af/defines.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <sort.hpp>
#include <copy.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline void sort(af_array *sx, af_array *ix, const af_array in, const bool dir, const unsigned dim)
{
    sort<T>(getWritableArray<T>(*sx), getWritableArray<uint>(*ix), getArray<T>(in), dir, dim);
}

af_err af_sort(af_array *sorted, af_array *indices, const af_array in, const bool dir, const unsigned dim)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        af::dim4 idims = info.dims();

        DIM_ASSERT(2, info.elements() > 0);
        // Only Dim 0 supported
        ARG_ASSERT(4, dim == 0);
        ARG_ASSERT(2, info.ndims() <= 2);

        af_array sx;
        af_array ix;
        af_create_handle(&sx, idims.ndims(), idims.get(), type);
        af_create_handle(&ix, idims.ndims(), idims.get(), u32);

        switch(type) {
            case f32: sort<float  >(&sx, &ix, in, dir, dim);  break;
            case f64: sort<double >(&sx, &ix, in, dir, dim);  break;
            case s32: sort<int    >(&sx, &ix, in, dir, dim);  break;
            case u32: sort<uint   >(&sx, &ix, in, dir, dim);  break;
            // case s8:  sort<char   >(&sx, &ix, in, dir, dim);  break;
            case u8:  sort<uchar  >(&sx, &ix, in, dir, dim);  break;
            default:  TYPE_ERROR(1, type);
        }
        std::swap(*sorted , sx);
        std::swap(*indices, ix);
    }
    CATCHALL;

    return AF_SUCCESS;
}
