#include <af/array.h>

namespace cpu {
    template<typename T>
    void copyData(T *data, const af_array &arr);

    template<typename T>
    void stridedCopy(T* dst, const T* src, const af::dim4 &dims, const af::dim4 &strides, unsigned dim);
}
