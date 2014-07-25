#include <af/array.h>

namespace cpu {
    template<typename T>
    T* copyData(const af_array &arr);

    template<typename T>
    void
    copyData(af_array &dst, const T* const src);
}
