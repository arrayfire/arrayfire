//#include <af/defines.h>
#include <af/array.h>

namespace opencl {
    template<typename T>
    T* copyData(const af_array &arr);

    template<typename T>
    void
    copyData(af_array &dst, const T* const src);
}
