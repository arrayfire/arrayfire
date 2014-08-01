#include <af/array.h>

namespace cpu {
    template<typename T>
    af_array createArrayHandle(af::dim4 d, double val);

    template<typename T>
    af_array createArrayHandle(af::dim4 d, const T * const data);

    template<typename T>
    void
    destroyArrayHandle(const af_array& arr);
}
