#include <af/array.h>

namespace cuda {
    template<typename T>
    af_array createArrayHandle(af::dim4 d, double val);
}
