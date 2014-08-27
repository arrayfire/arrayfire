#include <af/image.h>
#include <Array.hpp>

namespace opencl
{
    template<typename T>
    Array<T> *resize(const Array<T> &in, const dim_type odim0, const dim_type odim1,
                     const af_interp_type method);
}
