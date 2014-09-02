#include <af/image.h>
#include <Array.hpp>

namespace cuda
{
    template<typename T>
    Array<T> *transform(const Array<T> &in, const Array<float> &tf, const af::dim4 &odims, const bool inverse);
}
