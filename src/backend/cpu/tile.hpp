#include <af/array.h>
#include <Array.hpp>

namespace cpu
{
    template<typename T>
    Array<T> *tile(const Array<T> &in, const af::dim4 &tileDims);
}
