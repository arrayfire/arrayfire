#include <af/array.h>
#include <Array.hpp>

namespace cpu
{
    template<typename T>
    void sort(Array<T> &sx, Array<uint> &ix, const Array<T> &in, const bool dir, const unsigned dim);
}
