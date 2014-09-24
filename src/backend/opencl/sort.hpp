#include <af/array.h>
#include <Array.hpp>

namespace opencl
{
    template<typename T>
    void sort(Array<T> &sx, Array<T> &ix, const Array<T> &in, const bool dir, const unsigned dim);
}
