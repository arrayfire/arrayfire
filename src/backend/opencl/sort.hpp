#include <af/array.h>
#include <Array.hpp>

namespace opencl
{
    template<typename T, bool DIR>
    void sort(Array<T> &sx, Array<unsigned> &ix, const Array<T> &in, const unsigned dim);
}
