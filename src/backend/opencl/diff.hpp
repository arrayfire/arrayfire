#include <af/array.h>
#include <Array.hpp>

namespace opencl
{
    template<typename T>
    Array<T> *diff1(const Array<T> &in, const int dim);

    template<typename T>
    Array<T> *diff2(const Array<T> &in, const int dim);
}
