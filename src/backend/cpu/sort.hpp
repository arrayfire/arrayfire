#include <af/array.h>
#include <Array.hpp>

namespace cpu
{
    template<typename T, bool DIR>
    void sort(Array<T> &sx, const Array<T> &in, const unsigned dim);

    template<typename T, bool DIR>
    void sort_index(Array<T> &sx, Array<unsigned> &ix, const Array<T> &in, const unsigned dim);
}
