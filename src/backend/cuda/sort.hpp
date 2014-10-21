#include <af/array.h>
#include <Array.hpp>

namespace cuda
{
    template<typename T, bool DIR>
    void sort(Array<T> &val, const Array<T> &in, const unsigned dim);

    template<typename T, bool DIR>
    void sort_index(Array<T> &val, Array<unsigned> &idx, const Array<T> &in, const unsigned dim);

    template<typename Tk, typename Tv, bool DIR>
    void sort_by_key(Array<Tk> &okey, Array<Tv> &oval,
               const Array<Tk> &ikey, const Array<Tv> &ival, const unsigned dim);
}
