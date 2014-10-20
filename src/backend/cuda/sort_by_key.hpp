#include <af/array.h>
#include <Array.hpp>

namespace cuda
{
    template<typename Tk, typename Tv, bool DIR>
    void sort_by_key(Array<Tk> &okey, Array<Tv> &oval,
               const Array<Tk> &ikey, const Array<Tv> &ival, const unsigned dim);
}
