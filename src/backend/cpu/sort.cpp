#include <Array.hpp>
#include <sort.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cpu.hpp>
#include <algorithm>

using std::greater;
using std::less;
using std::sort;
using std::function;

namespace cpu
{
    // Based off of http://stackoverflow.com/a/12399290
    template<typename T>
    void sort(Array<T> &sx, Array<unsigned> &ix, const Array<T> &in, const bool dir, const unsigned dim)
    {
        // initialize original index locations
        unsigned *ixptr = ix.get();
        for (size_t i = 0; i != ix.elements(); ++i) ixptr[i] = i;

        const T *nptr = in.get();
        function<bool(size_t, size_t)> op = greater<T>();
        if(dir) { op = less<T>(); }
        auto comparator = [&nptr, &op](size_t i1, size_t i2) {return op(nptr[i1], nptr[i2]);};

        std::stable_sort(ix.get(), ix.get() + ix.elements(), comparator);

        T *sxptr = sx.get();

        for (int i = 0; i < sx.elements(); i++) {
            sxptr[i] = nptr[ixptr[i]];
        }

        return;
    }

#define INSTANTIATE(T)                                                              \
    template void sort<T>(Array<T> &sx, Array<unsigned> &ix, const Array<T> &in,    \
                          const bool dir, const unsigned dim);                      \

    INSTANTIATE(float)
    INSTANTIATE(double)
    //INSTANTIATE(cfloat)
    //INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
