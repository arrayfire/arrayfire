#include <Array.hpp>
#include <sort.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cpu.hpp>
#include <algorithm>
#include <queue>
#include <future>

using std::greater;
using std::less;
using std::sort;
using std::function;
using std::queue;
using std::future;
using std::async;

namespace cpu
{
    // Based off of http://stackoverflow.com/a/12399290
    template<typename T>
    void sort(Array<T> &sx, Array<uint> &ix, const Array<T> &in, const bool dir, const unsigned dim)
    {
        if(dim != 0) CPU_NOT_SUPPORTED();

        // initialize original index locations
        uint *ixptr = ix.get();
        for (size_t j = 0; j != ix.dims()[1]; ++j) {
            for (size_t i = 0; i != ix.dims()[0]; ++i){
                ixptr[j*ix.dims()[0]+i] = i;
            }
        }

        const T *nptr = in.get();
        function<bool(size_t, size_t)> op = greater<T>();
        if(dir) { op = less<T>(); }
        auto comparator = [&nptr, &op](size_t i1, size_t i2) {return op(nptr[i1], nptr[i2]);};

        uint *begin_ptr = nullptr;
        for (size_t col = 0; col < in.dims()[1]; col++) {
            begin_ptr = in.dims()[0] * col + ixptr;
            std::sort(begin_ptr, begin_ptr + in.dims()[0], comparator);
        }

        T *sxptr = sx.get();
        for (size_t j = 0; j != sx.dims()[1]; ++j) {
            uint offset = j*ix.dims()[0];
            for (size_t i = 0; i != sx.dims()[0]; ++i){
                sxptr[offset+i] = nptr[offset + ixptr[offset+i]];
            }
        }

        return;
    }

#define INSTANTIATE(T)                                                                                          \
    template void sort<T>(Array<T> &sx, Array<uint> &ix, const Array<T> &in, const bool dir, const unsigned dim);  \

    INSTANTIATE(float)
    INSTANTIATE(double)
    //INSTANTIATE(cfloat)
    //INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
