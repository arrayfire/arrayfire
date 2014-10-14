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
    template<typename T, bool DIR>
    void sort0(Array<T> &sx, const Array<T> &in)
    {
        // initialize original index locations
              T *sx_ptr = sx.get();

        function<bool(dim_type, dim_type)> op = greater<T>();
        if(DIR) { op = less<T>(); }


        T *comp_ptr = nullptr;
        for(dim_type w = 0; w < in.dims()[3]; w++) {
            for(dim_type z = 0; z < in.dims()[2]; z++) {
                for(dim_type y = 0; y < in.dims()[1]; y++) {

                    dim_type sxOffset = w * sx.strides()[3] + z * sx.strides()[2]
                                      + y * sx.strides()[1];

                    comp_ptr = sx_ptr + sxOffset;
                    std::stable_sort(comp_ptr, comp_ptr + sx.dims()[0], op);
                }
            }
        }
        return;
    }

    template<typename T, bool DIR>
    void sort0_index(Array<T> &sx, Array<uint> &ix, const Array<T> &in)
    {
        // initialize original index locations
           uint *ix_ptr = ix.get();
              T *sx_ptr = sx.get();
        const T *in_ptr = in.get();
        function<bool(dim_type, dim_type)> op = greater<T>();
        if(DIR) { op = less<T>(); }

        std::vector<unsigned> seq_vec(ix.dims()[0]);
        std::iota(seq_vec.begin(), seq_vec.end(), 0);

        const T *comp_ptr = nullptr;
        auto comparator = [&comp_ptr, &op](size_t i1, size_t i2) {return op(comp_ptr[i1], comp_ptr[i2]);};

        for(dim_type w = 0; w < in.dims()[3]; w++) {
            for(dim_type z = 0; z < in.dims()[2]; z++) {
                for(dim_type y = 0; y < in.dims()[1]; y++) {

                    dim_type sxOffset = w * sx.strides()[3] + z * sx.strides()[2]
                                      + y * sx.strides()[1];
                    dim_type inOffset = w * in.strides()[3] + z * in.strides()[2]
                                      + y * in.strides()[1];
                    dim_type ixOffset = w * ix.strides()[3] + z * ix.strides()[2]
                                      + y * ix.strides()[1];

                    uint *ptr = ix_ptr + ixOffset;
                    std::copy(seq_vec.begin(), seq_vec.end(), ptr);

                    comp_ptr = in_ptr + inOffset;
                    std::stable_sort(ptr, ptr + in.dims()[0], comparator);

                    for (dim_type i = 0; i < sx.dims()[0]; ++i){
                        sx_ptr[sxOffset + i] = in_ptr[inOffset + ix_ptr[ixOffset + i]];
                    }
                }
            }
        }

        return;
    }

    template<typename T, bool DIR>
    void sort(Array<T> &sx, const Array<T> &in, const unsigned dim)
    {
        switch(dim) {
            case 0: sort0<T, DIR>(sx, in);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

    template<typename T, bool DIR>
    void sort_index(Array<T> &sx, Array<unsigned> &ix, const Array<T> &in, const unsigned dim)
    {
        switch(dim) {
            case 0: sort0_index<T, DIR>(sx, ix, in);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

#define INSTANTIATE(T)                                                                          \
    template void sort<T, true>(Array<T> &sx, const Array<T> &in, const unsigned dim);          \
    template void sort<T,false>(Array<T> &sx, const Array<T> &in, const unsigned dim);          \
    template void sort_index<T, true>(Array<T> &sx, Array<unsigned> &ix, const Array<T> &in,    \
                                      const unsigned dim);                                      \
    template void sort_index<T,false>(Array<T> &sx, Array<unsigned> &ix, const Array<T> &in,    \
                                      const unsigned dim);                                      \

    INSTANTIATE(float)
    INSTANTIATE(double)
    //INSTANTIATE(cfloat)
    //INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
