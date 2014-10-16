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
    ///////////////////////////////////////////////////////////////////////////
    // Kernel Functions
    ///////////////////////////////////////////////////////////////////////////

    // Based off of http://stackoverflow.com/a/12399290
    template<typename T, bool DIR>
    void sort0(Array<T> &val, const Array<T> &in)
    {
        // initialize original index locations
              T *val_ptr = val.get();

        function<bool(dim_type, dim_type)> op = greater<T>();
        if(DIR) { op = less<T>(); }

        T *comp_ptr = nullptr;
        for(dim_type w = 0; w < in.dims()[3]; w++) {
            dim_type valW = w * val.strides()[3];
            for(dim_type z = 0; z < in.dims()[2]; z++) {
                dim_type valWZ = valW + z * val.strides()[2];
                for(dim_type y = 0; y < in.dims()[1]; y++) {

                    dim_type valOffset = valWZ + y * val.strides()[1];

                    comp_ptr = val_ptr + valOffset;
                    std::stable_sort(comp_ptr, comp_ptr + val.dims()[0], op);
                }
            }
        }
        return;
    }

    template<typename T, bool DIR>
    void sort0_index(Array<T> &val, Array<uint> &idx, const Array<T> &in)
    {
        // initialize original index locations
           uint *idx_ptr = idx.get();
              T *val_ptr = val.get();
        const T *in_ptr  = in.get();
        function<bool(dim_type, dim_type)> op = greater<T>();
        if(DIR) { op = less<T>(); }

        std::vector<unsigned> seq_vec(idx.dims()[0]);
        std::iota(seq_vec.begin(), seq_vec.end(), 0);

        const T *comp_ptr = nullptr;
        auto comparator = [&comp_ptr, &op](size_t i1, size_t i2) {return op(comp_ptr[i1], comp_ptr[i2]);};

        for(dim_type w = 0; w < in.dims()[3]; w++) {
            dim_type valW = w * val.strides()[3];
            dim_type idxW = w * idx.strides()[3];
            dim_type  inW = w *  in.strides()[3];
            for(dim_type z = 0; z < in.dims()[2]; z++) {
                dim_type valWZ = valW + z * val.strides()[2];
                dim_type idxWZ = idxW + z * idx.strides()[2];
                dim_type  inWZ =  inW + z *  in.strides()[2];
                for(dim_type y = 0; y < in.dims()[1]; y++) {

                    dim_type valOffset = valWZ + y * val.strides()[1];
                    dim_type idxOffset = idxWZ + y * idx.strides()[1];
                    dim_type inOffset  =  inWZ + y *  in.strides()[1];

                    uint *ptr = idx_ptr + idxOffset;
                    std::copy(seq_vec.begin(), seq_vec.end(), ptr);

                    comp_ptr = in_ptr + inOffset;
                    std::stable_sort(ptr, ptr + in.dims()[0], comparator);

                    for (dim_type i = 0; i < val.dims()[0]; ++i){
                        val_ptr[valOffset + i] = in_ptr[inOffset + idx_ptr[idxOffset + i]];
                    }
                }
            }
        }

        return;
    }

    template<typename T, bool DIR>
    void sort(Array<T> &val, const Array<T> &in, const unsigned dim)
    {
        switch(dim) {
            case 0: sort0<T, DIR>(val, in);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

    template<typename T, bool DIR>
    void sort_index(Array<T> &val, Array<unsigned> &idx, const Array<T> &in, const unsigned dim)
    {
        switch(dim) {
            case 0: sort0_index<T, DIR>(val, idx, in);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

#define INSTANTIATE(T)                                                                          \
    template void sort<T, true>(Array<T> &val, const Array<T> &in, const unsigned dim);         \
    template void sort<T,false>(Array<T> &val, const Array<T> &in, const unsigned dim);         \
    template void sort_index<T, true>(Array<T> &val, Array<unsigned> &idx, const Array<T> &in,  \
                                      const unsigned dim);                                      \
    template void sort_index<T,false>(Array<T> &val, Array<unsigned> &idx, const Array<T> &in,  \
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
