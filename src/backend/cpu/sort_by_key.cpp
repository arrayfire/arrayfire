#include <Array.hpp>
#include <sort_by_key.hpp>
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

    template<typename Tk, typename Tv, bool DIR>
    void sort0_by_key(Array<Tk> &okey, Array<Tv> &oval, const Array<Tk> &ikey, const Array<Tv> &ival)
    {
        function<bool(dim_type, dim_type)> op = greater<Tk>();
        if(DIR) { op = less<Tk>(); }

        // Get pointers and initialize original index locations
        Array<unsigned> *oidx = createValueArray(ikey.dims(), 0u);
            uint *oidx_ptr = oidx->get();
              Tk *okey_ptr = okey.get();
              Tv *oval_ptr = oval.get();
        const Tk *ikey_ptr = ikey.get();
        const Tv *ival_ptr = ival.get();

        std::vector<unsigned> seq_vec(oidx->dims()[0]);
        std::iota(seq_vec.begin(), seq_vec.end(), 0);

        const Tk *comp_ptr = nullptr;
        auto comparator = [&comp_ptr, &op](size_t i1, size_t i2) {return op(comp_ptr[i1], comp_ptr[i2]);};

        for(dim_type w = 0; w < ikey.dims()[3]; w++) {
            dim_type okeyW = w * okey.strides()[3];
            dim_type ovalW = w * oval.strides()[3];
            dim_type oidxW = w * oidx->strides()[3];
            dim_type ikeyW = w * ikey.strides()[3];
            dim_type ivalW = w * ival.strides()[3];

            for(dim_type z = 0; z < ikey.dims()[2]; z++) {
                dim_type okeyWZ = okeyW + z * okey.strides()[2];
                dim_type ovalWZ = ovalW + z * oval.strides()[2];
                dim_type oidxWZ = oidxW + z * oidx->strides()[2];
                dim_type ikeyWZ = ikeyW + z * ikey.strides()[2];
                dim_type ivalWZ = ivalW + z * ival.strides()[2];

                for(dim_type y = 0; y < ikey.dims()[1]; y++) {

                    dim_type okeyOffset = okeyWZ + y * okey.strides()[1];
                    dim_type ovalOffset = ovalWZ + y * oval.strides()[1];
                    dim_type oidxOffset = oidxWZ + y * oidx->strides()[1];
                    dim_type ikeyOffset = ikeyWZ + y * ikey.strides()[1];
                    dim_type ivalOffset = ivalWZ + y * ival.strides()[1];

                    uint *ptr = oidx_ptr + oidxOffset;
                    std::copy(seq_vec.begin(), seq_vec.end(), ptr);

                    comp_ptr = ikey_ptr + ikeyOffset;
                    std::stable_sort(ptr, ptr + ikey.dims()[0], comparator);

                    for (dim_type i = 0; i < oval.dims()[0]; ++i){
                        uint sortIdx = oidx_ptr[oidxOffset + i];
                        okey_ptr[okeyOffset + i] = ikey_ptr[ikeyOffset + sortIdx];
                        oval_ptr[ovalOffset + i] = ival_ptr[ivalOffset + sortIdx];
                    }
                }
            }
        }

        return;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Wrapper Functions
    ///////////////////////////////////////////////////////////////////////////
    template<typename Tk, typename Tv, bool DIR>
    void sort_by_key(Array<Tk> &okey, Array<Tv> &oval,
               const Array<Tk> &ikey, const Array<Tv> &ival, const unsigned dim)
    {
        switch(dim) {
            case 0: sort0_by_key<Tk, Tv, DIR>(okey, oval, ikey, ival);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

#define INSTANTIATE(Tk, Tv)                                                                     \
    template void                                                                               \
    sort_by_key<Tk, Tv, true>(Array<Tk> &okey, Array<Tv> &oval,                                 \
                        const Array<Tk> &ikey, const Array<Tv> &ival, const unsigned dim);      \
    template void                                                                               \
    sort_by_key<Tk, Tv,false>(Array<Tk> &okey, Array<Tv> &oval,                                 \
                        const Array<Tk> &ikey, const Array<Tv> &ival, const unsigned dim);      \

#define INSTANTIATE1(Tk)       \
    INSTANTIATE(Tk, float)     \
    INSTANTIATE(Tk, double)    \
    INSTANTIATE(Tk, int)       \
    INSTANTIATE(Tk, uint)      \
    INSTANTIATE(Tk, char)      \
    INSTANTIATE(Tk, uchar)     \

    INSTANTIATE1(float)
    INSTANTIATE1(double)
    INSTANTIATE1(int)
    INSTANTIATE1(uint)
    INSTANTIATE1(char)
    INSTANTIATE1(uchar)
}
