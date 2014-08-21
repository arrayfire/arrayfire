#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <diff.hpp>
#include <kernel/diff.hpp>
#include <cassert>

namespace cuda
{
    template<typename T>
    Array<T> *diff1(const Array<T> &in, const int dim)
    {
        const af::dim4 iDims = in.dims();
        af::dim4 oDims = iDims;
        oDims[dim]--;

        if(iDims.elements() == 0 || oDims.elements() == 0) {
            // error out
            assert(1!=1);
        }

        Array<T> *out = createEmptyArray<T>(oDims);

        kernel::diff1(out->get(), in.get(), dim,
                      oDims.elements(), oDims.ndims(), oDims.get(), out->strides().get(),
                      iDims.elements(), iDims.ndims(), iDims.get(), in.strides().get());

        return out;
    }

    template<typename T>
    Array<T> *diff2(const Array<T> &in, const int dim)
    {
        const af::dim4 iDims = in.dims();
        af::dim4 oDims = iDims;
        oDims[dim] -= 2;

        if(iDims.elements() == 0 || oDims.elements() == 0) {
            // error out
            assert(1!=1);
        }

        Array<T> *out = createEmptyArray<T>(oDims);

        kernel::diff2(out->get(), in.get(), dim,
                      oDims.elements(), oDims.ndims(), oDims.get(), out->strides().get(),
                      iDims.elements(), iDims.ndims(), iDims.get(), in.strides().get());

        return out;
    }

#define INSTANTIATE(T)                                                  \
    template Array<T>* diff1<T>  (const Array<T> &in, const int dim);   \
    template Array<T>* diff2<T>  (const Array<T> &in, const int dim);   \


    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)

}
