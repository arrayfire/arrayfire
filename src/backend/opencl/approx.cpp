#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <approx.hpp>
#include <kernel/approx.hpp>
#include <stdexcept>

namespace opencl
{
    template<typename Ty, typename Tp>
    Array<Ty> *approx1(const Array<Ty> &in, const Array<Tp> &pos,
                       const af_interp_type method, const float offGrid)
    {
        af::dim4 odims = in.dims();
        odims[0] = pos.dims()[0];

        // Create output placeholder
        Array<Ty> *out = createEmptyArray<Ty>(odims);

        switch(method) {
            case AF_INTERP_NEAREST:
                kernel::approx1<Ty, Tp, AF_INTERP_NEAREST>
                              (out->get(), out->dims().get(), out->elements(),
                               in.get(), in.dims().get(), in.elements(),
                               pos.get(), pos.dims().get(), out->strides().get(),
                               in.strides().get(), pos.strides().get(), offGrid,
                               in.getOffset(), pos.getOffset());
                break;
            case AF_INTERP_LINEAR:
                kernel::approx1<Ty, Tp, AF_INTERP_LINEAR>
                              (out->get(), out->dims().get(), out->elements(),
                               in.get(), in.dims().get(), in.elements(),
                               pos.get(), pos.dims().get(), out->strides().get(),
                               in.strides().get(), pos.strides().get(), offGrid,
                               in.getOffset(), pos.getOffset());
                break;
            default:
                break;
        }
        return out;
    }

    template<typename Ty, typename Tp>
    Array<Ty> *approx2(const Array<Ty> &in, const Array<Tp> &pos0, const Array<Tp> &pos1,
                       const af_interp_type method, const float offGrid)
    {
        af::dim4 odims = pos0.dims();
        odims[2] = in.dims()[2];
        odims[3] = in.dims()[3];

        // Create output placeholder
        Array<Ty> *out = createEmptyArray<Ty>(odims);

        switch(method) {
            case AF_INTERP_NEAREST:
                kernel::approx2<Ty, Tp, AF_INTERP_NEAREST>
                              (out->get(), out->dims().get(), out->elements(),
                               in.get(), in.dims().get(), in.elements(),
                               pos0.get(), pos0.dims().get(), pos1.get(), pos1.dims().get(),
                               out->strides().get(), in.strides().get(),
                               pos0.strides().get(), pos1.strides().get(),
                               offGrid, in.getOffset(), pos0.getOffset(), pos1.getOffset());
                break;
            case AF_INTERP_LINEAR:
                kernel::approx2<Ty, Tp, AF_INTERP_LINEAR>
                              (out->get(), out->dims().get(), out->elements(),
                               in.get(), in.dims().get(), in.elements(),
                               pos0.get(), pos0.dims().get(), pos1.get(), pos1.dims().get(),
                               out->strides().get(), in.strides().get(),
                               pos0.strides().get(), pos1.strides().get(),
                               offGrid, in.getOffset(), pos0.getOffset(), pos1.getOffset());
                break;
            default:
                break;
        }
        return out;
    }

#define INSTANTIATE(Ty, Tp)                                                                     \
    template Array<Ty>* approx1<Ty, Tp>(const Array<Ty> &in, const Array<Tp> &pos,              \
                                        const af_interp_type method, const float offGrid);      \
    template Array<Ty>* approx2<Ty, Tp>(const Array<Ty> &in, const Array<Tp> &pos0,             \
                                        const Array<Tp> &pos1, const af_interp_type method,     \
                                        const float offGrid);                                   \

    INSTANTIATE(float  , float )
    INSTANTIATE(double , double)
    INSTANTIATE(cfloat , float )
    INSTANTIATE(cdouble, double)
}
