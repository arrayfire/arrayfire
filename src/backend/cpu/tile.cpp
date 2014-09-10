#include <Array.hpp>
#include <tile.hpp>
#include <stdexcept>

namespace cpu
{
    template<typename T>
    Array<T> *tile(const Array<T> &in, const af::dim4 &tileDims)
    {
        const af::dim4 iDims = in.dims();
        af::dim4 oDims = iDims;
        oDims *= tileDims;

        if(iDims.elements() == 0 || oDims.elements() == 0) {
            throw std::runtime_error("Elements are 0");
        }

        Array<T> *out = createEmptyArray<T>(oDims);

        T* outPtr = out->get();
        const T* inPtr = in.get();

        //tile(*out, in, tileDims.get());
        for(dim_type ow = 0; ow < oDims[3]; ow++) {
            for(dim_type oz = 0; oz < oDims[2]; oz++) {
                for(dim_type oy = 0; oy < oDims[1]; oy++) {
                    for(dim_type ox = 0; ox < oDims[0]; ox++) {
                        const dim_type ix = (iDims[0] == oDims[0]) ? ox :
                                             ox - ((ox / iDims[0]) * iDims[0]);
                        const dim_type iy = (iDims[1] == oDims[1]) ? oy :
                                             oy - ((oy / iDims[1]) * iDims[1]);
                        const dim_type iz = (iDims[2] == oDims[2]) ? oz :
                                             oz - ((oz / iDims[2]) * iDims[2]);
                        const dim_type iw = (iDims[3] == oDims[3]) ? ow :
                                             ow - ((ow / iDims[3]) * iDims[3]);

                        unsigned iMem = iw * in.strides()[3] + iz * in.strides()[2] +
                                        iy * in.strides()[1] + ix;
                        unsigned oMem = ow * out->strides()[3] + oz * out->strides()[2] +
                                        oy * out->strides()[1] + ox;

                        outPtr[oMem] = inPtr[iMem];

                    }
                }
            }
        }

        return out;
    }

#define INSTANTIATE(T)                                                         \
    template Array<T>* tile<T>(const Array<T> &in, const af::dim4 &tileDims);  \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)

}
