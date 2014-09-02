#include <type_traits>
#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <resize.hpp>

namespace cpu
{
    /**
     * noop function for round to avoid compilation
     * issues due to lack of this function in C90 based
     * compilers, it is only present in C99 and C++11
     *
     * This is not a full fledged implementation, this function
     * is to be used only for positive numbers, i m using it here
     * for calculating dimensions of arrays
     */
    dim_type round2int(float value)
    {
        return (dim_type)(value+0.5f);
    }

    template<typename T, af_interp_type method>
    struct resize_op
    {
        void operator()(T *outPtr, const T *inPtr, const af::dim4 &odims, const af::dim4 &idims,
                  const af::dim4 &ostrides, const af::dim4 &istrides,
                  const dim_type x, const dim_type y)
        {
            return;
        }
    };

    template<typename T>
    struct resize_op<T, AF_INTERP_NEAREST>
    {
        void operator()(T *outPtr, const T *inPtr, const af::dim4 &odims, const af::dim4 &idims,
                const af::dim4 &ostrides, const af::dim4 &istrides,
                const dim_type x, const dim_type y)
        {
            // Compute Indices
            dim_type i_x = round2int((float)x / (odims[0] / (float)idims[0]));
            dim_type i_y = round2int((float)y / (odims[1] / (float)idims[1]));

            if (i_x >= idims[0]) i_x = idims[0] - 1;
            if (i_y >= idims[1]) i_y = idims[1] - 1;

            dim_type i_off = i_y * istrides[1] + i_x;
            dim_type o_off =   y * ostrides[1] + x;
            // Copy values from all channels
            for(dim_type z = 0; z < odims[2]; z++) {
                outPtr[o_off + z * ostrides[2]] = inPtr[i_off + z * istrides[2]];
            }
        }
    };

    template<typename T>
    struct resize_op<T, AF_INTERP_BILINEAR>
    {
        void operator()(T *outPtr, const T *inPtr, const af::dim4 &odims, const af::dim4 &idims,
                const af::dim4 &ostrides, const af::dim4 &istrides,
                const dim_type x, const dim_type y)
        {
            // Compute Indices
            float f_x = (float)x / (odims[0] / (float)idims[0]);
            float f_y = (float)y / (odims[1] / (float)idims[1]);

            dim_type i1_x  = floor(f_x);
            dim_type i1_y  = floor(f_y);

            if (i1_x >= idims[0]) i1_x = idims[0] - 1;
            if (i1_y >= idims[1]) i1_y = idims[1] - 1;

            float b   = f_x - i1_x;
            float a   = f_y - i1_y;

            dim_type i2_x  = (i1_x + 1 >= idims[0] ? idims[0] - 1 : i1_x + 1);
            dim_type i2_y  = (i1_y + 1 >= idims[1] ? idims[1] - 1 : i1_y + 1);

            dim_type o_off = y * ostrides[1] + x;
            // Copy values from all channels
            for(dim_type z = 0; z < odims[2]; z++) {
                T p1 = inPtr[i1_y * istrides[1] + i1_x + z * istrides[2]];
                T p2 = inPtr[i2_y * istrides[1] + i1_x + z * istrides[2]];
                T p3 = inPtr[i1_y * istrides[1] + i2_x + z * istrides[2]];
                T p4 = inPtr[i2_y * istrides[1] + i2_x + z * istrides[2]];

                outPtr[o_off + z * ostrides[2]] =
                                (1.0f - a) * (1.0f - b) * p1 +
                                     a     * (1.0f - b) * p2 +
                                (1.0f - a) *      b     * p3 +
                                     a     *      b     * p4;
            }
        }
    };

    template<typename T, af_interp_type method>
    void resize_(T *outPtr, const T *inPtr, const af::dim4 &odims, const af::dim4 &idims,
                 const af::dim4 &ostrides, const af::dim4 &istrides)
    {
        resize_op<T, method> op;
        for(dim_type y = 0; y < odims[1]; y++) {
            for(dim_type x = 0; x < odims[0]; x++) {
                op(outPtr, inPtr, odims, idims, ostrides, istrides, x, y);
            }
        }
    }

    template<typename T>
    Array<T>* resize(const Array<T> &in, const dim_type odim0, const dim_type odim1,
                     const af_interp_type method)
    {
        af::dim4 idims = in.dims();
        af::dim4 odims(odim0, odim1, idims[2], idims[3]);

        // Create output placeholder
        Array<T> *outArray = createValueArray(odims, (T)0);

        // Get pointers to raw data
        const T *inPtr = in.get();
              T *outPtr = outArray->get();

        af::dim4 ostrides = outArray->strides();
        af::dim4 istrides = in.strides();

        switch(method) {
            case AF_INTERP_NEAREST:
                resize_<T, AF_INTERP_NEAREST>(outPtr, inPtr, odims, idims, ostrides, istrides);
                break;
            case AF_INTERP_BILINEAR:
                resize_<T, AF_INTERP_BILINEAR>(outPtr, inPtr, odims, idims, ostrides, istrides);
                break;
            default:
                break;
        }
        return outArray;
    }


#define INSTANTIATE(T)                                                                            \
    template Array<T>* resize<T> (const Array<T> &in, const dim_type odim0, const dim_type odim1, \
                                  const af_interp_type method);


    INSTANTIATE(float)
    INSTANTIATE(double)
    //INSTANTIATE(cfloat)
    //INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
