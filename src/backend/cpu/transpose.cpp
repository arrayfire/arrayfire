#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <transpose.hpp>

#include <cassert>

using af::dim4;

namespace cpu
{

    static inline unsigned getIdx(const dim4 &strides,
            int i, int j = 0, int k = 0, int l = 0)
    {
        return (l * strides[3] +
                k * strides[2] +
                j * strides[1] +
                i );
    }

    template<typename T>
    Array<T> * transpose(const Array<T> &in)
    {
        const dim4 inDims = in.dims();

        dim4 outDims   = dim4(inDims[1],inDims[0],inDims[2],inDims[3]);

        // create an array with first two dimensions swapped
        Array<T>* out  = createEmptyArray<T>(outDims);

        // get data pointers for input and output Arrays
        T* outData          = out->get();
        const T*   inData   = in.get();

        for (int k=0; k<outDims[2]; ++k) {
            // Outermost loop handles batch mode
            // if input has no data along third dimension
            // this loop runs only once
            for (int j=0; j<outDims[1]; ++j) {
                for (int i=0; i<outDims[0]; ++i) {
                    // calculate array indices based on offsets and strides
                    // the helper getIdx takes care of indices
                    int inIdx  = getIdx(in.strides(),j,i,k);
                    int outIdx = getIdx(out->strides(),i,j,k);
                    outData[outIdx]  = inData[inIdx];
                }
            }
            // outData and inData pointers doesn't need to be
            // offset as the getIdx function is taking care
            // of the batch parameter
        }
        return out;
    }

#define INSTANTIATE(T)\
    template Array<T> * transpose(const Array<T> &in);

    INSTANTIATE(float)
    INSTANTIATE(cfloat)
    INSTANTIATE(double)
    INSTANTIATE(cdouble)
    INSTANTIATE(char)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)

}
