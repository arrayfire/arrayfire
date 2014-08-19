#include <type_traits>
#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <diff.hpp>

namespace cpu
{
    unsigned getIdx(af::dim4 strides, af::dim4 offs, int i, int j = 0, int k = 0, int l = 0)
    {
        return ((l + offs[3]) * strides[3] +
                (k + offs[2]) * strides[2] +
                (j + offs[1]) * strides[1] +
                (i + offs[0]));
    }

    template<typename T>
    Array<T>* diff1(const Array<T> &in, const int dim)
    {
        // Bool for dimension
        bool is_dim0 = dim == 0;
        bool is_dim1 = dim == 1;
        bool is_dim2 = dim == 2;
        bool is_dim3 = dim == 3;

        // Decrement dimension of select dimension
        af::dim4 dims = in.dims();
        dims[dim]--;

        // Create output placeholder
        Array<T> *outArray = createValueArray(dims, (T)0);

        // Get pointers to raw data
        const T *inPtr = in.get(false);
              T *outPtr = outArray->get();

        // TODO: Improve this
        for(dim_type l = 0; l < dims[3]; l++) {
            for(dim_type k = 0; k < dims[2]; k++) {
                for(dim_type j = 0; j < dims[1]; j++) {
                    for(dim_type i = 0; i < dims[0]; i++) {
                        // Operation: out[index] = in[index + 1 * dim_size] - in[index]
                        int idx = getIdx(in.strides(), in.offsets(), i, j, k, l);
                        int jdx = getIdx(in.strides(), in.offsets(),
                                         i + is_dim0, j + is_dim1,
                                         k + is_dim2, l + is_dim3);
                        int odx = getIdx(outArray->strides(), outArray->offsets(), i, j, k, l);
                        outPtr[odx] = inPtr[jdx] - inPtr[idx];
                    }
                }
            }
        }

        return outArray;
    }

    template<typename T>
    Array<T>* diff2(const Array<T> &in, const int dim)
    {
        // Bool for dimension
        bool is_dim0 = dim == 0;
        bool is_dim1 = dim == 1;
        bool is_dim2 = dim == 2;
        bool is_dim3 = dim == 3;

        // Decrement dimension of select dimension
        af::dim4 dims = in.dims();
        dims[dim] -= 2;

        // Create output placeholder
        Array<T> *outArray = createValueArray(dims, (T)0);

        // Get pointers to raw data
        const T *inPtr = in.get(false);
              T *outPtr = outArray->get();

        // TODO: Improve this
        for(dim_type l = 0; l < dims[3]; l++) {
            for(dim_type k = 0; k < dims[2]; k++) {
                for(dim_type j = 0; j < dims[1]; j++) {
                    for(dim_type i = 0; i < dims[0]; i++) {
                        // Operation: out[index] = in[index + 1 * dim_size] - in[index]
                        int idx = getIdx(in.strides(), in.offsets(), i, j, k, l);
                        int jdx = getIdx(in.strides(), in.offsets(),
                                         i + is_dim0, j + is_dim1,
                                         k + is_dim2, l + is_dim3);
                        int kdx = getIdx(in.strides(), in.offsets(),
                                         i + 2 * is_dim0, j + 2 * is_dim1,
                                         k + 2 * is_dim2, l + 2 * is_dim3);
                        int odx = getIdx(outArray->strides(), outArray->offsets(), i, j, k, l);
                        outPtr[odx] = inPtr[kdx] + inPtr[idx] - inPtr[jdx] - inPtr[jdx];
                    }
                }
            }
        }

        return outArray;
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
