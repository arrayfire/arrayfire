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
    af_array diff1(const af_array &in, const int dim)
    {
        // Bool for dimension
        bool is_dim0 = dim == 0;
        bool is_dim1 = dim == 1;
        bool is_dim2 = dim == 2;
        bool is_dim3 = dim == 3;

        // Create Array of input
        const Array<T> &inArray = getArray<T>(in);

        // Decrement dimension of select dimension
        af::dim4 dims = inArray.dims();
        dims[dim]--;

        // Create output placeholder
        Array<T> *outArray = createValueArray(dims, (T)0);

        // Get pointers to raw data
        const T *inPtr = inArray.get(false);
              T *outPtr = outArray->get();

        // TODO: Improve this
        for(dim_type l = 0; l < dims[3]; l++) {
            for(dim_type k = 0; k < dims[2]; k++) {
                for(dim_type j = 0; j < dims[1]; j++) {
                    for(dim_type i = 0; i < dims[0]; i++) {
                        // Operation: out[index] = in[index + 1 * dim_size] - in[index]
                        int idx = getIdx(inArray.strides(), inArray.offsets(), i, j, k, l);
                        int jdx = getIdx(inArray.strides(), inArray.offsets(),
                                         i + is_dim0, j + is_dim1,
                                         k + is_dim2, l + is_dim3);
                        int odx = getIdx(outArray->strides(), outArray->offsets(), i, j, k, l);
                        outPtr[odx] = inPtr[jdx] - inPtr[idx];
                    }
                }
            }
        }

        return getHandle(*outArray);
    }

    template af_array diff1<float>  (const af_array &in, const int dim);
    template af_array diff1<cfloat> (const af_array &in, const int dim);
    template af_array diff1<double> (const af_array &in, const int dim);
    template af_array diff1<cdouble>(const af_array &in, const int dim);
    template af_array diff1<char>   (const af_array &in, const int dim);
    template af_array diff1<int>    (const af_array &in, const int dim);
    template af_array diff1<uint>   (const af_array &in, const int dim);
    template af_array diff1<uchar>  (const af_array &in, const int dim);

    //////////////////////////////////////////////////////////////////////////

    template<typename T>
    af_array diff2(const af_array &in, const int dim)
    {
        // Bool for dimension
        bool is_dim0 = dim == 0;
        bool is_dim1 = dim == 1;
        bool is_dim2 = dim == 2;
        bool is_dim3 = dim == 3;

        // Create Array of input
        const Array<T> &inArray = getArray<T>(in);

        // Decrement dimension of select dimension
        af::dim4 dims = inArray.dims();
        dims[dim] -= 2;

        // Create output placeholder
        Array<T> *outArray = createValueArray(dims, (T)0);

        // Get pointers to raw data
        const T *inPtr = inArray.get(false);
              T *outPtr = outArray->get();

        // TODO: Improve this
        for(dim_type l = 0; l < dims[3]; l++) {
            for(dim_type k = 0; k < dims[2]; k++) {
                for(dim_type j = 0; j < dims[1]; j++) {
                    for(dim_type i = 0; i < dims[0]; i++) {
                        // Operation: out[index] = in[index + 1 * dim_size] - in[index]
                        int idx = getIdx(inArray.strides(), inArray.offsets(), i, j, k, l);
                        int jdx = getIdx(inArray.strides(), inArray.offsets(),
                                         i + is_dim0, j + is_dim1,
                                         k + is_dim2, l + is_dim3);
                        int kdx = getIdx(inArray.strides(), inArray.offsets(),
                                         i + 2 * is_dim0, j + 2 * is_dim1,
                                         k + 2 * is_dim2, l + 2 * is_dim3);
                        int odx = getIdx(outArray->strides(), outArray->offsets(), i, j, k, l);
                        outPtr[odx] = inPtr[kdx] + inPtr[idx] - inPtr[jdx] - inPtr[jdx];
                    }
                }
            }
        }

        return getHandle(*outArray);

    }

    template af_array diff2<float>  (const af_array &in, const int dim);
    template af_array diff2<cfloat> (const af_array &in, const int dim);
    template af_array diff2<double> (const af_array &in, const int dim);
    template af_array diff2<cdouble>(const af_array &in, const int dim);
    template af_array diff2<char>   (const af_array &in, const int dim);
    template af_array diff2<int>    (const af_array &in, const int dim);
    template af_array diff2<uint>   (const af_array &in, const int dim);
    template af_array diff2<uchar>  (const af_array &in, const int dim);
}
