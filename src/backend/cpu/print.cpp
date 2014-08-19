
#include <iostream>
#include <print.hpp>
#include <Array.hpp>
#include <iostream>

namespace cpu
{
    using std::ostream;
    using std::endl;

    template<typename T>
    void
    printer(ostream &out, const T* ptr, const Array<T> &info, unsigned dim)
    {

        dim_type stride =   info.strides()[dim];
        dim_type d      =   info.dims()[dim];

        if(dim == 0) {
            for(dim_type i = 0, j = 0; i < d; i++, j+=stride) {
                out << ptr[j] << "\t";
            }
            out << endl;
        }
        else {
            for(int i = 0; i < d; i++) {
                printer(out, ptr, info, dim - 1);
                ptr += stride;
            }
        }

    }

    template<typename T>
    void
    operator <<(ostream &out, const Array<T> &arr)
    {
        out << "TRANSPOSED\n";
        out << "Dim:" << arr.dims();
        out << "Offset: " << arr.offsets();
        out << "Stride: " << arr.strides();
        printer(out, arr.get(), arr, arr.ndims() - 1);
    }

    template<typename T>
    void
    print(const Array<T> &A)
    {
        std::cout << A;
    }

#define INSTANTIATE(T)                          \
    template void print<T> (const Array<T> &A); \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
