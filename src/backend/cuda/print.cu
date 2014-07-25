#include <iostream>
#include <Array.hpp>
#include <print.hpp>


namespace cuda
{
    using std::ostream;

    ostream&
    operator<<(ostream &out, const cfloat& var)
    {
        out << var.x << " " << var.y << "i";
        return out;
    }

    ostream&
    operator<<(ostream &out, const cdouble& var)
    {
        out << var.x << " " << var.y << "i";
        return out;
    }

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
            out << std::endl;
        }
        else {
            for(int i = 0; i < d; i++) {
                printer(out, ptr, info, dim - 1);
                ptr += stride;
            }
        }

    }

    template<typename T>
    ostream&
    operator <<(ostream &out, const Array<T> &arr)
    {
        out << "TRANSPOSED\n";
        out << "Dim:" << arr.dims();
        out << "Offset: " << arr.offsets();
        out << "Stride: " << arr.strides();
        printer(out, arr.get(), arr, arr.ndims() - 1);
        return out;
    }
    template<typename T>
    void
    print(const af_array &arr) {
        const Array<T> &impl = getArray<T>(arr);
        std::cout << impl;
    }

    template void print<float>                   (const af_array &arr);
    template void print<cfloat>                  (const af_array &arr);
    template void print<double>                  (const af_array &arr);
    template void print<cdouble>                 (const af_array &arr);
    template void print<char>                    (const af_array &arr);
    template void print<int>                     (const af_array &arr);
    template void print<unsigned>                (const af_array &arr);
    template void print<unsigned char>           (const af_array &arr);
}
