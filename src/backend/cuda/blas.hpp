#include <af/defines.h>
#include <af/blas.h>
#include <Array.hpp>

namespace cuda
{

template<typename T>
Array<T>* matmul(const Array<T> &lhs, const Array<T> &rhs,
                    af_blas_transpose optLhs, af_blas_transpose optRhs);
template<typename T>
Array<T>* dot(const Array<T> &lhs, const Array<T> &rhs,
                    af_blas_transpose optLhs, af_blas_transpose optRhs);

}
