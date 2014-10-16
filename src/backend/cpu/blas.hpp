#include <af/defines.h>
#include <af/blas.h>
#include <Array.hpp>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include <cblas.h>
}
#endif

namespace cpu
{

template<typename T>
Array<T>* matmul(const Array<T> &lhs, const Array<T> &rhs,
                    af_blas_transpose optLhs, af_blas_transpose optRhs);
template<typename T>
Array<T>* dot(const Array<T> &lhs, const Array<T> &rhs,
                    af_blas_transpose optLhs, af_blas_transpose optRhs);

}
