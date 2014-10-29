#include <af/array.h>
#include <af/blas.h>
#include "error.hpp"

namespace af
{
    array matmul(const array &lhs, const array &rhs,
                 af_blas_transpose optLhs, af_blas_transpose optRhs)
    {
        af_array out = 0;
        AF_THROW(af_matmul(&out, lhs.get(), rhs.get(), optLhs, optRhs));
        return array(out);
    }

    array dot   (const array &lhs, const array &rhs,
                 af_blas_transpose optLhs, af_blas_transpose optRhs)
    {
        af_array out = 0;
        AF_THROW(af_dot(&out, lhs.get(), rhs.get(), optLhs, optRhs));
        return array(out);
    }
}
