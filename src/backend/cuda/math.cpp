#include <math.hpp>

namespace cuda
{
    cfloat division(cfloat lhs, double rhs)
    {
        float resReal = real(lhs) / rhs;
        float resImag = imag(lhs) / rhs;
        return (cfloat){resReal, resImag};
    }

    cdouble division(cdouble lhs, double rhs)
    {
        double resReal = real(lhs) / rhs;
        double resImag = imag(lhs) / rhs;
        return (cdouble){resReal, resImag};
    }
}
