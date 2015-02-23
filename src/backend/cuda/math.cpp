#include <math.hpp>

namespace cuda
{
    cfloat division(cfloat lhs, double rhs)
    {
        cfloat retVal;
        retVal.x = real(lhs) / rhs;
        retVal.y = imag(lhs) / rhs;
        return retVal;
    }

    cdouble division(cdouble lhs, double rhs)
    {
        cdouble retVal;
        retVal.x = real(lhs) / rhs;
        retVal.y = imag(lhs) / rhs;
        return retVal;
    }
}
