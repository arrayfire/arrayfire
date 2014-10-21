#include <Array.hpp>
#include <optypes.hpp>
#include <math.hpp>
#include <err_cuda.hpp>

namespace cuda
{

    template<typename T, af_op_t op>
    Array<T>* unaryOp(const Array<T> &in)
    {
        CUDA_NOT_SUPPORTED();
    }

}
