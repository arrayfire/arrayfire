#include <af/defines.h>
#include <af/array.h>

namespace opencl
{
typedef void(*binaryOp)(af_array*, af_array, af_array);
binaryOp getFunction(af_dtype lhs, af_dtype rhs);
}
