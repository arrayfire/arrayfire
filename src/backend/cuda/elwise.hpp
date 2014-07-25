#include <af/defines.h>
#include <af/array.h>

namespace cuda
{
typedef void(*binaryOp)(af_array*, af_array, af_array);
binaryOp getFunction(af_dtype lhs, af_dtype rhs);
}
