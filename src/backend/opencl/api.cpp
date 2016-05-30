#include <af/array.h>
#include <af/opencl.h>

namespace af {
    template<> AFAPI cl_mem *array::device() const
    {
        cl_mem *mem_ptr = new cl_mem;
        af_err err = af_get_device_ptr((void **)mem_ptr, get());
        if (err != AF_SUCCESS) throw af::exception("Failed to get cl_mem from array object");
        return mem_ptr;
    }
}
