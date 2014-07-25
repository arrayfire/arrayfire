#if T == double || U == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif


kernel
void
binaryOp(   global  R*      out,
            global  T*      lhs,
            global  U*      rhs,
            const   unsigned long elements)
{
    size_t idx = get_global_id(0);
    if(idx < elements) {
        out[idx] = lhs[idx] OP rhs[idx];

    }
}

