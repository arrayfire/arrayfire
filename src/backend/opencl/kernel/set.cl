#if T == double || U == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

kernel
void
set(    global  T*      ptr,
                T       val,
        const   unsigned long  elements)
{
    if(get_global_id(0) < elements) {
        ptr[get_global_id(0)] = val;
    }
}

