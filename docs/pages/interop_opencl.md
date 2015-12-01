Interoperability with OpenCL {#interop_opencl}
========

As extensive as ArrayFire is, there are a few cases where you are still working
with custom [CUDA] (@ref interop_cuda) or [OpenCL] (@ref interop_opencl) kernels.
For example, you may want to integrate ArrayFire into an existing code base for 
productivity or you may want to keep it around the old implementation for testing
purposes. Arrayfire provides a number of functions that allow it to work alongside 
native OpenCL commands. In this tutorial we are going to talk about how to use 
native OpenCL memory operations and custom OpenCL kernels alongside ArrayFire
in a seamless fashion.

# OpenCL Kernels with Arrayfire arrays
First, we will see how custom OpenCL kernels can be integrated into Arrayfire code.
Let's consider the following code and then break it down bit by bit.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
int main() {
    af::array x = randu(num);
    af::array y = randu(num);

    float *d_x = x.device<float>();
    float *d_y = y.device<float>();

    // Launch kernel to do the following operations
    // y = sin(x)^2 + cos(x)^2
    launch_simple_kernel(d_x, d_y, num);

    x.unlock();
    y.unlock();

    // check for errors, should be 0,
    // since sin(x)^2 + cos(x)^2 == 1
    float err = af::sum<float>(af::abs(y-1));
    printf("Error: %f\n", err);
    return 0;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Breakdown
Most kernels require an input. In this case, we created a random uniform array **x**.
We also go ahead and prepare the output array. The necessary memory required is
allocated in array **y** before the kernel launch.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    af::array x = randu(num);
    af::array y = randu(num);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, the output is the same size as in the input. Note that the actual
output data type is not specified. For such cases, ArrayFire assumes the data type
is single precision floating point ( af::f32 ). If necessary, the data type can
be specified at the end of the array(..) constructor. Once you have the input and
output arrays, you will need to extract the device pointers / objects using 
array::device() method in the following manner.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    float *d_x = x.device<float>();
    float *d_y = y.device<float>();
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Accesing the device pointer in this manner internally sets a flag prohibiting 
the arrayfire object from further managing the memory. Ownership will need to be
returned to the af::array object once we are finished using it.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    // Launch kernel to do the following operations
    // y = sin(x)^2 + cos(x)^2
    launch_simple_kernel(d_x, d_y, num);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The function **launch_simple_kernel** handles the launching of your custom kernel.
We will have a look at the specific functions Arrayfire provides to interface with
OpenCL later in the post. 

Once you have finished your computations, you have to tell ArrayFire to take control
of the memory objects.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    x.unlock();
    y.unlock();
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is a very crucial step as ArrayFire believes the user is still in control
of the pointer. This means that ArrayFire will not perform garbage collection 
on these objects resulting in memory leaks. You can now proceed with the rest of
the program. In our particular example, we are just performing an error check and exiting.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
    // check for errors, should be 0,
    // since sin(x)^2 + cos(x)^2 == 1
    float err = af::sum(af::abs(y-1));
    printf("Error: %f\n", err);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Launching an OpenCL kernel
If you are integrating an OpenCL kernel into your ArrayFire code base you will
need several additional steps to access Arrayfire's internal OpenCL context. 
Once you have access to the same context ArrayFire is using, the rest of the 
process is exactly the same as launching a stand alone OpenCL context.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
void inline launch_simple_kernel(float *d_y,
                                 const float *d_x,
                                 const int num)
{
    std::string simple_kernel_str = CONST_KERNEL_STRING;

    // Get OpenCL context from memory buffer and create a Queue
    cl::Context context(afcl::getContext(true));
    cl::CommandQueue queue(afcl::getQueue(true));

    //Build program and get the required kernel
    cl::Program prog = cl::Program(context, simple_kernel_str, true);
    cl::Kernel  kern = cl::Kernel(prog, "simple_kernel");

    //set global work dimensions
    static const cl::NDRange global(num);

    //prepare argumenst
    kern.setArg(0, d_y);
    kern.setArg(1, d_x);
    kern.setArg(2, num);

    //run kernel
    queue.enqueueNDRangeKernel(kern, cl::NullRange, global);
    queue.finish();

    return;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First of all, to access to OpenCL and the interoperability functions we need to
include the appropriate headers.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
#include <af/opencl.h>
#include <CL/cl.hpp>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The **opencl.h** header includes a number of functions for getting and setting
the context, queue, and device ids used internally in Arrayfire. There are also
a number of methods to construct an af::array from an OpenCL cl_mem buffer object.
There are both C and C++ versions of these functions, and the C++ versions are
wrapped inside the afcl:: namespace. See full datails of these functions in the
[opencl interop documentation.] (\ref opencl_mat)


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
cl::Context context(afcl::getContext(true));
cl::CommandQueue queue(afcl::getQueue(true));
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We start to use these functions by getting Arrayfire's context and queue. For the
C++ api, a **true** flag must be passed for the retain parameter which calls the
clRetainQueue() and clRetainContext() functions before returning. This allows us
to use Arrayfire's internal OpenCL structures inside of the cl::Context and
cl::CommandQueue objects from the C++ api. Once we have them, we can proceed to 
set up and enqueue the kernel like we would in any other OpenCL program. 
The kernel we are using is actually simple and can be seen below.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
std::string CONST_KERNEL_STRING = R"(
__kernel
void simple_kernel(__global float *d_y,
                   __global const float *d_x,
                   const int num)
{
    const int id = get_global_id(0);

    if (id < num) {
        float x = d_x[id];
        float sin_x = sin(x);
        float cos_x = cos(x);
        d_y[id] = (sin_x * sin_x) + (cos_x * cos_x);
    }
}
)";
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Reversing the workflow: Arrayfire arrays from OpenCL Memory

Unfortunately, Arrayfire's interoperability functions don't yet allow us to work with
external OpenCL contexts. This is currently an open issue and can be tracked here:
https://github.com/arrayfire/arrayfire/issues/1002
Once the issue is addressed, it will be possible to take the reverse route and start with
completely custom OpenCL code, then transfer our results into af::array objects.

