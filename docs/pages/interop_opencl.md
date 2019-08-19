Interoperability with OpenCL {#interop_opencl}
========

Although ArrayFire is quite extensive, there remain many cases in which you
may want to write custom kernels in OpenCL or [CUDA](\ref interop_cuda).
For example, you may wish to add ArrayFire to an existing code base to increase
your productivity, or you may need to supplement ArrayFire's functionality
with your own custom implementation of specific algorithms.

ArrayFire manages its own context, queue, memory, and creates custom IDs
for devices. As such, most of the interoperability functions focus on reducing
potential synchronization conflicts between ArrayFire and OpenCL.

# Basics

It is fairly straightforward to interface ArrayFire with your own custom OpenCL
code. ArrayFire provides several functions to ease this process including:

| Function              | Purpose                                             |
|-----------------------|-----------------------------------------------------|
| af::array(...)        | Construct an ArrayFire array from cl_mem references or cl::Buffer objects |
| af::array.device()    | Obtain a pointer to the cl_mem reference (implies `lock()`) |
| af::array.lock()      | Removes ArrayFire's control of a cl_mem buffer            |
| af::array.unlock()    | Restores ArrayFire's control over a cl_mem buffer         |
| afcl::getPlatform()   | Get ArrayFire's current cl_platform                       |
| af::getDevice()       | Get the current ArrayFire Device ID                       |
| afcl::getDeviceId()   | Get ArrayFire's current cl_device_id                      |
| af::setDevice()       | Set ArrayFire's device from an ArrayFire device ID        |
| afcl::setDeviceId()   | Set ArrayFire's device from a cl_device_id                |
| afcl::setDevice()     | Set ArrayFire's device from a cl_device_id and cl_context |
| afcl::getContext()    | Get ArrayFire's current cl_context                        |
| afcl::getQueue()      | Get ArrayFire's current cl_command_queue                  |
| afcl::getDeviceType() | Get the current afcl_device_type                          |

Additionally, the OpenCL backend permits the programmer to add and remove custom
devices from the ArrayFire device manager. These permit you to attach ArrayFire
directly to the OpenCL queue used by other portions of your application.

| Function              | Purpose                                           |
|-----------------------|---------------------------------------------------|
| afcl::addDevice()     | Add a new device to ArrayFire's device manager    |
| afcl::deleteDevice()  | Remove a device from ArrayFire's device manager   |

Below we provide two worked examples on how ArrayFire can be integrated
into new and existing projects.

# Adding custom OpenCL kernels to an existing ArrayFire application

By default, ArrayFire manages its own context, queue, memory, and creates custom
IDs for devices. Thus there is some bookkeeping that needs to be done to
integrate your custom OpenCL kernel.

If your kernels can share operate in the same queue as ArrayFire, you should:

1. Add an include for `af/opencl.h` to your project
2. Obtain the OpenCL context, device, and queue used by ArrayFire
3. Obtain cl_mem references to af::array objects
4. Load, build, and use your kernels
5. Return control of af::array memory to ArrayFire

Note, ArrayFire uses an in-order queue, thus when ArrayFire and your kernels
are operating in the same queue, there is no need to perform any
synchronization operations.

This process is best illustrated with a fully worked example:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
#include <arrayfire.h>
// 1. Add the af/opencl.h include to your project
#include <af/opencl.h>

int main() {
    size_t length = 10;

    // Create ArrayFire array objects:
    af::array A = af::randu(length, f32);
    af::array B = af::constant(0, length, f32);

    // ... additional ArrayFire operations here

    // 2. Obtain the device, context, and queue used by ArrayFire
    static cl_context af_context = afcl::getContext();
    static cl_device_id af_device_id = afcl::getDeviceId();
    static cl_command_queue af_queue = afcl::getQueue();

    // 3. Obtain cl_mem references to af::array objects
    cl_mem * d_A = A.device<cl_mem>();
    cl_mem * d_B = B.device<cl_mem>();

    // 4. Load, build, and use your kernels.
    //    For the sake of readability, we have omitted error checking.
    int status = CL_SUCCESS;

    // A simple copy kernel, uses C++11 syntax for multi-line strings.
    const char * kernel_name = "copy_kernel";
    const char * source = R"(
        void __kernel
        copy_kernel(__global float * gA, __global float * gB)
        {
            int id = get_global_id(0);
            gB[id] = gA[id];
        }
    )";

    // Create the program, build the executable, and extract the entry point
    // for the kernel.
    cl_program program = clCreateProgramWithSource(af_context, 1, &source, NULL, &status);
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, kernel_name, &status);

    // Set arguments and launch your kernels
    clSetKernelArg(kernel, 0, sizeof(cl_mem), d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), d_B);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, NULL, &length, NULL, 0, NULL, NULL);

    // 5. Return control of af::array memory to ArrayFire
    A.unlock();
    B.unlock();

    // ... resume ArrayFire operations

    // Because the device pointers, d_x and d_y, were returned to ArrayFire's
    // control by the unlock function, there is no need to free them using
    // clReleaseMemObject()

    return 0;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your kernels needs to operate in their own OpenCL queue, the process is
essentially identical, except you need to instruct ArrayFire to complete
its computations using the af::sync() function prior to launching your
own kernel and ensure your kernels are complete using `clFinish`
(or similar) commands prior to returning control of the memory to ArrayFire:

1. Add an include for `af/opencl.h` to your project
2. Obtain the OpenCL context, device, and queue used by ArrayFire
3. Obtain cl_mem references to af::array objects
4. Instruct ArrayFire to finish operations using af::sync()
5. Load, build, and use your kernels
6. Instruct OpenCL to finish operations using clFinish() or similar commands.
5. Return control of af::array memory to ArrayFire

# Adding ArrayFire to an existing OpenCL application

Adding ArrayFire to an existing OpenCL application is slightly more involved
and can be somewhat tricky due to several optimizations we implement. The
most important are as follows:

* ArrayFire assumes control of all memory provided to it.
* ArrayFire does not (in general) support in-place memory transactions.

We will discuss the implications of these items below. To add ArrayFire
to existing code you need to:

1. Add includes
2. Instruct OpenCL to complete its operations using clFinish (or similar)
3. Instruct ArrayFire to use the user-created OpenCL Context
4. Create ArrayFire arrays from OpenCL memory objects
5. Perform ArrayFire operations on the Arrays
6. Instruct ArrayFire to finish operations using af::sync()
7. Obtain cl_mem references for important memory
8. Continue your OpenCL application

To create the af::array objects, you should use one of the following
constructors:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// 1D - 3D af::array constructors
static af::array    array (dim_t dim0, cl_mem buf, af::dtype type, bool retain=false)
static af::array    array (dim_t dim0, dim_t dim1, cl_mem buf, af::dtype type, bool retain=false)
static af::array    array (dim_t dim0, dim_t dim1, dim_t dim2, cl_mem buf, af::dtype type, bool retain=false)
static af::array    array (dim_t dim0, dim_t dim1, dim_t dim2, dim_t dim3, cl_mem buf, af::dtype type, bool retain=false)

// af::array constructor using a dim4 object
static af::array    array (af::dim4 idims, cl_mem buf, af::dtype type, bool retain=false)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*NOTE*: With all of these constructors, ArrayFire's memory manager automatically
assumes responsibility for any memory provided to it. If you are creating
an array from a `cl::Buffer`, you should specify `retain=true` to ensure your
memory is not deallocated if your `cl::Buffer` were to go out of scope.
We use this technique in the example below.
If you do not wish for ArrayFire to manage your memory, you may call the
`array::unlock()` function and manage the memory yourself; however, if you do
so, please be cautious not to call `clReleaseMemObj` on a `cl_mem`  when
ArrayFire might be using it!

The eight steps above are best illustrated using a fully-worked example. Below we
use the OpenCL 2.0 C++ API and omit error checking to keep the code readable. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
#include <CL/cl2.hpp>

// 1. Add arrayfire.h and af/opencl.h to your application
#include "arrayfire.h"
#include "af/opencl.h"

#include <cstdio>
#include <vector>

int main() {

  // Set up the OpenCL context, device, and queues
  cl::Context context(CL_DEVICE_TYPE_ALL);
  vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Device device = devices[0];
  cl::CommandQueue queue(context, device);

  // Create a buffer of size 10 filled with ones, copy it to the device
  int length = 10;
  vector<float> h_A(length, 1);
  cl::Buffer cl_A(context, CL_MEM_READ_WRITE, length * sizeof(float), h_A.data());

  // 2. Instruct OpenCL to complete its operations using clFinish (or similar)
  queue.finish();

  // 3. Instruct ArrayFire to use the user-created context
  //    First, create a device from the current OpenCL device + context + queue
  afcl::addDevice(device(), context(), queue());
  //    Next switch ArrayFire to the device using the device and context as 
  //    identifiers:
  afcl::setDevice(device(), context());

  // 4. Create ArrayFire arrays from OpenCL memory objects
  af::array af_A = afcl::array(length, cl_A(), f32, true);

  // 5. Perform ArrayFire operations on the Arrays
  af_A = af_A + af::randu(length);
    
  // NOTE: ArrayFire does not perform the above transaction using in-place memory, 
  // thus the underlying OpenCL buffers containing the memory containing memory to 
  // probably have changed

  // 6. Instruct ArrayFire to finish operations using af::sync
  af::sync();

  // 7. Obtain cl_mem references for important memory
  cl_A = *af_A.device<cl_mem>();

  // 8. Continue your OpenCL application

  // ... 
  
  return 0;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Using multiple devices

If you are using ArrayFire and OpenCL with multiple devices be sure to use
`afcl::addDevice` to add your custom context + device + queue to ArrayFire's
device manager. This will let you switch ArrayFire devices using your current
`cl_device_id` and `cl_context`.
