Interoperability with CUDA {#interop_cuda}
========

Although ArrayFire is quite extensive, there remain many cases in which you
may want to write custom kernels in CUDA or [OpenCL](\ref interop_opencl).
For example, you may wish to add ArrayFire to an existing code base to increase
your productivity, or you may need to supplement ArrayFire's functionality
with your own custom implementation of specific algorithms.

ArrayFire manages its own memory, runs within its own CUDA stream, and
creates custom IDs for devices. As such, most of the interoperability functions
focus on reducing potential synchronization conflicts between ArrayFire and CUDA.

# Basics

It is fairly straightforward to interface ArrayFire with your own custom CUDA
code. ArrayFire provides several functions to ease this process including:

| Function              | Purpose                                             |
|-----------------------|-----------------------------------------------------|
| af::array(...)        | Construct an ArrayFire Array from device memory     |
| af::array.device()    | Obtain a pointer to the device memory (implies `lock()`) |
| af::array.lock()      | Removes ArrayFire's control of a device memory pointer |
| af::array.unlock()    | Restores ArrayFire's control over a device memory pointer |
| af::getDevice()       | Gets the current ArrayFire device ID                |
| af::setDevice()       | Switches ArrayFire to the specified device          |
| afcu::getNativeId()   | Converts an ArrayFire device ID to a CUDA device ID |
| afcu::setNativeId()   | Switches ArrayFire to the specified CUDA device ID  |
| afcu::getStream()     | Get the current CUDA stream used by ArrayFire       |


Below we provide two worked examples on how ArrayFire can be integrated
into new and existing projects.

# Adding custom CUDA kernels to an existing ArrayFire application

By default, ArrayFire manages its own memory and operates in its own CUDA
stream. Thus there is a slight amount of bookkeeping that needs to be done
in order to integrate your custom CUDA kernel.

If your kernels can share the ArrayFire CUDA stream, you should:

1. Include the 'af/afcuda.h' header in your source code
2. Use ArrayFire as normal
3. Ensure any JIT kernels have executed using `af::eval()`
4. Obtain device pointers from ArrayFire array objects using `array::device()`
5. Determine ArrayFire's CUDA stream
6. Set arguments and run your kernel in ArrayFire's stream
7. Return control of af::array memory to ArrayFire
8. Compile with `nvcc`, linking with the `afcuda` library.

Notice that since ArrayFire and your kernels are sharing the same CUDA
stream, there is no need to perform any synchronization operations as
operations within a stream are executed in order.

This process is best illustrated with a fully worked example:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// 1. Add includes
#include <arrayfire.h>
#include <af/cuda.h>

int main() {

    // 2. Use ArrayFire as normal
    size_t num = 10;
    af::array x = af::constant(0, num);

    // ... many ArrayFire operaitons here

    // 3. Ensure any JIT kernels have executed
    x.eval();
    af_print(x);

    // Run a custom CUDA kernel in the ArrayFire CUDA stream

    // 4. Obtain device pointers from ArrayFire array objects using
    //    the array::device() function:
    float *d_x = x.device<float>();

    // 5. Determine ArrayFire's CUDA stream
    int af_id = af::getDevice();
    int cuda_id = afcu::getNativeId(af_id);
    cudaStream_t af_cuda_stream = afcu::getStream(cuda_id);

    // 6. Set arguments and run your kernel in ArrayFire's stream
    //    Here launch with 1 block of 10 threads
    increment<<<1, num, 0, af_cuda_stream>>>(d_x);

    // 7. Return control of af::array memory to ArrayFire using
    //    the array::unlock() function:
    x.unlock();

    // ... resume ArrayFire operations
    af_print(x);

    // Because the device pointer `d_x` was returned to ArrayFire's
    // control by the unlock function, there is no need to free them using
    // cudaFree()

    return 0;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your kernels needs to operate in their own CUDA stream, the process is
essentially identical, except you need to instruct ArrayFire to complete
its computations using the af::sync() function prior to launching your
own kernel and ensure your kernels are complete using `cudaDeviceSynchronize()`
(or similar) commands prior to returning control of the memory to ArrayFire:

1. Include the 'af/afcuda.h' header in your source code
2. Use ArrayFire as normal
3. Ensure any JIT kernels have executed using `af::eval()`
4. Instruct ArrayFire to finish operations using af::sync()
5. Obtain device pointers from ArrayFire array objects using
6. Determine ArrayFire's CUDA stream
7. Set arguments and run your kernel in your custom stream
8. Ensure CUDA operations have finished using `cudaDeviceSyncronize()`
   or similar commands.
9. Return control of af::array memory to ArrayFire
10. Compile with `nvcc`, linking with the `afcuda` library.

# Adding ArrayFire to an existing CUDA application

Adding ArrayFire to an existing CUDA application is slightly more involved
and can be somewhat tricky due to several optimizations we implement. The
most important are as follows:

* ArrayFire assumes control of all memory provided to it.
* ArrayFire does not (in general) support in-place memory transactions.

We will discuss the implications of these items below. To add ArrayFire
to existing code you need to:

1. Include `arrayfire.h` and `af/cuda.h` in your source file
2. Finish any pending CUDA operations
   (e.g. use cudaDeviceSynchronize() or similar stream functions)
3. Create ArrayFire arrays from existing CUDA pointers
4. Perform operations on ArrayFire arrays
5. Instruct ArrayFire to finish operations using af::eval() and af::sync()
6. Obtain pointers to important memory
7. Continue your CUDA application.
8. Free non-managed memory
9. Compile and link with the appropriate paths and the `-lafcuda` flags.

To create the af::array objects, you should use one of the following
constructors with `src=afDevice`:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// 1D - 3D af::array constructors
af::array (dim_t dim0, const T *pointer, af::source src=afHost)
af::array (dim_t dim0, dim_t dim1, const T *pointer, af::source src=afHost)
af::array (dim_t dim0, dim_t dim1, dim_t dim2, const T *pointer, af::source src=afHost)
af::array (dim_t dim0, dim_t dim1, dim_t dim2, dim_t dim3, const T *pointer, af::source src=afHost)

// af::array constructor using a dim4 object
af::array (const dim4 &dims, const T *pointer, af::source src=afHost)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*NOTE*: With all of these constructors, ArrayFire's memory manager automatically
assumes responsibility for any memory provided to it. Thus ArrayFire could free
or reuse the memory at any later time. If this behavior is not desired, you
may call `array::unlock()` and manage the memory yourself. However, if you do
so, please be cautious not to free memory when ArrayFire might be using it!

The seven steps above are best illustrated using a fully-worked example:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// 1. Add includes
#include <arrayfire.h>
#include <af/cuda.h>

using namespace std;

int main() {

    // Create and populate CUDA memory objects
    const int elements = 100;
    size_t size = elements * sizeof(float);
    float *cuda_A;
    cudaMalloc((void**) &cuda_A, size);

    // ... perform many CUDA operations here

    // 2. Finish any pending CUDA operations
    cudaDeviceSynchronize();

    // 3. Create ArrayFire arrays from existing CUDA pointers.
    //    Be sure to specify that the memory type is afDevice.
    af::array d_A(elements, cuda_A, afDevice);

    // NOTE: ArrayFire now manages cuda_A

    // 4. Perform operations on the ArrayFire Arrays.
    d_A = d_A * 2;

    // NOTE: ArrayFire does not perform the above transaction using
    // in-place memory, thus the pointers containing memory to d_A have
    // likely changed.

    // 5. Instruct ArrayFire to finish pending operations using eval and sync.
    af::eval(d_A);
    af::sync();

    // 6. Get pointers to important memory objects.
    //    Once device is called, ArrayFire will not manage the memory.
    float * outputValue = d_A.device<float>();

    // 7. continue CUDA application as normal

    // 8. Free non-managed memory
    //    We removed outputValue from ArrayFire's control, we need to free it
    cudaFree(outputValue);

    return 0;
}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Using multiple devices

If you are using multiple devices with ArrayFire and CUDA kernels, there is
one "gotcha" of which you should be aware. ArrayFire implements its own internal
order of compute devices, thus a CUDA device ID may not be the same as an
ArrayFire device ID. Thus when switching between devices it is important
that you use our interoperability functions to get/set the correct device
IDs. Below is a quick listing of the various functions needed to switch
between devices along with some disambiguation as to the device identifiers
used with each function:

| Function            | ID Type     | Purpose                                 |
|---------------------|-------------|-----------------------------------------|
| cudaGetDevice()     | CUDA        | Gets the current CUDA device ID         |
| cudaSetDevice()     | CUDA        |Sets the current CUDA device             |
| af::getDevice()     | AF          | Gets the current ArrayFire device ID    |
| af::setDevice()     | AF          | Sets the current ArrayFire device       |
| afcu::getNativeId() | AF -> CUDA  | Convert an ArrayFire device ID to a CUDA device ID |
| afcu::setNativeId() | CUDA -> AF  |Set the current ArrayFire device from a CUDA ID |

