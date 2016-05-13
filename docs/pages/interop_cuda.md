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
| af::array.device()    | Returns a pointer to the underlying CUDA memory     |
| af::array.lock()      | Takes control of a CUDA pointer from ArrayFire      |
| af::array.unlock()    | Returns control of a CUDA pointer to ArrayFire      |
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

1. Add an include for `af/cuda.h` to your project
2. Obtain a device pointer from ArrayFire af::array objects
3. Determine ArrayFire's CUDA stream
4. Set arguments and launch your kernel in ArrayFire's CUDA stream
5. Return control of af::array memory to ArrayFire
6. Compile your application using `nvcc` with the appropriate paths.

Notice that since ArrayFire and your kernels are sharing the same CUDA
stream, there is no need to perform any synchronization operations as
operations within a stream are executed in order.

This process is best illustrated with a fully worked example:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// 1. Add includes
#include <arrayfire.h>
#include <af/cuda.h>

int main() {

    // Create ArrayFire array objects:
    af::array x = randu(num);
    af::array y = randu(num);

    // ... many ArrayFire operations here

    // Run a custom CUDA kernel in the ArrayFire CUDA stream

    // 2. Obtain device pointers from ArrayFire array objects using
    //    the array::device() function:
    float *d_x = x.device<float>();
    float *d_y = y.device<float>();

    // 3. Determine ArrayFire's CUDA stream
    int af_id = af::getDevice();
    cudaStream_t af_cuda_stream = afcu::getStream(af_id);

    // 4. Set arguments and run your kernel in ArrayFire's stream
    run_custom_kernel<blocks, threads, 0, stream>(d_x, d_y);

    // 5. Return control of af::array memory to ArrayFire using
    //    the array::unlock() function:
    x.unlock();
    y.unlock();

    // ... resume ArrayFire operations

    // Because the device pointers, d_x and d_y, were returned to ArrayFire's
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

1. Add an include for `af/cuda.h` to your project.
2. Instruct ArrayFire to finish operations using af::sync()
3. Obtain a device pointer from ArrayFire af::array objects
4. Determine ArrayFire's CUDA stream using afcu::getStream()
5. Set arguments and launch your kernel in ArrayFire's CUDA stream
6. Ensure CUDA operations have finished using `cudaDeviceSyncronize()`
   or similar commands.
7. Return control of af::array memory to ArrayFire
8. Compile your application using `nvcc` with the appropriate paths.

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
5. Instruct ArrayFire to finish operations using af::sync()
6. Obtain pointers to important memory
7. Continue your CUDA application.
8. Free non-managed memory
9. Compile and link with the appropriate paths and the `-lafcuda` flags.

To create the af::array objects, you should use one of the following
constructors with `src=afDevice`:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
// 1D - 3D af::array constructors
array (dim_t dim0, const T *pointer, af::source src=afHost)
array (dim_t dim0, dim_t dim1, const T *pointer, af::source src=afHost)
array (dim_t dim0, dim_t dim1, dim_t dim2, const T *pointer, af::source src=afHost)
array (dim_t dim0, dim_t dim1, dim_t dim2, dim_t dim3, const T *pointer, af::source src=afHost)

// af::array constructor using a dim4 object
array (const dim4 &dims, const T *pointer, af::source src=afHost)
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

int main() {

    // Create CUDA memory objects
    const int elements = 100;
    size_t size = elements * sizeof(float);
    float *inputSignal;
    cudaMalloc((void**) &inputSignal, size);

    // ... perform many CUDA operations here

    // 2. Finish any pending CUDA operations
    cudaDeviceSynchronize();

    // 3. Create ArrayFire arrays from existing CUDA pointers.
    //    Be sure to specify that the memory type is afDevice.
    af::array d_A(size, inputSignal, afDevice);

    // NOTE: ArrayFire now manages inputSignal

    // 4. Perform operations on the ArrayFire Arrays.

    // For example, add uniformly distributed noise to a signal
    d_A = d_A + randu(elements);

    // NOTE: ArrayFire does not perform the above transaction using
    // in-place memory, thus the pointers containing memory to d_A have
    // likely changed.

    // 5. Instruct ArrayFire to finish pending operations
    af::sync()

    // 6. Get pointers to important memory objects.
    //    Once device is called, ArrayFire will not manage the memory.
    float * outputSignal = d_A.device<float>();

    // 7. continue CUDA application as normal

    // 8. Free non-managed memroy
    //    We removed outputSignal from ArrayFire's control, we need to free it
    cudaFree(outputSignal);

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

