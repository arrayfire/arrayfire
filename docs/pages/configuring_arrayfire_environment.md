Configuring ArrayFire Environment {#configuring_environment}
===============================================================================

This page lists environment and runtime configurations that will help enhance
your experience with ArrayFire.

[TOC]

Environment Variables{#environment_variables}
===============================================================================

The following are useful environment variable that can be used with ArrayFire.

AF_PATH {#af_path}
-------------------------------------------------------------------------------

This is the path with ArrayFire gets installed, ie. the includes and libs are
present in this directory. You can use this variable to add include paths and
libraries to your projects.

AF_PRINT_ERRORS {#af_print_errors}
-------------------------------------------------------------------------------

When AF_PRINT_ERRORS is set to 1, the exceptions thrown are more verbose and
detailed. This helps in locating the exact failure.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AF_PRINT_ERRORS=1 ./myprogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AF_CUDA_DEFAULT_DEVICE {#af_cuda_default_device}
-------------------------------------------------------------------------------

Use this variable to set the default CUDA device. Valid values for this
variable are the device identifiers shown when af::info is run.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AF_CUDA_DEFAULT_DEVICE=1 ./myprogram_cuda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note: af::setDevice call in the source code will take precedence over this
variable.

AF_OPENCL_DEFAULT_DEVICE {#af_opencl_default_device}
-------------------------------------------------------------------------------

Use this variable to set the default OpenCL device. Valid values for this
variable are the device identifiers shown when af::info is run.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AF_OPENCL_DEFAULT_DEVICE=1 ./myprogram_opencl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note: af::setDevice call in the source code will take precedence over this
variable.

AF_OPENCL_DEFAULT_DEVICE_TYPE {#af_opencl_default_device_type}
-------------------------------------------------------------------------------

Use this variable to set the default OpenCL device type. Valid values for this
variable are: CPU, GPU, ACC (Accelerators).

When set, the first device of the specified type is chosen as default device.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AF_OPENCL_DEFAULT_DEVICE_TYPE=CPU ./myprogram_opencl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note: `AF_OPENCL_DEFAULT_DEVICE` and af::setDevice takes precedence over this variable.

AF_OPENCL_DEVICE_TYPE {#af_opencl_device_type}
-------------------------------------------------------------------------------

Use this variable to only choose OpenCL devices of specified type. Valid values for this
variable are:

- ALL: All OpenCL devices. (Default behavior).
- CPU: CPU devices only.
- GPU: GPU devices only.
- ACC: Accelerator devices only.

When set, the remaining OpenCL device types are ignored by the OpenCL backend.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AF_OPENCL_DEVICE_TYPE=CPU ./myprogram_opencl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AF_OPENCL_CPU_OFFLOAD {#af_opencl_cpu_offload}
-------------------------------------------------------------------------------

When ArrayFire runs on devices with unified memory with the host (ie.
`CL_DEVICE_HOST_UNIFIED_MENORY` is true for the device) then certain functions
are offloaded to run on the CPU using mapped buffers.

ArrayFire takes advantage of fast libraries such as MKL while spending no time
copying memory from device to host. The device memory is mapped to a host
pointer which can be used in the offloaded functions.

This functionality can be disabled by using the environment variable
`AF_OPENCL_CPU_OFFLOAD=0`.

The default bevaior of this has changed in version 3.4.

Prior to v3.4, CPU Offload functionality was used only when the user set
`AF_OPENCL_CPU_OFFLOAD=1` and disabled otherwise.

From v3.4 onwards, CPU Offload is enabled by default and is disabled only when
`AF_OPENCL_CPU_OFFLOAD=0` is set.

AF_OPENCL_SHOW_BUILD_INFO {#af_opencl_show_build_info}
-------------------------------------------------------------------------------

This variable is useful when debuggin OpenCL kernel compilation failures. When
this variable is set to 1, and an error occurs during a OpenCL kernel
compilation, then the log and kernel are printed to screen.

AF_DISABLE_GRAPHICS {#af_disable_graphics}
-------------------------------------------------------------------------------

Setting this variable to 1 will disable window creation when graphics
functions are being called. Disabling window creation will disable all other
graphics calls at runtime as well.

This is a useful enviornment variable when running code on servers and systems
without displays. When graphics calls are run on such machines, they will
print warning about window creation failing. To suppress those calls, set this
variable.

AF_SYNCHRONOUS_CALLS {#af_synchronous_calls}
-------------------------------------------------------------------------------

When this environment variable is set to 1, ArrayFire will execute all
functions synchronously.

AF_SHOW_LOAD_PATH {#af_show_load_path}
-------------------------------------------------------------------------------

When using the Unified backend, if this variable is set to 1, it will show the
path where the ArrayFire backend libraries are loaded from.

If the libraries are loaded from system paths, such as PATH or LD_LIBRARY_PATH
etc, then it will print "system path". If the libraries are loaded from other
paths, then those paths are shown in full.

AF_MEM_DEBUG {#af_mem_debug}
-------------------------------------------------------------------------------

When AF_MEM_DEBUG is set to 1 (or anything not equal to 0), the caching mechanism in the memory manager is disabled.
The device buffers are allocated using native functions as needed and freed when going out of scope.

When the environment variable is not set, it is treated to be zero.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AF_MEM_DEBUG=1 ./myprogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


AF_MAX_BUFFERS {#af_max_buffers}
-------------------------------------------------------------------------

When AF_MAX_BUFFERS is set, this environment variable specifies the maximum number of buffers allocated before garbage collection kicks in.

Please note that the total number of buffers that can exist simultaneously can be higher than this number. This variable tells the garbage collector that it should free any available buffers immediately if the treshold is reached.

When not set, the default value is 1000.

AF_OPENCL_MAX_JIT_LEN {#af_opencl_max_jit_len}
-------------------------------------------------------------------------------

When set, this environment variable specifies the maximum height of the OpenCL JIT tree after which evaluation is forced.

The default value, as of v3.4, is 50 on OSX, 100 everywhere else. This value was 20 for older versions.

AF_CUDA_MAX_JIT_LEN {#af_cuda_max_jit_len}
-------------------------------------------------------------------------------

When set, this environment variable specifies the maximum height of the CUDA JIT tree after which evaluation is forced.

The default value, as of v3.4, 100. This value was 20 for older versions.

AF_CPU_MAX_JIT_LEN {#af_cpu_max_jit_len}
-------------------------------------------------------------------------------

When set, this environment variable specifies the maximum length of the CPU JIT tree after which evaluation is forced.

The default value, as of v3.4, 100. This value was 20 for older versions.
