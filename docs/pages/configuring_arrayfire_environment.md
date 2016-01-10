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

Note: `AF_OPENCL_DEFAULT_DEVICE`  and af::setDevice takes precedence over this variable.

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

AF_DISABLE_GRAPHICS {#af_disable_graphics}
-------------------------------------------------------------------------------

Setting this variable will disable window creation when graphics functions are
being called. Simply setting this variable will disable functionality, any
value will suffice. Disabling window creation will disable all other graphics
calls at runtime as well.

This is a useful enviornment variable when running code on servers and systems
without displays. When graphics calls are run on such machines, they will
print warning about window creation failing. To suppress those calls, set this
variable.

AF_PRINT_ERRORS {#af_print_errors}
-------------------------------------------------------------------------------

When AF_PRINT_ERRORS is set to 1, the exceptions thrown are more verbose and
detailed. This helps in locating the exact failure.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AF_PRINT_ERRORS=1 ./myprogram_opencl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
