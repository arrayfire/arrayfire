Debugging ArrayFire Issues {#debugging}
===============================================================================

Some general advice for debugging mysterious issues with ArrayFire follows.

First, try running with additional debugging information:

    export AF_PRINT_ERRORS=1
    export AF_TRACE=all         #OR export AF_TRACE=mem
    ./my_program

Next, add some debugging output to your code.

C++:

    af_print_mem_info("message", -1); //-1 is the active device; otherwise, an integer specifies a device

Python:

    arrayfire.device.print_mem_info("message")

See the [ArrayFire README](https://github.com/arrayfire/arrayfire) for support information.