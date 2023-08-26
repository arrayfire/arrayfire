Timing ArrayFire Code {#timing}
================

In performance-sensitive applications, it is vital to profile and measure the
execution time of operations. ArrayFire provides mechanisms to achieve this.

ArrayFire employs an asynchronous evaluation model for all of its
functions. This means that operations are queued to execute but do not
necessarily complete prior to function return. Hence, directly measuring the
time taken for an ArrayFire function could be misleading. To accurately
measure time, one must ensure the operations are evaluated and synchronize the
ArrayFire stream.

ArrayFire also employs a lazy evaluation model for its elementwise arithmetic
operations. This means operations are not queued for execution until the
result is needed by downstream operations blocking until the operations are
complete.

The following describes how to time ArrayFire code using the eval and sync
functions along with the timer and timeit functions. A final note on kernel
caching also provides helpful details about ArrayFire runtimes.

## Using ArrayFire eval and sync functions

ArrayFire provides functions to force the evaluation of lazy functions and to
block until all asynchoronous operations complete.

1. The [eval](\ref af::eval) function:

   Forces the evaluation of an ArrayFire array. It ensures the execution of
   operations queued up for a specific array.

   It is only required for timing purposes if elementwise arithmetic functions
   are called on the array, since these are handled by the ArrayFire JIT.

   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
   af::array A = af::randu(1000, 1000);
   af::array B = A + A;                 // Elementwise arithmetic operation.
   B.eval();                            // Forces evaluation of B.
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The function initializes the evaluation of the JIT-tree for that array and
   may return prior to the completion of those operations. To ensure proper
   timing, combine with a [sync](\ref af::sync) function.

2. The [sync](\ref af::sync) function:

   Synchronizes the ArrayFire stream. It waits for all the previous operations
   in the stream to finish. It is often used after [eval](\ref af::eval) to
   ensure that operations have indeed been completed.

   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
   af::sync();  // Waits for all previous operations to complete.
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Using ArrayFire timer and timeit functions

ArrayFire provides a simple timer functions that returns the current time in
seconds.

1. The [timer](\ref af::timer) function:

   timer() : A platform-independent timer with microsecond accuracy:
   * [timer::start()](\ref af::timer::start) starts a timer

   * [timer::start()](\ref af::timer::stop) seconds since last \ref
     af::timer::start "start"

   * \ref af::timer::stop(af::timer start) "timer::stop(timer start)" seconds
     since 'start'

   Example: single timer

   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
       // start timer
       // - be sure to use the eval and sync functions so that previous code
       //   does not get timed as part of the execution segment being measured
       timer::start();
       // run a code segment
       // - be sure to use the eval and sync functions to ensure the code
       //   segment operations have been completed
       // stop timer
       printf("elapsed seconds: %g\n", timer::stop());
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   Example: multiple timers

   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
       // start timers
       // - be sure to use the eval and sync functions so that previous code
       //   does not get timed as part of the execution segment being measured
       timer start1 = timer::start();
       timer start2 = timer::start();
       // run a code segment
       // - be sure to use the eval and sync functions to ensure the code
       //   segment operations have been completed
       // stop timer1
       printf("elapsed seconds: %g\n", timer::stop(start1));
       // run another code segment
       // - be sure to use the eval and sync functions to ensure the code
       //   segment operations have been completed
       // stop timer2
       printf("elapsed seconds: %g\n", timer::stop(start2));
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   Accurate and reliable measurement of performance involves several factors:
   * Executing enough iterations to achieve peak performance.
   * Executing enough repetitions to amortize any overhead from system timers.

2. The [timeit](\ref af::timeit) function:

   To take care of much of this boilerplate, [timeit](\ref af::timeit) provides
   accurate and reliable estimates of both CPU or GPU code.

   Here is a stripped down example of [Monte-Carlo estimation of PI](\ref
   benchmarks/pi.cpp) making use of [timeit](\ref af::timeit). Notice how it
   expects a `void` function pointer.

   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
   #include <stdio.h>
   #include <arrayfire.h>
   using namespace af;

   void pi_function() {
     int n = 20e6; // 20 million random samples
     array x = randu(n, f32), y = randu(n, f32);
     // how many fell inside unit circle?
     float pi = 4.0 * sum<float>(sqrt(x*x + y*y)) < 1) / n;
   }

   int main() {
     printf("pi_function took %g seconds\n", timeit(pi_function));
     return 0;
   }
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This produces:

       pi_function took 0.007252 seconds
       (test machine: Core i7 920 @ 2.67GHz with a Tesla C2070)


## A note on kernel caching

The first run of ArrayFire code exercises any JIT compilation in the
application, automatically saving a cache of the compilation to
disk. Subsequent runs load the cache from disk, executing without
compilation. Therefore, it is typically best to "warm up" the code with one
run to initiate the application's kernel cache. Afterwards, subsequent runs do
not include the compile time and are tend to be faster than the first run.

Averaging the time taken is always the best approach and one reason why the
[timeit](\ref af::timeit) function is helpful.
