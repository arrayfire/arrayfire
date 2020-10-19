Timing Your Code {#timing}
================

timer() : A platform-independent timer with microsecond accuracy:
* [timer::start()](\ref af::timer::start) starts a timer

* [timer::start()](\ref af::timer::stop) seconds since last \ref af::timer::start "start"

* \ref af::timer::stop(af::timer start) "timer::stop(timer start)" seconds since 'start'

Example: single timer

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
   // start timer
   timer::start();
   // run your code
   printf("elapsed seconds: %g\n", timer::stop());
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example: multiple timers

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
   // start timers
   timer start1 = timer::start();
   timer start2 = timer::start();
   // run some code
   printf("elapsed seconds: %g\n", timer::stop(start1));
   // run more code
   printf("elapsed seconds: %g\n", timer::stop(start2));
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accurate and reliable measurement of performance involves several factors:
* Executing enough iterations to achieve peak performance.
* Executing enough repetitions to amortize any overhead from system timers.

To take care of much of this boilerplate, [timeit](\ref af::timeit) provides
accurate and reliable estimates of both CPU or GPU code.

Here`s a stripped down example of
[Monte-Carlo estimation of PI](\ref benchmarks/pi.cpp) making use
of [timeit](\ref af::timeit).  Notice how it expects a `void` function pointer.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
#include <stdio.h>
#include <arrayfire.h>
using namespace af;

void pi_function() {
  int n = 20e6; // 20 million random samples
  array x = randu(n,f32), y = randu(n,f32);
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
