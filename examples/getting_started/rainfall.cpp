/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Investigate rainfall measurements across sites and days
// demonstrating various simple tasks

// Compute various values:
// - total rainfall at each site
// - rain between days 1 and 5
// - number of days with rain
// - total rainfall on each day
// - number of days with over five inches
// - total rainfall at each site

// note: example adapted from
//  "Rapid Problem Solving Using Thrust", Nathan Bell, NVIDIA

#include <arrayfire.h>
#include <af/util.h>
#include <stdio.h>
#include <cstdlib>
using namespace af;

int main(int argc, char **argv)
{
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        int days = 9, sites = 4;
        int n = 10; // measurements
        float day_[] =         {0, 0, 1, 2, 5, 5, 6, 6, 7, 8 }; // ascending
        float site_[] =        {2, 3, 0, 1, 1, 2, 0, 1, 2, 1 };
        float measurement_[] = {9, 5, 6, 3, 3, 8, 2, 6, 5, 10}; // inches
        array day(n,day_);
        array site(n,site_);
        array measurement(n,measurement_);

        array rainfall = constant(0, sites);
        gfor (seq s, sites) {
            rainfall(s) = sum(measurement * (site == s));
        }

        printf("total rainfall at each site:\n");
        af_print(rainfall);

        array is_between = 1 <= day && day <= 5; // days 1 and 5
        float rain_between = sum<float>(measurement * is_between);
        printf("rain between days: %g\n", rain_between);

        printf("number of days with rain: %g\n", sum<float>(diff1(day) > 0) + 1);

        array per_day = constant(0, days);
        gfor (seq d, days)
            per_day(d) = sum(measurement * (day == d));

        printf("total rainfall each day:\n");
        af_print(per_day);

        printf("number of days over five: %g\n", sum<float>(per_day > 5));
    } catch (af::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    #ifdef WIN32 // pause in Windows
    if (!(argc == 2 && argv[1][0] == '-')) {
        std::cout << "hit [enter]...";
        fflush(stdout);
        getchar();
    }
    #endif
    return 0;
}
