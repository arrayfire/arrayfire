/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <iostream>
#include <cstdio>

using namespace af;
using namespace std;

static const int width = 512, height = 512;
static const int pixels_per_unit = 20;

af::array p_x;
af::array p_y;
af::array vels_x;
af::array vels_y;
af::array forces_x;
af::array forces_y;

void simulate(float dt){
    p_x += vels_x * pixels_per_unit * dt;
    p_y += vels_y * pixels_per_unit * dt;

    //calculate distance to center
    af::array diff_x = p_x - width/2;
    af::array diff_y = p_y - height/2;
    af::array dist = sqrt( diff_x*diff_x + diff_y*diff_y );

    //calculate normalised force vectors
    forces_x = -1 * diff_x / dist;
    forces_y = -1 * diff_y / dist;
    //update force scaled to time and magnitude constant
    forces_x *= pixels_per_unit * dt;
    forces_y *= pixels_per_unit * dt;

    //dampening
    vels_x *= 1 - (0.005*dt);
    vels_y *= 1 - (0.005*dt);

    //update velocities from forces
    vels_x += forces_x;
    vels_y += forces_y;

}

void collisions(){
    //clamp particles inside screen border
    af::array projected_px = min(width, max(0, p_x));
    af::array projected_py = min(height - 1, max(0, p_y));

    //calculate distance to center
    af::array diff_x = projected_px - width/2;
    af::array diff_y = projected_py - height/2;
    af::array dist = sqrt( diff_x*diff_x + diff_y*diff_y );

    //collide with center sphere
    const int radius = 50;
    const float elastic_constant = 0.91f;
    if(sum<int>(dist<radius) > 0) {
        vels_x(dist<radius) = -elastic_constant * vels_x(dist<radius);
        vels_y(dist<radius) = -elastic_constant * vels_y(dist<radius);

        //normalize diff vector
        diff_x /= dist;
        diff_y /= dist;
        //place all particle colliding with sphere on surface
        p_x(dist<radius) = width/2 + diff_x(dist<radius) * radius;
        p_y(dist<radius) = height/2 +  diff_y(dist<radius) * radius;
    }
}


int main(int argc, char *argv[])
{
    try {
        const static int total_particles = 1000;
        static const int reset = 500;

        af::info();

        af::Window myWindow(width, height, "Gravity Simulation using ArrayFire");

        int frame_count = 0;

        // Initialize the kernel array just once
        const af::array draw_kernel = gaussianKernel(3, 3);

        // Generate a random starting state
        p_x = af::randu(total_particles) * width;
        p_y = af::randu(total_particles) * height;

        vels_x = af::randn(total_particles);
        vels_y = af::randn(total_particles);

        forces_x = af::randn(total_particles);
        forces_y = af::randn(total_particles);

        af::array image = af::constant(0, width, height);
        af::array ids(total_particles, u32);

        af::timer timer = af::timer::start();
        while(!myWindow.close()) {
            float dt = af::timer::stop(timer);
            timer = af::timer::start();

            ids = (p_x.as(u32) * height) + p_y.as(u32);
            image(ids) += 255;
            image = convolve2(image, draw_kernel);
            myWindow.image(image);
            image = af::constant(0, image.dims());
            frame_count++;

            // Generate a random starting state
            if(frame_count % reset == 0) {
                p_x = af::randu(total_particles) * width;
                p_y = af::randu(total_particles) * height;

                vels_x = af::randn(total_particles);
                vels_y = af::randn(total_particles);
            }

            //check for collisions and adjust velocities accordingly
            collisions();

            //run force simulation and update particles
            simulate(dt);

        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    #ifdef WIN32 // pause in Windows
    if (!(argc == 2 && argv[1][0] == '-')) {
        printf("hit [enter]...");
        fflush(stdout);
        getchar();
    }
    #endif
    return 0;
}

