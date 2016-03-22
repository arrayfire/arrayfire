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

static const int width = 768, height = 768;
static const float eps = 10.f;
static const int gravity_constant = 5000;

void simulate(af::array *pos, af::array *vels, af::array *forces, float dt){
    pos[0] += vels[0] * dt;
    pos[1] += vels[1] * dt;

    //calculate forces to each particle
    af::array diff_x = tile(pos[0], 1, pos[0].dims(0))-transpose(tile(pos[0], 1, pos[0].dims(0)));
    af::array diff_y = tile(pos[1], 1, pos[1].dims(0))-transpose(tile(pos[1], 1, pos[1].dims(0)));
    af::array dist = af::sqrt( diff_x*diff_x + diff_y*diff_y );
    //dist = af::max(eps, dist);
    dist *= dist * dist;

    //calculate force vectors
    forces[0] = -diff_x / dist;
    forces[1] = -diff_y / dist;
    forces[0](af::isNaN(forces[0]))  = 0;
    forces[1](af::isNaN(forces[1]))  = 0;
    forces[0] = sum(forces[0], 1);
    forces[1] = sum(forces[1], 1);

    //update force scaled to time, magnitude constant
    forces[0] *= (gravity_constant);
    forces[1] *= (gravity_constant);

    //noise
    /*
    forces[0] += 0.1 * af::randn(forces[0].dims(0));
    forces[0] += 0.1 * af::randn(forces[0].dims(0));
    */

    //dampening
    /*
    vels[0] *= 1 - (0.005*dt);
    vels[1] *= 1 - (0.005*dt);
    */

    //update velocities from forces
    vels[0] += forces[0] * dt;
    vels[1] += forces[1] * dt;

    //temporary
    vels[0] = min(100, vels[0]);
    vels[1] = min(100, vels[1]);

}

void collisions(af::array *pos, af::array *vels){
    //clamp particles inside screen border
    af::array invalid_x = -2 * (pos[0] > width-1 || pos[0] < 0) + 1;
    af::array invalid_y = -2 * (pos[1] > height-1 || pos[1] < 0) + 1;
    vels[0]= invalid_x * vels[0] ;
    vels[1]= invalid_y * vels[1] ;

    af::array projected_px = min(width-1, max(0, pos[0]));
    af::array projected_py = min(height - 1, max(0, pos[1]));
    pos[0] = projected_px;
    pos[1] = projected_py;

    /*
    //calculate distance to center
    af::array diff_x = projected_px - width/2;
    af::array diff_y = projected_py - height/2;
    af::array dist = sqrt( diff_x*diff_x + diff_y*diff_y );

    //collide with center sphere
    const int radius = 20;
    const float elastic_constant = 0.91f;
    if(sum<int>(dist<radius) > 0) {
        vels[0](dist<radius) = -elastic_constant * vels[0](dist<radius);
        vels[1](dist<radius) = -elastic_constant * vels[1](dist<radius);

        //normalize diff vector
        diff_x /= dist;
        diff_y /= dist;
        //place all particle colliding with sphere on surface
        pos[0](dist<radius) = width/2 + diff_x(dist<radius) * radius;
        pos[1](dist<radius) = height/2 +  diff_y(dist<radius) * radius;
    }
    */
}


int main(int argc, char *argv[])
{
    try {
        const static int total_particles = 300;
        static const int reset = 5000;

        af::info();

        af::Window myWindow(width, height, "Gravity Simulation using ArrayFire");

        int frame_count = 0;

        // Initialize the kernel array just once
        const af::array draw_kernel = gaussianKernel(3, 3);

        af::array pos[2];
        af::array vels[2];
        af::array forces[2];

        // Generate a random starting state
        pos[0] = af::randu(total_particles) * width;
        pos[1] = af::randu(total_particles) * height;

        vels[0] = 1 * af::randn(total_particles);
        vels[1] = 10 * af::randn(total_particles);

        forces[0] = af::randn(total_particles);
        forces[1] = af::randn(total_particles);

        af::array image = af::constant(0, width, height);
        af::array ids(total_particles, u32);

        af::timer timer = af::timer::start();
        while(!myWindow.close()) {
            float dt = af::timer::stop(timer);
            timer = af::timer::start();

            ids = (pos[0].as(u32) * height) + pos[1].as(u32);
            image(ids) += 5.f;
            image = convolve(image, draw_kernel);
            myWindow.image(image);
            image = af::constant(0, image.dims());
            frame_count++;

            // Generate a random starting state
            if(frame_count % reset == 0) {
                pos[0] = af::randu(total_particles) * width;
                pos[1] = af::randu(total_particles) * height;

                vels[0] = af::randn(total_particles);
                vels[1] = af::randn(total_particles);
            }


            //run force simulation and update particles
            simulate(pos, vels, forces, dt);

            //check for collisions and adjust positions/velocities accordingly
            collisions(pos, vels);

        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}

