/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include "gravity_sim_init.h"

using namespace af;
using namespace std;

static const bool is3D           = true;
const static int total_particles = 4000;
static const int reset           = 3000;
static const float min_dist      = 3;
static const int width = 768, height = 768, depth = 768;
static const int gravity_constant = 20000;

float mass_range = 0;
float min_mass   = 0;

void initial_conditions_rand(af::array &mass, vector<af::array> &pos,
                             vector<af::array> &vels,
                             vector<af::array> &forces) {
    for (int i = 0; i < (int)pos.size(); ++i) {
        pos[i]    = af::randn(total_particles) * width + width;
        vels[i]   = 0 * af::randu(total_particles) - 0.5;
        forces[i] = af::constant(0, total_particles);
    }
    mass = af::constant(gravity_constant, total_particles);
}

void initial_conditions_galaxy(af::array &mass, vector<af::array> &pos,
                               vector<af::array> &vels,
                               vector<af::array> &forces) {
    af::array initial_cond_consts(af::dim4(7, total_particles), hbd);
    initial_cond_consts = initial_cond_consts.T();

    for (int i = 0; i < (int)pos.size(); ++i) {
        pos[i]    = af::randn(total_particles) * width + width;
        vels[i]   = 0 * (af::randu(total_particles) - 0.5);
        forces[i] = af::constant(0, total_particles);
    }

    mass    = initial_cond_consts(span, 0);
    pos[0]  = (initial_cond_consts(span, 1) / 32 + 0.6) * width;
    pos[1]  = (initial_cond_consts(span, 2) / 32 + 0.3) * height;
    pos[2]  = (initial_cond_consts(span, 3) / 32 + 0.5) * depth;
    vels[0] = (initial_cond_consts(span, 4) / 32) * width;
    vels[1] = (initial_cond_consts(span, 5) / 32) * height;
    vels[2] = (initial_cond_consts(span, 6) / 32) * depth;

    pos[0](seq(0, pos[0].dims(0) - 1, 2)) -= 0.4 * width;
    pos[1](seq(0, pos[0].dims(0) - 1, 2)) += 0.4 * height;
    vels[0](seq(0, pos[0].dims(0) - 1, 2)) += 4;

    min_mass   = min<float>(mass);
    mass_range = max<float>(mass) - min<float>(mass);
}

af::array ids_from_pos(vector<af::array> &pos) {
    return (pos[0].as(u32) * height) + pos[1].as(u32);
}

af::array ids_from_3D(vector<af::array> &pos, float Rx, float Ry, float Rz) {
    af::array x0 = (pos[0] - width / 2);
    af::array y0 =
        (pos[1] - height / 2) * cos(Rx) + (pos[2] - depth / 2) * sin(Rx);
    af::array z0 =
        (pos[2] - depth / 2) * cos(Rx) - (pos[2] - depth / 2) * sin(Rx);

    af::array x1 = x0 * cos(Ry) - z0 * sin(Ry);
    af::array y1 = y0;

    af::array x2 = x1 * cos(Rz) + y1 * sin(Rz);
    af::array y2 = y1 * cos(Rz) - x1 * sin(Rz);

    x2 += width / 2;
    y2 += height / 2;

    return (x2.as(u32) * height) + y2.as(u32);
}

af::array ids_from_3D(vector<af::array> &pos, float Rx, float Ry, float Rz,
                      af::array filter) {
    af::array x0 = (pos[0](filter) - width / 2);
    af::array y0 = (pos[1](filter) - height / 2) * cos(Rx) +
                   (pos[2](filter) - depth / 2) * sin(Rx);
    af::array z0 = (pos[2](filter) - depth / 2) * cos(Rx) -
                   (pos[2](filter) - depth / 2) * sin(Rx);

    af::array x1 = x0 * cos(Ry) - z0 * sin(Ry);
    af::array y1 = y0;

    af::array x2 = x1 * cos(Rz) + y1 * sin(Rz);
    af::array y2 = y1 * cos(Rz) - x1 * sin(Rz);

    x2 += width / 2;
    y2 += height / 2;

    return (x2.as(u32) * height) + y2.as(u32);
}

void simulate(af::array &mass, vector<af::array> &pos, vector<af::array> &vels,
              vector<af::array> &forces, float dt) {
    for (int i = 0; i < (int)pos.size(); ++i) {
        pos[i] += vels[i] * dt;
        pos[i].eval();
    }

    // calculate forces to each particle
    vector<af::array> diff(pos.size());
    af::array dist = af::constant(0, pos[0].dims(0), pos[0].dims(0));

    for (int i = 0; i < (int)pos.size(); ++i) {
        diff[i] = tile(pos[i], 1, pos[i].dims(0)) -
                  transpose(tile(pos[i], 1, pos[i].dims(0)));
        dist += (diff[i] * diff[i]);
    }

    dist = sqrt(dist);
    dist = af::max(min_dist, dist);
    dist *= dist * dist;

    for (int i = 0; i < (int)pos.size(); ++i) {
        // calculate force vectors
        forces[i] = diff[i] / dist;
        forces[i].eval();

        // af::array idx = af::where(af::isNaN(forces[i]));
        // if(idx.elements() > 0)
        //    forces[i](idx) = 0;
        // forces[i] = sum(forces[i]).T();
        forces[i] = matmul(forces[i].T(), mass);

        // update force scaled to time, magnitude constant
        forces[i] *= (gravity_constant);
        forces[i].eval();

        // update velocities from forces
        vels[i] += forces[i] * dt;
        vels[i].eval();

        // noise
        // forces[i] += 0.1 * af::randn(forces[i].dims(0));

        // dampening
        // vels[i] *= 1 - (0.005*dt);
    }
}

void collisions(vector<af::array> &pos, vector<af::array> &vels, bool is3D) {
    // clamp particles inside screen border
    af::array invalid_x = -2 * (pos[0] > width - 1 || pos[0] < 0) + 1;
    af::array invalid_y = -2 * (pos[1] > height - 1 || pos[1] < 0) + 1;
    // af::array invalid_x = (pos[0] < width-1 || pos[0] > 0);
    // af::array invalid_y = (pos[1] < height-1 || pos[1] > 0);
    vels[0] = invalid_x * vels[0];
    vels[1] = invalid_y * vels[1];

    af::array projected_px = min(width - 1, max(0, pos[0]));
    af::array projected_py = min(height - 1, max(0, pos[1]));
    pos[0]                 = projected_px;
    pos[1]                 = projected_py;

    if (is3D) {
        af::array invalid_z    = -2 * (pos[2] > depth - 1 || pos[2] < 0) + 1;
        vels[2]                = invalid_z * vels[2];
        af::array projected_pz = min(depth - 1, max(0, pos[2]));
        pos[2]                 = projected_pz;
    }
}

int main(int, char **) {
    try {
        af::info();

        af::Window myWindow(width, height,
                            "Gravity Simulation using ArrayFire");
        myWindow.setColorMap(AF_COLORMAP_HEAT);

        int frame_count = 0;

        // Initialize the kernel array just once
        const af::array draw_kernel = gaussianKernel(7, 7);

        const int dims = (is3D) ? 3 : 2;

        vector<af::array> pos(dims);
        vector<af::array> vels(dims);
        vector<af::array> forces(dims);
        af::array mass;

        // Generate a random starting state
        initial_conditions_galaxy(mass, pos, vels, forces);

        af::array image = af::constant(0, width, height);
        af::array ids(total_particles, u32);

        af::timer timer = af::timer::start();
        while (!myWindow.close()) {
            float dt = af::timer::stop(timer);
            timer    = af::timer::start();

            af::array mid = mass(span) > (min_mass + mass_range / 3);
            ids = (is3D) ? ids_from_3D(pos, 0, 0 + frame_count / 150.f, 0, mid)
                         : ids_from_pos(pos);
            // ids = (is3D)? ids_from_3D(pos, 0, 0, 0, mid) : ids_from_pos(pos);
            // //uncomment for no 3d rotation
            image(ids) += 4.f;

            mid = mass(span) > (min_mass + 2 * mass_range / 3);
            ids = (is3D) ? ids_from_3D(pos, 0, 0 + frame_count / 150.f, 0, mid)
                         : ids_from_pos(pos);
            // ids = (is3D)? ids_from_3D(pos, 0, 0, 0, mid) : ids_from_pos(pos);
            // //uncomment for no 3d rotation
            image(ids) += 4.f;

            ids = (is3D) ? ids_from_3D(pos, 0, 0 + frame_count / 150.f, 0)
                         : ids_from_pos(pos);
            // ids = (is3D)? ids_from_3D(pos, 0, 0, 0) :  ids_from_pos(pos);
            // //uncomment for no 3d rotation
            image(ids) += 4.f;

            image = convolve(image, draw_kernel);
            myWindow.image(image);
            image = af::constant(0, image.dims());

            frame_count++;

            // Generate a random starting state
            if (frame_count % reset == 0) {
                initial_conditions_galaxy(mass, pos, vels, forces);
            }

            // simulate
            simulate(mass, pos, vels, forces, dt);

            // check for collisions and adjust positions/velocities accordingly
            collisions(pos, vels, is3D);
        }
    } catch (af::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
