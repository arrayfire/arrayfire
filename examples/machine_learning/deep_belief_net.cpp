/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <af/util.h>
#include <math.h>
#include "mnist_common.h"

using namespace af;
using std::vector;

float accuracy(const array& predicted, const array& target)
{
    array val, plabels, tlabels;
    max(val, tlabels, target, 1);
    max(val, plabels, predicted, 1);
    return 100 * count<float>(plabels == tlabels) / tlabels.elements();
}

// Activation function
array sigmoid(const array &val)
{
    return 1 / (1 + exp(-val));
}

// Derivative of the activation function
array deriv(const array &out)
{
    return out * (1 - out);
}

// Cost function
double error(const array &out,
             const array &pred)
{
    array dif = (out - pred);
    return sqrt((double)(sum<float>(dif * dif)));
}

array sigmoid_binary(const array in)
{
    // Choosing "1" with probability sigmoid(in)
    return (sigmoid(in) > randu(in.dims())).as(f32);
}

class rbm {

private:
    array weights;
    array h_bias;
    array v_bias;

public:
    rbm(int v_size, int h_size) :
        weights(randu(h_size, v_size)/100.f),
        h_bias(constant(0, 1, h_size)),
        v_bias(constant(0, 1, v_size))
    {
    }

    array get_weights()
    {
        return transpose(join(1, weights, transpose(h_bias)));
    }

    void train(const array &in, double lr, int num_epochs, int batch_size, bool verbose)
    {
        const int num_samples = in.dims(0);
        const int num_batches = num_samples / batch_size;

        for (int i = 0; i <  num_epochs; i++) {

            double err = 0;

            for (int j = 0; j < num_batches - 1; j++) {

                int st = j * batch_size;
                int en = std::min(num_samples - 1, st + batch_size);
                int num = en - st + 1;

                array v_pos = in(seq(st, en), span);

                array h_pos = sigmoid_binary(tile(h_bias, num) +
                                             matmulNT(v_pos, weights));

                array v_neg = sigmoid_binary(tile(v_bias, num) +
                                             matmul(h_pos, weights));

                array h_neg = sigmoid_binary(tile(h_bias, num) +
                                             matmulNT(v_neg, weights));


                array c_pos = matmulTN(h_pos, v_pos);
                array c_neg = matmulTN(h_neg, v_neg);

                array delta_w = lr * (c_pos - c_neg) / num;
                array delta_vb = lr * sum(v_pos - v_neg) / num;
                array delta_hb = lr * sum(h_pos - h_neg) / num;

                weights += delta_w;
                v_bias += delta_vb;
                h_bias += delta_hb;

                if (verbose) {
                    err += error(v_pos, v_neg);
                }
            }

            if (verbose) {
                printf("Epoch %d: Reconstruction error: %0.4f\n", i + 1, err / num_batches);
            }
        }
    }

    array prop_up(const array &in)
    {
        return sigmoid(tile(h_bias, in.dims(0)) +
                       matmulNT(in, weights));
    }
};

class dbn {

private:
    const int in_size;
    const int out_size;
    const int num_hidden;
    const int num_total;
    std::vector<array> weights;
    std::vector<int> hidden;

    array add_bias(const array &in)
    {
        // Bias input is added on top of given input
        return join(1, constant(1, in.dims(0), 1), in);
    }

    vector<array> forward_propagate(const array& input)
    {
        // Get activations at each layer
        vector<array> signal(num_total);
        signal[0] = input;

        for (int i = 0; i < num_total - 1; i++) {
            array in = add_bias(signal[i]);
            array out = matmul(in, weights[i]);
            signal[i + 1] = sigmoid(out);
        }

        return signal;
    }

    void back_propagate(const vector<array> signal,
                        const array &target,
                        const double &alpha)
    {

        // Get error for output layer
        array out = signal[num_total  - 1];
        array err = (out - target);
        int m = target.dims(0);

        for (int i = num_total - 2; i >= 0; i--) {
            array in = add_bias(signal[i]);
            array delta = (deriv(out) * err).T();

            // Adjust weights
            array grad = -(alpha * matmul(delta, in)) / m;
            weights[i] += grad.T();

            // Input to current layer is output of previous
            out = signal[i];
            err = matmulTT(delta, weights[i]);

            // Remove the error of bias and propagate backward
            err = err(span, seq(1, out.dims(1)));
        }
    }

public:

    dbn(const int in_sz, const int out_sz,
        const std::vector<int> hidden_layers) :
        in_size(in_sz),
        out_size(out_sz),
        num_hidden(hidden_layers.size()),
        num_total(hidden_layers.size() + 2),
        weights(hidden_layers.size() + 1),
        hidden(hidden_layers)
    {
    }

    void train(const array &input, const array &target,
               double lr_rbm = 1.0,
               double lr_nn  = 1.0,
               const int epochs_rbm = 15,
               const int epochs_nn = 300,
               const int batch_size = 100,
               double maxerr = 1.0, bool verbose=false)
    {

        // Pre-training hidden layers
        array X = input;
        for (int i = 0; i < num_hidden; i++) {

            if (verbose) {
                printf("Training Hidden Layer %d\n", i);
            }

            int visible = (i == 0) ? in_size : hidden[i - 1];

            rbm r(visible, hidden[i]);
            r.train(X, lr_rbm, epochs_rbm, batch_size, verbose);

            X = r.prop_up(X);
            weights[i] = r.get_weights();

            if (verbose) {
                printf("\n");
            }
        }

        weights[num_hidden] = 0.05 * randu(hidden[num_hidden - 1] + 1, out_size) - 0.0025;

        const int num_samples = input.dims(0);
        const int num_batches = num_samples / batch_size;

        // Training the entire network
        for (int i = 0; i < epochs_nn; i++) {

            for (int j = 0; j < num_batches; j++) {

                int st = j * batch_size;
                int en = std::min(num_samples - 1, st + batch_size);

                array x = input(seq(st, en), span);
                array y = target(seq(st, en), span);

                // Propagate the inputs forward
                vector<array> signals = forward_propagate(x);
                array out = signals[num_total - 1];

                // Propagate the error backward
                back_propagate(signals, y, lr_nn);
            }


            // Validate with last batch
            int st = (num_batches - 1) * batch_size;
            int en = num_samples - 1;
            array out = predict(input(seq(st, en), span));
            double err = error(out, target(seq(st, en), span));

            // Check if convergence criteria has been met
            if (err < maxerr) {
                printf("Converged on Epoch: %4d\n", i + 1);
                return;
            }

            if (verbose) {
                if ((i + 1) % 10 == 0) printf("Epoch: %4d, Error: %0.4f\n", i+1, err);
            }
        }
    }

    array predict(const array &input)
    {
        vector<array> signal = forward_propagate(input);
        array out = signal[num_total - 1];
        return out;
    }

};

int dbn_demo(bool console, int perc)
{
    printf("** ArrayFire DBN Demo **\n\n");

    array train_images, test_images;
    array train_target, test_target;
    int num_classes, num_train, num_test;

    // Load mnist data
    float frac = (float)(perc) / 100.0;
    setup_mnist<true>(&num_classes, &num_train, &num_test,
                      train_images, test_images, train_target, test_target, frac);

    int feature_size = train_images.elements() / num_train;

    // Reshape images into feature vectors
    array train_feats = moddims(train_images, feature_size, num_train).T();
    array test_feats  = moddims(test_images , feature_size, num_test ).T();

    train_target = train_target.T();
    test_target  = test_target.T();

    // Network parameters
    vector<int> layers;
    layers.push_back(100);
    layers.push_back(50);

    // Create network
    dbn network(train_feats.dims(1), num_classes, layers);

    // Train network
    timer::start();
    network.train(train_feats, train_target,
                  0.2,  // rbm learning rate
                  4.0,  // nn learning rate
                  15,   // rbm epochs
                  250,  // nn epochs
                  100,  // batch_size
                  0.5,  // max error
                  true);// verbose
    af::sync();
    double train_time = timer::stop();

    // Run the trained network and test accuracy.
    array train_output = network.predict(train_feats);
    array test_output  = network.predict(test_feats );


    // Benchmark prediction
    af::sync();
    timer::start();
    for (int i = 0; i < 100; i++) {
        network.predict(test_feats);
    }
    af::sync();
    double test_time = timer::stop() / 100;

    printf("\nTraining set:\n");
    printf("Accuracy on training data: %2.2f\n",
           accuracy(train_output, train_target));

    printf("\nTest set:\n");
    printf("Accuracy on testing  data: %2.2f\n",
           accuracy(test_output , test_target ));

    printf("\nTraining time: %4.4lf s\n", train_time);
    printf("Prediction time: %4.4lf s\n\n", test_time);

    if (!console) {
        // Get 20 random test images.
        test_output = test_output.T();
        display_results<true>(test_images, test_output, test_target.T(), 20);
    }

    return 0;
}

int main(int argc, char** argv)
{
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;
    int perc     = argc > 3 ? atoi(argv[3]) : 60;

    try {

        af::setDevice(device);
        af::info();
        return dbn_demo(console, perc);

    } catch (af::exception &ae) {
        std::cerr << ae.what() << std::endl;
    }

}
