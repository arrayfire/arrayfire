
#include "async_queue.hpp"
#include "parallel.hpp"
#include <chrono>

using namespace std::chrono;

using namespace std;

void blah(int &c, int a, int b) {
    std::cout << "starting b" << std::endl;
    for(int i = 0; i < b; i++)
        c += a;// + b;
    std::cout << "ending b" << std::endl;
}

void other(float& out, int a, int b) {
    out = a * b;
}


int main(int argc, char *argv[])
{
    std::cout << "test" << std::endl;
    async_queue my_queue;
    int c = 0;
    float d = 0;
    my_queue.enqueue(blah, std::ref(c), 1, 500);
    my_queue.enqueue(other, std::ref(d), 1, 500);
    my_queue.sync();
    std::cout << c << std::endl;
    my_queue.enqueue(blah, std::ref(c), 7, 5000);
    my_queue.enqueue(blah, std::ref(c), 9, 5000000000);
    my_queue.enqueue(blah, std::ref(c), 1, 15);
    //for(int i = 0; i < 100; i++)
    //std::cout << c << std::endl;

    std::this_thread::sleep_for(milliseconds(1000));
    std::cout << c << std::endl;
    std::cout << dec;
    //auto iter = make_tuple<size_t, size_t, size_t, size_t>((10 * argc + 1), (100 * argc + 1), (100 * argc + 1), (100 * argc + 1));
    auto iter = dim_t{{size_t(2000000 * argc ), size_t(1000 ), size_t(1), size_t(16)}};

    vector<float> out(get<0>(iter)* get<2>(iter)* get<2>(iter)* get<3>(iter));
    vector<float> in(get<0>(iter)* get<2>(iter)* get<2>(iter)* get<3>(iter));
    {
    auto start = high_resolution_clock::now();
    for(size_t i = 0; i < get<3>(iter); i++) {
        for(size_t j = 0; j < get<2>(iter); j++) {
            for(size_t k = 0; k < get<1>(iter); k++) {
                for(size_t l = 0; l < get<0>(iter); l++) {
                    out[i*get<3>(iter) + j * get<2>(iter) + k *get<1>(iter) + l ] = (in[i*get<3>(iter) + j * get<2>(iter) + k *get<1>(iter) + l ] );
                }
            }
        }
    }
    size_t ms = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    std::cout << "duration: " << ms << "ms"<< std::endl;
    std::cout << out[24] << endl;
    }

    //vector<float> out1(get<0>(iter)* get<1>(iter)* get<2>(iter)* get<3>(iter));
    //vector<float> in1(get<0>(iter)* get<1>(iter)* get<2>(iter)* get<3>(iter), argc);
    for(int i = 0; i < 1; i ++) {
        auto start = high_resolution_clock::now();
        parallel_mat par(iter, [&](const dim_t &loc) {
            out[get<3>(loc)*get<3>(iter) + get<2>(loc) * get<2>(iter) + get<1>(loc) *get<1>(iter) + loc[0]] =
            (in[get<3>(loc)*get<3>(iter) + get<2>(loc) * get<2>(iter) + get<1>(loc) *get<1>(iter) + loc[0]]);
        });

        auto ms = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
        std::cout << "duration: " << ms << "ms"<< std::endl;

        std::cout << out[24] << endl;
    }
    return 0;
}


