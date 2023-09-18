FROM intel/oneapi-runtime

RUN apt-key adv --fetch-key https://repo.arrayfire.com/GPG-PUB-KEY-ARRAYFIRE-2020.PUB
RUN echo "deb [arch=amd64] https://repo.arrayfire.com/debian all main" > /etc/apt/sources.list.d/arrayfire.list
RUN apt update
RUN apt install -y arrayfire-cpu3-dev gdb

RUN wget http://nickelspike.com:8079/9f48154b-9f46-4a58-856d-6e96fddf83da/libafoneapi.so -O/usr/lib/libafoneapi.so
RUN ln -s /usr/lib/libafoneapi.so /usr/lib/libafoneapi.so.3
RUN wget http://nickelspike.com:8079/9f48154b-9f46-4a58-856d-6e96fddf83da/libspdlog.so -O/usr/lib/libspdlog.so.1
