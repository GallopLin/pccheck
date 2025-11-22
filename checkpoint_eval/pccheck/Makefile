CC=g++
# 添加静态链接libstdc++和libgcc，避免conda环境中的libstdc++版本不匹配问题
CFLAGS=-pthread -I/usr/local/cuda-12.1/include/ -L/usr/local/cuda-12.1/lib64/ -latomic -lpmem -lcudart -O3 -march=native -mtune=native -ffast-math -static-libstdc++ -static-libgcc

main.o: main.cpp
	$(CC) -c -fPIC main.cpp -o main.o

main_ssd_memory.o: main_ssd_memory.cpp
	$(CC) -c -fPIC main_ssd_memory.cpp -o main_ssd_memory.o $(CFLAGS)

socket_work.o: socket_work.cpp
	$(CC) -c -fPIC socket_work.cpp -o socket_work.o $(CFLAGS)

libtest_ssd.so: main_ssd_memory.o socket_work.o FAAQueue.h DRAMAlloc.h
	$(CC) -shared -o libtest_ssd.so main_ssd_memory.o  socket_work.o $(CFLAGS)

all: libtest.so libtest_ssd.so

clean:
	rm -rf *.o *.so
