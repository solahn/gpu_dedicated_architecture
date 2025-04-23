gcc -o gpu_dedicated_arch gpu_dedicated_arch.c -lpthread -lrt
./gpu_dedicated_arch
python3 plot_threads.py