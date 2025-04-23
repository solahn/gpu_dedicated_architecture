#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define MAX_QUEUE_SIZE 128
#define TASKS_PER_WORKER 10
#define VISUAL 0

pthread_mutex_t task_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t task_cond = PTHREAD_COND_INITIALIZER;

pthread_mutex_t result_mutex[MAX_QUEUE_SIZE];
pthread_cond_t result_cond[MAX_QUEUE_SIZE];

FILE *fp_gpu;
FILE *fp_worker;
pthread_mutex_t file_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    float *input;
    float *output;
    int input_size;
    int task_id;
    int completed;
    double request_time;
    double gpu_start_time;
    double gpu_end_time;
    double worker_start_time;
    double worker_request_time;
    double worker_receive_time;
    double worker_end_time;
} gpu_task_t;

gpu_task_t task_queue[MAX_QUEUE_SIZE];
int task_head = 0, task_tail = 0;
int available_cores;

double current_time_in_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void log_gpu_task(gpu_task_t task, int thread_id) {
    pthread_mutex_lock(&file_mutex);
    fprintf(fp_gpu, "%d,%.2f,%.2f,%.2f\n", thread_id, task.request_time, task.gpu_start_time, task.gpu_end_time);
    pthread_mutex_unlock(&file_mutex);
}

void log_worker_task(gpu_task_t task, int thread_id) {
    pthread_mutex_lock(&file_mutex);
    fprintf(fp_worker, "%d,%.2f,%.2f,%.2f,%.2f\n", thread_id, task.worker_start_time, task.worker_request_time, task.worker_receive_time, task.worker_end_time);
    pthread_mutex_unlock(&file_mutex);
}

void* gpu_dedicated_thread(void* arg) {
    int core_id = sched_getcpu();
    printf("GPU-dedicated thread bound to core %d\n", core_id);
    gpu_task_t current_task;

    while (1) {
        pthread_mutex_lock(&task_mutex);
        while (task_head == task_tail)
            pthread_cond_wait(&task_cond, &task_mutex);

        current_task = task_queue[task_head % MAX_QUEUE_SIZE];
        task_head++;
        pthread_mutex_unlock(&task_mutex);

        if (VISUAL) printf("GPU Thread: Received task %d\n", current_task.task_id);

        current_task.gpu_start_time = current_time_in_ms();
        if (VISUAL) printf("GPU Thread: Start GPU task %d\n", current_task.task_id);
        usleep(1000);
        current_task.gpu_end_time = current_time_in_ms();
        if (VISUAL) printf("GPU Thread: End GPU task %d\n", current_task.task_id);

        pthread_mutex_lock(&result_mutex[current_task.task_id]);
        current_task.completed = 1;
        task_queue[current_task.task_id] = current_task;
        pthread_cond_signal(&result_cond[current_task.task_id]);
        pthread_mutex_unlock(&result_mutex[current_task.task_id]);

        log_gpu_task(current_task, current_task.task_id);
    }
}

void* worker_thread(void* arg) {
    int thread_id = *(int*)arg;
    int core_id = sched_getcpu();
    printf("Worker thread %d bound to core %d\n", thread_id, core_id);

    for (int i = 0; i < TASKS_PER_WORKER; i++) {
        gpu_task_t task;
        task.input_size = 1024;
        task.input = malloc(sizeof(float) * task.input_size);
        task.output = malloc(sizeof(float) * task.input_size);
        task.completed = 0;

        task.worker_start_time = current_time_in_ms();
        if (VISUAL) printf("Worker %d: Start preprocessing (task %d)\n", thread_id, i);
        usleep(500);

        pthread_mutex_lock(&task_mutex);
        task.task_id = task_tail;
        task.request_time = current_time_in_ms();
        task.worker_request_time = task.request_time;
        task_queue[task_tail % MAX_QUEUE_SIZE] = task;
        task_tail++;
        pthread_cond_signal(&task_cond);
        pthread_mutex_unlock(&task_mutex);

        if (VISUAL) printf("Worker %d: Requested GPU task (task %d)\n", thread_id, i);

        pthread_mutex_lock(&result_mutex[task.task_id]);
        while (!task_queue[task.task_id].completed)
            pthread_cond_wait(&result_cond[task.task_id], &result_mutex[task.task_id]);
        pthread_mutex_unlock(&result_mutex[task.task_id]);

        task.worker_receive_time = current_time_in_ms();
        if (VISUAL) printf("Worker %d: Received GPU result (task %d)\n", thread_id, i);

        usleep(500);
        task.worker_end_time = current_time_in_ms();
        if (VISUAL) printf("Worker %d: Finished postprocessing (task %d)\n", thread_id, i);

        log_worker_task(task, thread_id);

        free(task.input);
        free(task.output);

        if (VISUAL) printf("Worker %d: Completed task %d\n", thread_id, i);
    }

    pthread_exit(NULL);
}

int main() {
    available_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int num_workers = available_cores - 1; // GPU 스레드용 코어 1개 제외


    pthread_t gpu_thread, workers[num_workers];
    int thread_ids[num_workers];

    fp_gpu = fopen("gpu_task_log.csv", "w");
    fprintf(fp_gpu, "thread_id,request_time,gpu_start_time,gpu_end_time\n");

    fp_worker = fopen("worker_task_log.csv", "w");
    fprintf(fp_worker, "thread_id,worker_start_time,worker_request_time,worker_receive_time,worker_end_time\n");

    pthread_create(&gpu_thread, NULL, gpu_dedicated_thread, NULL);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); // 코어 0은 GPU 스레드용으로 예약
    pthread_setaffinity_np(gpu_thread, sizeof(cpuset), &cpuset);

    for (int i = 0; i < num_workers; i++) {
        thread_ids[i] = i + 1;
        pthread_create(&workers[i], NULL, worker_thread, &thread_ids[i]);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        int core_id = 1 + (i % (available_cores - 1)); // 1부터 available_cores까지 사용
        CPU_SET(core_id, &cpuset);
        pthread_setaffinity_np(workers[i], sizeof(cpuset), &cpuset);
    }

    for (int i = 0; i < num_workers; i++) {
        pthread_join(workers[i], NULL);
    }

    pthread_cancel(gpu_thread);
    pthread_join(gpu_thread, NULL);

    fclose(fp_gpu);
    fclose(fp_worker);

    return 0;
}
