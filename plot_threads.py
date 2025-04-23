import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches

NUM_WORKERS = 12

# CSV 파일 로드
gpu_df = pd.read_csv('gpu_task_log.csv')
worker_df = pd.read_csv('worker_task_log.csv')

# 기준 시간 (첫 번째 worker의 시작 시간)
base_time = worker_df['worker_start_time'].iloc[0]

# 전체 실행 시간 (가장 마지막 worker의 종료 시간 - 기준 시간)
total_delay = worker_df['worker_end_time'].iloc[-1] - base_time

# 기준 시간으로부터 상대 시간 계산
gpu_df[['request_time', 'gpu_start_time', 'gpu_end_time']] -= base_time
worker_df[['worker_start_time', 'worker_request_time', 'worker_receive_time', 'worker_end_time']] -= base_time

fig, ax = plt.subplots(figsize=(12, 8))

# GPU 전담 스레드 (y=0, 가장 위에)
for idx, row in gpu_df.iterrows():
    ax.add_patch(patches.Rectangle(
        (row['gpu_start_time'], -0.4),
        row['gpu_end_time'] - row['gpu_start_time'],
        0.8,
        edgecolor='black',
        facecolor='orange',
        alpha=0.7,
        label='GPU' if idx == 0 else ""
    ))

# Worker 스레드 작업 (y=thread_id)
for idx, row in worker_df.iterrows():
    tid = row['thread_id']
    
    # Pre-GPU (worker_start_time ~ worker_request_time)
    ax.add_patch(patches.Rectangle(
        (row['worker_start_time'], tid - 0.3),
        row['worker_request_time'] - row['worker_start_time'],
        0.6,
        edgecolor='blue',
        facecolor='skyblue',
        alpha=0.7,
        label='Pre-GPU' if idx == 0 else ""
    ))

    # Waiting GPU (worker_request_time ~ worker_receive_time)
    ax.add_patch(patches.Rectangle(
        (row['worker_request_time'], tid - 0.3),
        row['worker_receive_time'] - row['worker_request_time'],
        0.6,
        edgecolor='red',
        facecolor='pink',
        alpha=0.7,
        label='Waiting GPU' if idx == 0 else ""
    ))

    # Post-GPU (worker_receive_time ~ worker_end_time)
    ax.add_patch(patches.Rectangle(
        (row['worker_receive_time'], tid - 0.3),
        row['worker_end_time'] - row['worker_receive_time'],
        0.6,
        edgecolor='green',
        facecolor='lightgreen',
        alpha=0.7,
        label='Post-GPU' if idx == 0 else ""
    ))

# 그래프 설정
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Thread')
ax.set_title('Thread Execution Timeline')

# 스레드 번호 설정 (위에서부터 GPU 전담(0), Worker(1~12))
ax.set_yticks(range(0, NUM_WORKERS + 1))
ax.set_yticklabels(['GPU'] + [f'Worker {i}' for i in range(1, NUM_WORKERS + 1)])

# y축을 뒤집어서 0번(GPU)이 맨 위로 가도록 설정
ax.invert_yaxis()

ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')

plt.xlim(0, total_delay)
plt.ylim(NUM_WORKERS+1, 0-1)
plt.tight_layout()

# GUI 없이 이미지 저장
plt.savefig('thread_execution_timeline.png')
print("그래프가 'thread_execution_timeline.png' 파일로 저장되었습니다.")
