# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches

import multiprocessing

# GPU 스레드용 코어 1개 제외
num_workers = 11

# CSV 파일 로드
# gpu_df = pd.read_csv('gpu_task_log.csv')
# worker_df = pd.read_csv('worker_task_log.csv')
gpu_df = pd.read_csv('gpu_task_log_G0_306.csv')
worker_df = pd.read_csv('worker_task_log_G0_306.csv')

# 제외할 앞뒤 행 개수
cut = 5 * num_workers

# 앞뒤 cut개 제외
gpu_df = gpu_df.iloc[cut:-cut].reset_index(drop=True)
worker_df = worker_df.iloc[cut:-cut].reset_index(drop=True)

# 기준 시간 (첫 번째 worker의 시작 시간)
base_time = worker_df['worker_start_time'].iloc[0]

# 전체 실행 시간 (가장 마지막 worker의 종료 시간 - 기준 시간)
total_delay = worker_df['worker_end_time'].iloc[-1] - base_time

# 기준 시간으로부터 상대 시간 계산
gpu_df[['request_time', 'gpu_start_time', 'gpu_end_time',
        'push_start_time', 'push_end_time', 
        'pull_start_time', 'pull_end_time']] -= base_time
worker_df[['worker_start_time', 'worker_request_time', 'worker_receive_time', 'worker_end_time']] -= base_time

fig, ax = plt.subplots(figsize=(12, 8))

# GPU 스레드 시작 및 종료 시점에 세로 점선 추가
for idx, row in gpu_df.iterrows():
    # GPU 시작 시점에 세로 회색 점선
    ax.axvline(x=row['gpu_start_time'], color='gray', linestyle='--', alpha=0.5, zorder = -100)
    # GPU 종료 시점에 세로 회색 점선
    ax.axvline(x=row['gpu_end_time'], color='gray', linestyle='--', alpha=0.5, zorder = -100)

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

# GPU 전담 Worker 0의 Push/Pull 작업 표시 (y=1)
for idx, row in gpu_df.iterrows():
    # Push 작업 (Push는 파란색으로 예시)
    ax.add_patch(patches.Rectangle(
        (row['push_start_time'], 1 - 0.3),
        row['push_end_time'] - row['push_start_time'],
        0.6,
        edgecolor='purple',
        facecolor='mediumpurple',
        alpha=0.7,
        label='Push (Worker 0)' if idx == 0 else ""
    ))
    
    # Pull 작업 (Pull은 초록색으로 예시)
    ax.add_patch(patches.Rectangle(
        (row['pull_start_time'], 1 - 0.3),
        row['pull_end_time'] - row['pull_start_time'],
        0.6,
        edgecolor='brown',
        facecolor='peru',
        alpha=0.7,
        label='Pull (Worker 0)' if idx == 0 else ""
    ))

# Worker 스레드 작업 (y=thread_id)
for idx, row in worker_df.iterrows():
    tid = row['thread_id'] + 1
    
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
        edgecolor='grey',
        facecolor='grey',
        alpha=0.2,
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

# GPU(0), Worker 0(1), 나머지 Worker 스레드들(2부터 시작)
ax.set_yticks(range(0, num_workers + 2))  # num_workers+1 => num_workers+2로 변경
ax.set_yticklabels(
    ['GPU', 'Worker 0'] + [f'Worker {i}' for i in range(1, num_workers + 1)]
)

# y축을 뒤집어서 0번(GPU)이 맨 위로 가도록 설정
ax.invert_yaxis()

# ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')

ax.set_xlim(0, total_delay)
ax.set_ylim(num_workers + 2, -1)  # GPU(0) 가장 위
plt.tight_layout()

# GUI 없이 이미지 저장
plt.savefig('thread_execution_timeline.png')
print("그래프가 'thread_execution_timeline.png' 파일로 저장되었습니다.")