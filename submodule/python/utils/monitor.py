import wnvml
import time
import os
import torch
import torch.distributed as dist

def monitor_gpu_power(stop_event, node_idx, node_rank, trace_path):
    res_power = []
    res_timer = []
    wnvml.nvmlInit()
    start_time = time.time()
    while not stop_event.value:
        device = wnvml.nvmlDeviceGetHandleByIndex(node_rank)
        power = wnvml.nvmlDeviceGetPowerUsage(device) / 1000.0
        current_time = time.time() - start_time
        res_power.append(power)
        res_timer.append(current_time)

    export_log(node_idx, node_rank, res_power, res_timer, trace_path)

def export_log(node_idx, node_rank, res_power, res_timer, trace_path):
    wh = 0.0
    previous_time = res_timer[0]
    for i in range(1, len(res_timer)):
        current_time = res_timer[i]
        wh += (res_power[i - 1] + res_power[i]) * (current_time - previous_time) / 2
        previous_time = current_time
    wh /= 3600.0

    with open(f"{trace_path}/energy/node{node_idx}/result{node_rank}.txt", 'w') as file:
        file.write(str(wh) + '\n')
    res_power_transpose = list(map(list, zip(res_power)))
    with open(f"{trace_path}/energy/node{node_idx}/power{node_rank}.txt", 'w') as file:
        for powers in res_power_transpose:
            power_string = ' '.join(str(power) for power in powers)
            file.write(f"{power_string}\n")
    res_timer_transpose = list(map(list, zip(res_timer)))
    with open(f"{trace_path}/energy/node{node_idx}/timer{node_rank}.txt", 'w') as file:
        for timers in res_timer_transpose:
            timer_string = ' '.join(str(round(timer, 5)) for timer in timers)
            file.write(f"{timer_string}\n")

def cal_energy(nodes, node_world_size, trace_path):
    kwh = 0
    for i in range(nodes):
        for j in range(node_world_size):
            with open(f"{trace_path}/energy/node{i}/result{j}.txt") as file:
                wh = float(file.readline().strip())
                kwh += wh
    kwh /= 1000.0
    return kwh
