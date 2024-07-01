import numpy as np
import matplotlib.pyplot as plt
import tqdm

# 定义参数
warmup_periods = 5
steps_per_period = 1000
first_stage_epochs = 10
second_stage_epochs = 20
learn_rate_init = 0.001
learn_rate_end = 0.0001

# 计算 warmup_steps 和 train_steps
warmup_steps = warmup_periods * steps_per_period
train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

# 初始化 global_step 和 learn_rate 列表
global_steps = np.arange(1, train_steps + 1)
learn_rates = []

# 计算每个 global_step 对应的 learn_rate
for step in global_steps:
    if step < warmup_steps:
        learn_rate = step / warmup_steps * learn_rate_init
    else:
        learn_rate = learn_rate_end + 0.5 * (learn_rate_init - learn_rate_end) * (1 + np.cos((step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
    learn_rates.append(learn_rate)

# 绘制 learn_rate 曲线并保存到文件
plt.figure(figsize=(10, 6))
plt.plot(global_steps, learn_rates, label='Learning Rate')
plt.xlabel('Global Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()
plt.grid(True)

# 保存图像到文件
plt.savefig('learning_rate_schedule.png')

