import torch
import numpy as np

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_periods, first_stage_epochs, second_stage_epochs, steps_per_period, learn_rate_init, learn_rate_end):
        self.optimizer = optimizer
        self.warmup_steps = warmup_periods * steps_per_period
        self.train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period
        self.learn_rate_init = learn_rate_init
        self.learn_rate_end = learn_rate_end
        
    
    def get_lr(self, global_step):
        self.global_step = global_step
        if self.global_step < self.warmup_steps:
            lr = self.learn_rate_init * self.global_step / self.warmup_steps 
        else:
            lr = self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) * (1 + np.cos((self.global_step - self.warmup_steps) / (self.train_steps - self.warmup_steps) * np.pi))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
#####################################################################################################################

################################这里是一个例子################################
# 实例化模型
model = SimpleNet()

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 设置初始学习率和最大学习率
base_lr = 0.01
max_lr = 0.1

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=base_lr)

# 定义自定义学习率调度器
warmup_epochs = 5
total_epochs = 30
scheduler = CustomLRScheduler(optimizer, warmup_epochs, total_epochs, base_lr, max_lr)

# 假设我们有一些训练数据
inputs = torch.randn(32, 1, 28, 28)  # 32个样本，每个样本是1x28x28的图像
labels = torch.randint(0, 10, (32,))  # 32个样本的标签，取值范围是0-9

# 训练过程
for epoch in range(total_epochs):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()  # 梯度清零
    loss.backward()        # 反向传播
    optimizer.step()       # 更新参数
    
    # 更新学习率
    scheduler.step()
    
    # 打印当前学习率
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{total_epochs}, Loss: {loss.item()}, Learning Rate: {current_lr}")

print("训练完成")