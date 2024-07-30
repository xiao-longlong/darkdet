import os
import time
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import core.utils as utils
from core.dataset_lowlight import Dataset
from core.yolov3_lowlight import YOLOV3
from core.config_lowlight import cfg
from core.config_lowlight import args
import random

torch.autograd.set_detect_anomaly(True)

# wxl：替换的步骤不一定等价，显卡可见待修改。
# Check device
device = torch.device("cuda" if args.use_gpu else "cpu")

exp_folder = os.path.join(args.exp_dir, 'exp_{}'.format(args.exp_num))
set_ckpt_dir = args.ckpt_dir
args.ckpt_dir = os.path.join(exp_folder, set_ckpt_dir)
if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

config_log = os.path.join(exp_folder, 'config.txt')
arg_dict = args.__dict__
msg = ['{}: {}\n'.format(k, v) for k, v in arg_dict.items()]
utils.write_mes(msg, config_log, mode='w')

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_periods, first_stage_epochs, second_stage_epochs, steps_per_period, learn_rate_init, learn_rate_end):
        self.optimizer = optimizer
        self.warmup_steps = warmup_periods * steps_per_period
        self.train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period
        self.learn_rate_init = learn_rate_init
        self.learn_rate_end = learn_rate_end
        
    
    def step(self, global_step):
        self.global_step = global_step
        if self.global_step < self.warmup_steps:
            lr = self.learn_rate_init * self.global_step / self.warmup_steps 
        else:
            lr = self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) * (1 + np.cos((self.global_step - self.warmup_steps) / (self.train_steps - self.warmup_steps) * np.pi))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

class YoloTrain:
    def __init__(self):
        self.anchor_per_scale       = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes                = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes            = len(self.classes)
        self.learn_rate_init        = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end         = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs     = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs    = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods         = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight         = cfg.TRAIN.INITIAL_WEIGHT
        self.time                   = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay       = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale     = 150
        self.train_logdir           = "./data/log/train"
        self.trainset               = Dataset('train')
        self.testset                = Dataset('test')
        self.steps_per_period       = len(self.trainset)

        # Model, optimizer, and loss function
        self.model = YOLOV3().to(device)
        # wxl：使用优化器函数直接改写了
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate_init)
        self.scheduler = WarmupCosineAnnealingLR(self.optimizer, self.warmup_periods, 
                                                 self.first_stage_epochs, self.second_stage_epochs, 
                                                 self.steps_per_period, self.learn_rate_init, self.learn_rate_end)
        
        
        if os.path.exists(self.initial_weight):
            self.model.load_state_dict(torch.load(self.initial_weight))
        self.global_step = 0

    # 没有二阶段

    def train(self):
        best_loss = float('inf')

        for epoch in range(1, self.first_stage_epochs + self.second_stage_epochs + 1):
            self.model.train()
            train_epoch_loss = []
            pbar = tqdm(self.trainset,desc=f"Epoch {epoch}/{self.first_stage_epochs + self.second_stage_epochs}")

            for train_data in pbar:
                self.global_step += 1
                
                images, labels_sbbox, labels_mbbox, labels_lbbox, \
                true_sbboxes, true_mbboxes, true_lbboxes = train_data
                
                lowlight_param = 1
                if args.lowlight_FLAG:
                    if random.randint(0, 2) > 0:
                        lowlight_param = random.uniform(1.5, 5)
                else:
                    input_data = images


                input_data = np.power(images, lowlight_param)
                input_data_clean = images

                input_data = torch.from_numpy(input_data).to(device).to(torch.float32)
                # b h w a n
                labels_sbbox = torch.from_numpy(labels_sbbox).to(device)
                labels_mbbox = torch.from_numpy(labels_mbbox).to(device)
                labels_lbbox = torch.from_numpy(labels_lbbox).to(device)
                # b ? ?
                true_sbboxes = torch.from_numpy(true_sbboxes).to(device)
                true_mbboxes = torch.from_numpy(true_mbboxes).to(device)
                true_lbboxes = torch.from_numpy(true_lbboxes).to(device)

                input_data_clean = torch.from_numpy(input_data_clean).to(device).to(torch.float32)

                # bchw
                input_data = input_data.permute(0, 3, 1, 2)
                # bchw
                input_data_clean = input_data_clean.permute(0, 3, 1, 2)
                # todo 检查model中是否有错误的转置
                self.model(input_data, input_data_clean)
                giou_loss, conf_loss, prob_loss, recovery_loss = self.model.compute_loss(labels_sbbox, labels_mbbox, labels_lbbox, true_sbboxes, true_mbboxes, true_lbboxes)
                loss = giou_loss + conf_loss + prob_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss.item())
                pbar.set_description("train loss: %.2f" % loss.item())
                self.scheduler.step(self.global_step)

            train_epoch_loss = np.mean(train_epoch_loss)
            print(f"Epoch {epoch} finished with average loss: {train_epoch_loss:.2f}")
            ckpt_file = os.path.join(args.ckpt_dir, "epoch%d.pth" % self.global_step)
            torch.save(self.model.state_dict(), ckpt_file)

            if train_epoch_loss < best_loss:
                best_loss = train_epoch_loss
                best_model_path = os.path.join(args.ckpt_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Model saved with loss: {best_loss:.2f}")

    def evaluate(self, epoch):
        self.model.eval()
        test_epoch_loss = []
        pbar = tqdm(self.testset, desc="Evaluating")

        with torch.no_grad():
            for test_data in self.testset:
                images, labels_sbbox, labels_mbbox, labels_lbbox, \
                true_sbboxes, true_mbboxes, true_lbboxes = test_data
                
                if args.lowlight_FLAG:
                    lowlight_param = 1
                    if random.randint(0, 2) > 0:
                        lowlight_param = random.uniform(1.5, 5)
                    input_data = np.power(images, lowlight_param)
                else:
                    input_data = images
                input_data_clean = images
                    
                input_data = input_data.to(device)
                labels_sbbox = labels_sbbox.to(device)
                labels_mbbox = labels_mbbox.to(device)
                labels_lbbox = labels_lbbox.to(device)
                true_sbboxes = true_sbboxes.to(device)
                true_mbboxes = true_mbboxes.to(device)
                true_lbboxes = true_lbboxes.to(device)
                input_data_clean = input_data_clean.to(device)

                self.model(input_data, input_data_clean)
                giou_loss, conf_loss, prob_loss, recovery_loss = self.model.compute_loss(
                    labels_sbbox, labels_mbbox, labels_lbbox, true_sbboxes, true_mbboxes, true_lbboxes
                )
                loss = giou_loss + conf_loss + prob_loss

                test_epoch_loss.append(loss.item())
                pbar.set_description(f"Eval loss: {loss.item():.2f}")
            
            test_epoch_loss = np.mean(test_epoch_loss)
            print(f"Evaluation finished with average loss: {eval_epoch_loss:.2f}")

            ckpt_file = os.path.join(args.ckpt_dir, "yolov3_test_loss=%.4f.pth" % test_epoch_loss)
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..." % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            torch.save(self.model.state_dict(), ckpt_file)


if __name__ == '__main__':
    YoloTrain().train()