import torch
import matplotlib.pyplot as plt
import tqdm as tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, eval_step=100):
        """
        model: 神经网络模型
        train_loader: 训练集的DataLoader
        val_loader: 验证集的DataLoader
        criterion: 损失函数
        optimizer: 优化器
        device: 设备 (如 "cpu" 或 "cuda")
        eval_step: 训练过程中多少个batch后验证一次
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.eval_step = eval_step

        self.model.to(self.device)

        # 用于绘图的损失和准确率历史记录（按batch记）
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
    
    def train(self, num_epochs):
        """
        num_epochs: 训练的轮数
        """
        global_step = 0
        # 训练循环
        for epoch in range(num_epochs):
            self.model.train()  # 设置模型为训练模式
            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            # 进度条显示训练进度
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_idx, batch in enumerate(self.train_loader):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # 前向传播
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    # 反向传播和优化
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    batch_loss = loss.item()
                    predicted = torch.argmax(outputs, dim=1)
                    batch_correct = (predicted == labels).sum().item()
                    batch_total = labels.size(0)
                    batch_acc = batch_correct / batch_total if batch_total > 0 else 0
                    
                    self.train_loss_history.append(batch_loss)
                    self.train_acc_history.append(batch_acc)                    
                    # 统计损失和准确率
                    running_loss += batch_loss * batch_total
                    running_corrects += batch_correct
                    running_total += batch_total
                    
                    global_step += 1

                    # 更新进度条
                    pbar.set_postfix({
                        'Loss': f"{batch_loss:.4f}",
                        'Acc': f"{batch_acc:.4f}"
                    })

                    if self.eval_step is not None and global_step % self.eval_step == 0 and self.eval_step > 0:
                        val_loss, val_acc = self.evaluate_once()
                        self.val_loss_history.append(val_loss)
                        self.val_acc_history.append(val_acc)
                        print(f"[Step {global_step}] Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

                avg_loss = running_loss / len(self.train_loader.dataset)
                avg_acc = running_corrects / running_total if running_total > 0 else 0
                print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f} Train Acc: {avg_acc:.4f}")

    def evaluate_once(self):
        """
        评估模型一次，返回损失和准确率
        """
        self.model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                predicted = torch.argmax(outputs, dim=1)
                val_corrects += (predicted == labels).sum().item()
                val_total += labels.size(0)

        avg_loss = val_loss / len(self.val_loader.dataset)
        avg_acc = val_corrects / val_total if val_total > 0 else 0
        return avg_loss, avg_acc

    def plot_loss_acc(self):
        """
        绘制训练损失和准确率的图表
        """
        import numpy as np
        
        sample_step = 1000
        
        # 采样数据点
        train_loss_sampled = self.train_loss_history[::sample_step]
        # val_loss_sampled = self.val_loss_history[::sample_step]
        train_acc_sampled = self.train_acc_history[::sample_step]
        # val_acc_sampled = self.val_acc_history[::sample_step]
        train_x_step = np.arange(0, len(self.train_loss_history), sample_step)
        
        if hasattr(self, 'eval_step') and self.eval_step is not None and self.eval_step > 0:
            val_sample_interval = max(1, sample_step // self.eval_step)
        else:
            val_sample_interval = 1
        
        val_loss_sampled = self.val_loss_history[::val_sample_interval]
        val_acc_sampled = self.val_acc_history[::val_sample_interval]
        val_x_step = np.arange(0, len(self.val_loss_history)*self.eval_step, self.eval_step)[::val_sample_interval]
        
        plt.figure(figsize=(12, 4))
        
        # 绘制损失图表
        plt.subplot(1, 2, 1)
        plt.plot(train_x_step, train_loss_sampled, label='Train Loss', marker='o')
        plt.plot(train_x_step, val_loss_sampled, label='Val Loss', marker='o')
        if len(val_loss_sampled) > 0:
            plt.plot(val_x_step, val_loss_sampled, label='Val Loss', marker='o')
        plt.title('Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制准确率图表
        plt.subplot(1, 2, 2)
        plt.plot(train_x_step, train_acc_sampled, label='Train Acc', marker='o')
        plt.plot(train_x_step, val_acc_sampled, label='Val Acc', marker='x')
        if len(val_acc_sampled) > 0:
            plt.plot(val_x_step, val_acc_sampled, label='Val Acc', marker='x') 
        plt.title('Accuracy')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
