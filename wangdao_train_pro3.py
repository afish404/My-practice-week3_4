import torch
import matplotlib.pyplot as plt

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
        global_step = 0
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for batch_idx, batch in enumerate(self.train_loader):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # 当前batch损失和准确率
                batch_loss = loss.item()
                predicted = torch.argmax(outputs, dim=1)
                batch_correct = (predicted == targets).sum().item()
                batch_total = targets.size(0)
                batch_acc = batch_correct / batch_total if batch_total > 0 else 0

                # 记录本batch的损失和准确率
                self.train_loss_history.append(batch_loss)
                self.train_acc_history.append(batch_acc)

                train_loss += batch_loss * batch_total
                train_correct += batch_correct
                train_total += batch_total

                global_step += 1

                # eval_step的整数倍时进行一次验证集计算
                if self.eval_step is not None and self.eval_step > 0 and global_step % self.eval_step == 0:
                    val_loss, val_acc = self.evaluate_once()
                    self.val_loss_history.append(val_loss)
                    self.val_acc_history.append(val_acc)
                    print(f"[Step {global_step}] Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

            # 可以在一个epoch末尾也再次eval一次（可选）
            # val_loss, val_acc = self.evaluate_once()
            # self.val_loss_history.append(val_loss)
            # self.val_acc_history.append(val_acc)

            avg_train_loss = train_loss / len(self.train_loader.dataset)
            avg_train_acc = train_correct / train_total if train_total > 0 else 0
            print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}  Train Acc: {avg_train_acc:.4f}")

    def evaluate_once(self):
        """评估整个val_loader的损失和准确率"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        avg_val_loss = val_loss / len(self.val_loader.dataset)
        val_acc = correct / total if total > 0 else 0
        return avg_val_loss, val_acc

    def plot_curves(self):
        """
        每1000步采样一次训练集loss/acc进行绘图，验证集采用每sample_step/eval_step采样绘图。
        """
        import numpy as np

        sample_step = 1000  # 采样间隔

        # 采样训练曲线
        train_loss_sampled = self.train_loss_history[::sample_step]
        train_acc_sampled = self.train_acc_history[::sample_step]
        train_x_steps = np.arange(0, len(self.train_loss_history), sample_step)

        # 采样验证曲线（每sample_step/eval_step拿一个点）
        if hasattr(self, 'eval_step') and self.eval_step and self.eval_step > 0:
            val_sample_interval = max(1, sample_step // self.eval_step)
        else:
            val_sample_interval = 1  # fallback

        val_loss_sampled = self.val_loss_history[::val_sample_interval]
        val_acc_sampled = self.val_acc_history[::val_sample_interval]
        # 验证集点的x坐标（每eval_step加一个，采样后同样间隔拉伸）
        val_x_steps = np.arange(0, len(self.val_loss_history)*self.eval_step, self.eval_step)[::val_sample_interval]

        plt.figure(figsize=(12, 4))  # 画布

        # ------------------- 损失曲线 -------------------
        plt.subplot(1, 2, 1)
        plt.plot(train_x_steps, train_loss_sampled, label='Train Loss', marker='o')
        if len(val_loss_sampled) > 0:
            plt.plot(val_x_steps, val_loss_sampled, label='Validation Loss', marker='x')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)

        # ------------------- 准确率曲线 -------------------
        plt.subplot(1, 2, 2)
        plt.plot(train_x_steps, train_acc_sampled, label='Train Acc', marker='o')
        if len(val_acc_sampled) > 0:
            plt.plot(val_x_steps, val_acc_sampled, label='Validation Acc', marker='x')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
