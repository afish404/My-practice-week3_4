import torch
import matplotlib.pyplot as plt

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        eval_step=100,
        early_stopping=None,
        early_stopping_kwargs=None,
        model_checkpoint=None,  # 新增参数：ModelCheckpoint对象
        tensorboard_callback=None,  # 新增参数：TensorBoardCallback对象
    ):
        """
        model: 神经网络模型
        train_loader: 训练集的DataLoader
        val_loader: 验证集的DataLoader
        criterion: 损失函数
        optimizer: 优化器
        device: 设备 (如 "cpu" 或 "cuda")
        eval_step: 训练过程中多少个batch后验证一次
        early_stopping: EarlyStopping对象 或 None 禁用早停
        early_stopping_kwargs: 如果不传early_stopping对象，可以传此参数作为EarlyStopping的参数(dict)
        model_checkpoint: ModelCheckpoint对象，用于保存模型，默认为None
        tensorboard_callback: TensorBoardCallback对象，用于写入Tensorboard，默认为None
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.eval_step = eval_step

        self.model.to(self.device)

        # 新增: ModelCheckpoint和TensorBoardCallback
        self.model_checkpoint = model_checkpoint
        self.tensorboard_callback = tensorboard_callback

        # 用于绘图的损失和准确率历史记录（按batch记）
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # early stopping
        if early_stopping is not None:
            self.early_stopping = early_stopping
        elif early_stopping_kwargs is not None:
            self.early_stopping = EarlyStopping(**early_stopping_kwargs)
        else:
            self.early_stopping = None

    def train(self, num_epochs):
        global_step = 0
        stop_training = False
        for epoch in range(num_epochs):
            if stop_training:
                print(f"Early stopping triggered. Stopping at epoch {epoch}.")
                break
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for batch in self.train_loader:
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

                    # --- 新增模型保存 ---
                    if self.model_checkpoint is not None:
                        # monitor参数判据: ModelCheckpoint 设置为监控val_loss(val_acc)
                        if self.model_checkpoint.monitor == 'val_loss':
                            self.model_checkpoint(val_loss, self.model, epoch=epoch)
                        elif self.model_checkpoint.monitor == 'val_acc':
                            self.model_checkpoint(val_acc, self.model, epoch=epoch)
                        else:  # 默认用val_loss
                            self.model_checkpoint(val_loss, self.model, epoch=epoch)

                    # --- 新增TensorBoard写入 ---
                    if self.tensorboard_callback is not None:
                        # 这里支持记录loss/acc
                        self.tensorboard_callback(
                            global_step,
                            loss=batch_loss,
                            val_loss=val_loss,
                            acc=batch_acc,
                            val_acc=val_acc,
                            lr=self.optimizer.param_groups[0]['lr'] if hasattr(self.optimizer, 'param_groups') else None,
                        )

                    # early stopping 检查，默认监控val_loss
                    if self.early_stopping is not None:
                        if self.early_stopping.mode == 'min':
                            should_stop = self.early_stopping.step(val_loss)
                        else:
                            should_stop = self.early_stopping.step(val_acc)
                        if should_stop:
                            print(f"Early stopping triggered at step {global_step}.")
                            stop_training = True
                            break

            if stop_training:
                break
            # 可以在一个epoch末尾也再次eval一次（可选）
            # val_loss, val_acc = self.evaluate_once()
            # self.val_loss_history.append(val_loss)
            # self.val_acc_history.append(val_acc)

            avg_train_loss = train_loss / len(self.train_loader.dataset)
            avg_train_acc = train_correct / train_total if train_total > 0 else 0
            print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}  Train Acc: {avg_train_acc:.4f}")

    def train_regression(self, num_epochs):
        """
        回归训练函数，包含早停和模型保存功能
        """
        import torch
        
        self.model.train()
        global_step = 0
        stop_training = False
        
        for epoch in range(num_epochs):
            train_loss = 0.0
            train_total = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                batch_loss = loss.item()
                train_loss += batch_loss * inputs.size(0)
                train_total += inputs.size(0)
                
                # 记录训练损失历史
                self.train_loss_history.append(batch_loss)
                
                global_step += 1
                
                # 定期评估
                if self.eval_step is not None and self.eval_step > 0 and global_step % self.eval_step == 0:
                    val_loss = self.evaluate_regression_once()
                    self.val_loss_history.append(val_loss)
                    print(f"[Step {global_step}] Val Loss: {val_loss:.4f}")
                    
                    # 模型保存
                    if self.model_checkpoint is not None:
                        self.model_checkpoint(val_loss, self.model, epoch=epoch)
                    
                    # TensorBoard记录
                    if self.tensorboard_callback is not None:
                        self.tensorboard_callback.add_loss_scalars(
                            global_step, 
                            loss=batch_loss, 
                            val_loss=val_loss
                        )
                        if hasattr(self.optimizer, 'param_groups'):
                            self.tensorboard_callback.add_lr_scalars(
                                global_step, 
                                learning_rate=self.optimizer.param_groups[0]['lr']
                            )
                    
                    # 早停检查（基于val_loss）
                    if self.early_stopping is not None:
                        should_stop = self.early_stopping.step(val_loss)
                        if should_stop:
                            print(f"Early stopping triggered at step {global_step}.")
                            stop_training = True
                            break
            
            if stop_training:
                break
                
            avg_train_loss = train_loss / train_total if train_total > 0 else 0
            print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}")
    
    def evaluate_regression_once(self):
        """评估回归模型的损失"""
        self.model.eval()
        val_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        avg_val_loss = val_loss / total_samples if total_samples > 0 else 0
        return avg_val_loss

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

    def plot_curves(self, sample_step=1000):
        """
        每sample_step步采样一次训练集loss/acc进行绘图，验证集采用每sample_step/eval_step采样绘图。
        如果train_acc_history为空，只绘制损失图。
        """
        import numpy as np

        # 采样训练曲线
        train_loss_sampled = self.train_loss_history[::sample_step]
        train_x_steps = np.arange(0, len(self.train_loss_history), sample_step)

        # 采样验证曲线（每sample_step/eval_step拿一个点）
        if hasattr(self, 'eval_step') and self.eval_step and self.eval_step > 0:
            val_sample_interval = max(1, sample_step // self.eval_step)
        else:
            val_sample_interval = 1  # fallback

        val_loss_sampled = self.val_loss_history[::val_sample_interval]
        # 验证集点的x坐标（每eval_step加一个，采样后同样间隔拉伸）
        val_x_steps = np.arange(0, len(self.val_loss_history)*self.eval_step, self.eval_step)[::val_sample_interval]

        # 检查是否有准确率数据
        has_acc_data = len(self.train_acc_history) > 0

        if has_acc_data:
            # 有准确率数据，绘制两个子图
            train_acc_sampled = self.train_acc_history[::sample_step]
            val_acc_sampled = self.val_acc_history[::val_sample_interval]
            
            plt.figure(figsize=(12, 4))  # 画布

            # ------------------- 损失曲线 -------------------
            plt.subplot(1, 2, 1)
            plt.plot(train_x_steps, train_loss_sampled, label='Train Loss')
            if len(val_loss_sampled) > 0:
                plt.plot(val_x_steps, val_loss_sampled, label='Validation Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title('Loss Curves')
            plt.legend()
            plt.grid(True)

            # ------------------- 准确率曲线 -------------------
            plt.subplot(1, 2, 2)
            plt.plot(train_x_steps, train_acc_sampled, label='Train Acc',)
            if len(val_acc_sampled) > 0:
                plt.plot(val_x_steps, val_acc_sampled, label='Validation Acc')
            plt.xlabel('Batch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Curves')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
        else:
            # 没有准确率数据，只绘制损失图
            plt.figure(figsize=(8, 4))  # 画布

            # ------------------- 损失曲线 -------------------
            plt.plot(train_x_steps, train_loss_sampled, label='Train Loss')
            if len(val_loss_sampled) > 0:
                plt.plot(val_x_steps, val_loss_sampled, label='Validation Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title('Loss Curves')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()

        plt.show()


# INSERT_YOUR_CODE
class EarlyStopping:
    """
    提前停止训练（Early Stopping）：
    在验证集损失在设定patience轮内没有提升，则提前终止训练。
    """

    def __init__(self, patience=5, min_delta=0.01, mode='min'):
        """
        参数:
        patience (int): 容忍的epoch数，如果在patience个epoch内指标未提升，则停止训练
        min_delta (float): 指标提升的最小变化量，只有大于这个才认为有提升
        mode (str): 'min' 表示希望监控指标越小越好（如loss），'max' 表示指标越大越好（如accuracy）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None #记录最佳指标
        self.counter = 0
        self.early_stop = False

        if self.mode not in ['min', 'max']:
            raise ValueError("mode只能是'min'或'max'")

        if self.mode == 'min':
            self.monitor_op = lambda curr, best: curr < best - self.min_delta #监控指标小于最佳指标,val_loss越小越好
        else:
            self.monitor_op = lambda curr, best: curr > best + self.min_delta #监控指标大于最佳指标,val_acc越大越好

    def step(self, current):
        """
        应在每次验证后调用step()方法，传入当前的验证集监控指标（如val_loss或者val_acc）。
        返回True，如果训练应提前停止；否则返回False。
        """
        if self.best_score is None:
            self.best_score = current #记录最佳指标
            self.counter = 0 #重置计数器
            self.early_stop = False
            return False

        if self.monitor_op(current, self.best_score): #监控指标小于最佳指标,val_loss越小越好
            self.best_score = current #记录最佳指标
            self.counter = 0 #重置计数器
            self.early_stop = False
        else:
            self.counter += 1 #计数器加1
            if self.counter >= self.patience:
                self.early_stop = True #训练应提前停止  
        return self.early_stop

    def reset(self):
        """重置early stopping状态"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False

     
class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, mode='min', min_delta=0.01):
        """
        参数:
        filepath (str): 保存模型的文件路径，可以包含格式化字符串，如 'model_epoch{epoch}_val_loss{val_loss:.4f}.pt'
        monitor (str): 监控的指标名称，仅用于信息记录
        save_best_only (bool): 是否只保存性能最优的模型
        mode (str): 'min' 或 'max'，确定“更好”是数值更小或更大
        min_delta (float): 指标提升的最小变化量
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None

        if mode not in ['min', 'max']:
            raise ValueError("mode只能是'min'或者'max'")

        if self.mode == 'min':
            self.monitor_op = lambda curr, best: curr < best - self.min_delta
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda curr, best: curr > best + self.min_delta
            self.best_score = -float('inf')

    def __call__(self, current, model, epoch=None):
        """
        current: 当前验证集监控指标（如val_loss或val_acc）
        model: 要保存的模型（必须有.state_dict()方法）
        epoch: 当前epoch编号（可选，用于文件命名）
        """
        import os
        # 保存模型
        if self.save_best_only:
            if self.monitor_op(current, self.best_score):
                filepath = os.path.dirname(self.filepath) # 获取文件夹路径
                os.makedirs(filepath, exist_ok=True) # 创建文件夹
                self.best_score = current
                # 保存
                file_name = self.filepath.format(epoch='best')
                self._save_model(model, file_name) # 保存模型
                return True  # 有保存
            return False  # 无保存
        else:
            # 每次都保存
            if epoch is not None:
                filepath = self.filepath.format(epoch=epoch, **{self.monitor: current})
            else:
                filepath = self.filepath.format(**{self.monitor: current})
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self._save_model(model, filepath)
            return True

    def _save_model(self, model, filepath):
        """保存模型，假设是PyTorch模型"""
        # 只实现通用保存接口，具体框架层可根据需求调整
        try:
            import torch
            torch.save(model.state_dict(), filepath)
        except ImportError:
            raise RuntimeError("需要torch库用于保存模型")

from torch.utils.tensorboard import SummaryWriter


class TensorBoardCallback:
    def __init__(self, log_dir, flush_secs=10):
        """
        Args:
            log_dir (str): dir to write log.
            flush_secs (int, optional): write to dsk each flush_secs seconds. Defaults to 10.
        """
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs) # 实例化SummaryWriter, log_dir是log存放路径，flush_secs是每隔多少秒写入磁盘

    def draw_model(self, model, input_shape):#graphs
        self.writer.add_graph(model, input_to_model=torch.randn(input_shape)) # 画模型图
        
    def add_loss_scalars(self, step, loss, val_loss):
        self.writer.add_scalars(
            main_tag="training/loss", 
            tag_scalar_dict={"loss": loss, "val_loss": val_loss},
            global_step=step,
            ) # 画loss曲线, main_tag是主tag，tag_scalar_dict是子tag，global_step是步数
        
    def add_acc_scalars(self, step, acc, val_acc):
        self.writer.add_scalars(
            main_tag="training/accuracy",
            tag_scalar_dict={"accuracy": acc, "val_accuracy": val_acc},
            global_step=step,
        ) # 画acc曲线, main_tag是主tag，tag_scalar_dict是子tag，global_step是步数
        
    def add_lr_scalars(self, step, learning_rate):
        self.writer.add_scalars(
            main_tag="training/learning_rate",
            tag_scalar_dict={"learning_rate": learning_rate},
            global_step=step,
        ) # 画lr曲线, main_tag是主tag，tag_scalar_dict是子tag，global_step是步数
    
    def __call__(self, step, **kwargs):
        # add loss,把loss，val_loss取掉，画loss曲线
        loss = kwargs.pop("loss", None)
        val_loss = kwargs.pop("val_loss", None)
        if loss is not None and val_loss is not None:
            self.add_loss_scalars(step, loss, val_loss) # 画loss曲线
        # add acc
        acc = kwargs.pop("acc", None)
        val_acc = kwargs.pop("val_acc", None)
        if acc is not None and val_acc is not None:
            self.add_acc_scalars(step, acc, val_acc) # 画acc曲线
        # add lr
        learning_rate = kwargs.pop("lr", None)
        if learning_rate is not None:
            self.add_lr_scalars(step, learning_rate) # 画lr曲线
