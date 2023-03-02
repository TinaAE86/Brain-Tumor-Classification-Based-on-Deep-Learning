import os
import pickle
import torch
import Model
from TumorDataset import TumorDataset  # 为了能够正常读取pkl文件需要导入反序列化所需的类
import matplotlib.pyplot as plt


class TrainResNet(object):
    tumor_dataset_path = r".\pkl"
    batch_size = 4
    epoch_num = 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.BCELoss()  # 二分类的交叉熵损失函数
    train_loss = None
    ls_train_f_loss = []  # 用于存储训练过程中每个epoch中loss的变化
    test_loss = None
    ls_test_f_loss = []  # 用于存储测试过程中每个epoch中loss的变化
    ls_epoch = []

    def __init__(self):
        self.td = self.get_td()
        self.train_dataloader = self.td.train_dataloader
        self.test_dataloader = self.td.test_dataloader
        self.model = Model.ResNet()
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.5)

    def get_td(self):  # 读取td数据集
        file_r = open(os.path.join(self.tumor_dataset_path, 'TumorDataset.pkl'), 'rb')
        td = pickle.load(file_r)
        file_r.close()
        return td

    def train(self, total_epoch):
        self.ls_train_f_loss = []  # 先清零再训练
        self.ls_test_f_loss = []  # 先清零再测试
        for epoch in range(total_epoch):
            print("epoch", epoch, "开始")
            self._train()  # 计算本轮epoch中训练集上的loss
            # self._test()  # 计算本轮epoch中测试集上的loss
        self.ls_epoch = [epoch for epoch in range(total_epoch)]
        self.plot()

    def _train(self):  # train()调用，计算训练集上的loss
        f_loss = 0.0
        total_batch = 0
        for batch, data in enumerate(self.train_dataloader, 0):
            total_batch = batch + 1  # batch从0开始
            features, targets = data
            targets = targets.to(torch.float)  # 将targets中标签编码转为torch.float避免报错
            features, targets = features.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            predicts = self.model(features)
            self.train_loss = self.criterion(predicts, targets)
            f_loss += self.train_loss.item()
            self.train_loss.backward()
            self.optimizer.step()
        f_loss /= total_batch  # 将f_loss按照total_batch求平均值
        print("train: f_loss=", f_loss)
        self.ls_train_f_loss.append(f_loss)

    def _test(self):  # train()调用，计算测试集上的loss
        f_loss = 0.0
        total_batch = 0
        with torch.no_grad():
            for batch, data in enumerate(self.test_dataloader):
                total_batch = batch + 1
                features, targets = data
                targets = targets.to(torch.float)  # 将targets中标签编码转为torch.float避免报错
                features, targets = features.to(self.device), targets.to(self.device)
                predicts = self.model(features)
                self.test_loss = self.criterion(predicts, targets)
                f_loss += self.test_loss.item()
            f_loss /= total_batch
            print("test: f_loss=", f_loss)
            self.ls_test_f_loss.append(f_loss)

    def plot(self):
        plt.plot(self.ls_epoch, self.ls_train_f_loss, label='training set', color='blue')
        # plt.plot(self.ls_epoch, self.ls_test_f_loss, label='test set', color='red')
        plt.xlabel("epoch")
        plt.ylabel("f_loss")
        plt.title("loss on training set")
        plt.show()

    def test(self):
        correct = 0
        total = 0
        print("测试开始")
        with torch.no_grad():
            for __, data in enumerate(self.test_dataloader):
                total = total + 1
                features, targets = data
                targets = targets.to(torch.float)  # 将targets中标签编码转为torch.float避免报错
                features, targets = features.to(self.device), targets.to(self.device)
                predicts = self.model(features)
                if abs(predicts.item() - targets.item()) < 0.5:
                    correct = correct + 1
            print("预测准确率为", correct / total)


if __name__ == '__main__':
    trn = TrainResNet()
    trn.train(60)
    trn.test()
