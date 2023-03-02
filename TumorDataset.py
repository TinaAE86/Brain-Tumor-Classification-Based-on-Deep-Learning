import os
import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class TumorDataset(Dataset):
    def __init__(self):
        # 将输入的列表构成的字典dict_dataset转化为元组构成的列表self.dataset
        self.tumor_dict_dataset_addr = r'.\pkl\dict_dataset.pkl'
        self.tumor_dataset_path = r'.\pkl'  # 此路径用于保存torch可以直接使用的数据集
        self.dataset = []  # 列表的列表，每个子列表保存单个样本，feature部分为tensor类型，target部分为分类编码
        self.train_set = []
        self.test_set = []
        self.train_dataloader = None
        self.test_dataloader = None

    def __getitem__(self, index) -> dict:
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset), len(self.train_set), len(self.test_set)

    def crt_torch_dataset(self):
        print("开始创建torch总数据集...")
        file_r = open(self.tumor_dict_dataset_addr, 'rb')
        dict_dataset = pickle.load(file_r)
        for i in range(len(dict_dataset['target'])):
            # feature为tensor图片，target为分类编码
            feature = torch.from_numpy(dict_dataset['feature'][i])
            feature = feature.type(torch.cuda.FloatTensor)  # double转float，避免训练时前向传播报错
            target = dict_dataset['target'][i]
            t_sample = (feature, target)  # 元组t_sample用于存储单个样本的feature和target
            self.dataset.append(t_sample)
            del t_sample  # 由于元组不可修改，故需要删除元组，便于下一轮循环继续使用同名元组存放样本
        print("torch总数据集创建完成")
        print("开始划分训练集和测试集...")
        self.train_set, self.test_set = train_test_split(self.dataset, test_size=0.1, random_state=1)
        print("训练集和测试集划分完成")
        print("开始生成dataloader...")
        self.train_dataloader = DataLoader(self.train_set, batch_size=16)
        self.test_dataloader = DataLoader(self.test_set, batch_size=1)
        print("dataloader生成完成")


if __name__ == '__main__':
    td = TumorDataset()
    td.crt_torch_dataset()
    # 存储TumorDataset()类对象td
    file_w = open(os.path.join(td.tumor_dataset_path, 'TumorDataset.pkl'), 'wb')
    pickle.dump(td, file_w)
    print("数据集、训练集、测试集中中样本个数分别为", td.__len__())
    file_w.close()
