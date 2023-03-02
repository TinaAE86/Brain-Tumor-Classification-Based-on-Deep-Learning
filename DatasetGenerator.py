import os
import SimpleITK as sitk
import numpy as np
import pickle


class DatasetGenerator(object):
    def __init__(self):
        self.dict_dataset = {'feature': [], 'target': []}  # dict_dataset的格式
        # 上面的target中，LGG编码为0，HGG编码为1
        self.t_ori_shape = (155, 240, 240)  # 三维图片的原始大小，t表示tuple
        self.tumor_data_path = r'.\tumor_data'
        self.tumor_dict_dataset_path = r'.\pkl'

    @staticmethod
    def get_mid(image: np.array) -> int:
        """
        获得肿瘤所在的切片层
        """
        start = 0
        end = image.shape[0]
        for n_slice in range(image.shape[0]):
            slice_n = image[n_slice, :, :]
            s_unique = np.unique(slice_n)
            if s_unique.shape == (4,):
                start = n_slice
                break
        for n_slice in range(image.shape[0] - 1, 0, -1):  # 起始位置为第154层，终止为0层，步长为-1，
            slice_n = image[n_slice, :, :]
            s_unique = np.unique(slice_n)
            if s_unique.shape == (4,):
                end = n_slice
                break
        i_mid = int((start + end) / 2)
        return i_mid

    def get_one_data(self, arr_img_t1ce, arr_img_seg) -> np.array:
        """
        由输入的seg和t1ce生成一张1*240*240的含肿瘤的二维切片arr，并完成归一化
        """
        i_mid = self.get_mid(arr_img_seg)
        arr_img_t1ce = arr_img_t1ce[i_mid, :, :]
        # 下面完成归一化
        arr_img_t1ce = (arr_img_t1ce - np.min(arr_img_t1ce)) / (np.max(arr_img_t1ce) - np.min(arr_img_t1ce))
        # 补上切片层维度1
        arr_img_t1ce = arr_img_t1ce[np.newaxis, :, :]
        return arr_img_t1ce

    def crt_dataset(self) -> dict:
        print("开始创建arr数据集...")
        for s_root, s_dirs, s_files in os.walk(self.tumor_data_path):
            if len(s_dirs) != 0:
                continue
            else:  # s_dirs长度为0时，说明进入了最内层目录，有5个.nii.gz文件
                arr_img_t1ce = np.zeros(self.t_ori_shape)
                arr_img_seg = np.zeros(self.t_ori_shape)
                for s_file in s_files:
                    if s_file.endswith('t1ce.nii.gz'):
                        arr_img_t1ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(s_root, s_file)))
                    elif s_file.endswith('seg.nii.gz'):
                        arr_img_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(s_root, s_file)))
                arr_img_t1ce = self.get_one_data(arr_img_t1ce, arr_img_seg)
                self.dict_dataset['feature'].append(arr_img_t1ce)
                s_target = os.path.split(os.path.split(s_root)[0])[1]
                if s_target == 'LGG':
                    self.dict_dataset['target'].append(0)
                elif s_target == 'HGG':
                    self.dict_dataset['target'].append(1)
                    print("数据集分类标签获取异常")
        print("数据集创建完成")
        return self.dict_dataset


if __name__ == '__main__':
    dg = DatasetGenerator()
    dg.crt_dataset()
    file_w = open(os.path.join(dg.tumor_dict_dataset_path,
                               'dict_dataset.pkl'),
                  'wb')
    pickle.dump(dg.dict_dataset, file_w)  # 将数据集存入.pkl文件
    file_w.close()
