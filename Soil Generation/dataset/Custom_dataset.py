import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from scipy import signal
import ast

class CustomVAEDataset(Dataset):
    def __init__(self, csv_file, num_folds=5, test_fold=None, split='train', resample_length=None):
        # 读取 CSV 文件
        df = pd.read_csv(csv_file)

        # 解析 'Processed Data' 列：将每行的空格分隔字符串转为长度2100的float32数组
        sequences = []
        for data_str in df['Processed Data']:
            # 将字符串拆分为浮点数列表并转换为 float32 数组
            seq = np.array(data_str.split(), dtype=np.float32)
            # 确保序列长度为2100，如不足则填充0，超出则截断
            if len(seq) != 2100:
                if len(seq) < 2100:
                    seq = np.pad(seq, (0, 2100 - len(seq)), mode='constant', constant_values=0)
                else:
                    # print("正在进行截断")
                    seq = seq[-2100:].astype(np.float32)
            sequences.append(seq)
        # 将列表转换为numpy数组 (形状: [样本数, 2100])
        sequences = np.stack(sequences)

        # 如果指定了重采样长度，则对每条序列进行重采样至新长度
        if resample_length is not None:
            resampled_sequences = []
            for seq in sequences:
                # 使用 SciPy 的信号重采样函数调整序列长度
                new_seq = signal.resample(seq, resample_length)
                resampled_sequences.append(new_seq.astype(np.float32))
            sequences = np.stack(resampled_sequences)

        # 定义需要提取的标签列
        # self.label_cols = [
        #     "Coarse", "Clay", "Sand", "Silt",
        #     "pH(CaCl2)", "pH(H2O)", "EC", "OC",
        #     "CaCO3", "P", "N", "K", "Elevation"
        # ]
        # self.label_cols = [
        #     "pH(CaCl2)", "pH(H2O)","OC",
        #     "CaCO3", "P", "N", "K"
        # ]
        # self.label_cols = [
        #     "Coarse", "Clay", "Sand", "Silt",
        #     "pH(CaCl2)", "pH(H2O)", "OC",
        #     "CaCO3", "P", "N", "K"
        # # ]
        self.label_cols = [
            "Clay", "Sand", "Silt", "pH(H2O)"
        ]

        # self.label_cols = [
        #     "Sand", "pH(H2O)"
        # ]

        # self.label_cols = ["text_embed", "Struct_embed"]

        # self.label_cols = ["text_embed"]

        # labels_matrix = []
        # for label_cols in df['text_embed']:
        #     # 将字符串拆分为浮点数列表并转换为 float32 数组
        #     col = np.array(label_cols.split(), dtype=np.float)
        #     label_cols.append(col)
        # labels_matrix = np.stack(label_cols)

        # Processed Data, Coarse, Clay, Sand, Silt, pH(CaCl2), pH(H2O), OC, CaCO3, P, N, K
        # 确保 DataFrame 包含所有标签列（若缺失则填充 NaN）
        for col in self.label_cols:
            if col not in df.columns:
                df[col] = np.nan
        # 4. 处理包含 Tensor 数据的列，将字符串表示的 Tensor 转换为 numpy 数组

        # def parse_tensor_string(x):
        #
        #     data_str = x.replace("[", "").replace("]", "").replace("...", "").strip()
        #     # 将数值部分拆分为列表
        #     values = data_str.split()
        #     # 将列表转换为浮点数类型的 numpy 数组
        #     tensor_data = np.array([float(value) for value in values], dtype=np.float32)
        #     print(tensor_data.shape)
        #     return tensor_data
        # for col in self.label_cols:
        #     df[col] = df[col].apply(parse_tensor_string)

        # 提取标签数据为 numpy 矩阵（float32类型，缺失值为 NaN）
        labels_matrix = df[self.label_cols].to_numpy(dtype=np.float32)
        # labels_matrix = df[self.label_cols]
        # labels_matrix = df[col]
        # 准备索引列表，根据 K 折参数划分训练/测试集
        self.indices = list(range(len(df)))
        if test_fold is not None:
            # 为每个样本指定折号
            if 'Point_ID' in df.columns:
                # 若存在 Point_ID，则按唯一 Point_ID 分组划分折号，确保相同 Point_ID 的样本在同一折
                unique_ids = df['Point_ID'].unique()
                id_to_fold = {pid: idx % num_folds for idx, pid in enumerate(unique_ids)}
                sample_folds = [id_to_fold[pid] for pid in df['Point_ID']]
            else:
                # 无 Point_ID 时，直接按顺序对索引取模指派折号
                sample_folds = [i % num_folds for i in range(len(df))]
            # 根据 split 参数过滤出训练或测试集的索引
            if split == 'train':
                self.indices = [i for i, fold in enumerate(sample_folds) if fold != test_fold]
            elif split == 'test':
                self.indices = [i for i, fold in enumerate(sample_folds) if fold == test_fold]
            else:
                raise ValueError("split 参数必须是 'train' 或 'test'")

        # 保存最终的序列数据和标签矩阵
        self.data = sequences  # numpy 数组，形状: [N, 序列长度]
        self.labels_matrix = labels_matrix  # numpy 数组，形状: [N, 13]

    def __len__(self):
        # 返回当前数据集包含的样本数量（根据划分后的 indices 列表）
        return len(self.indices)

    def __getitem__(self, idx):
        # 获取实际样本索引
        actual_idx = self.indices[idx]
        # 提取光谱序列数据
        x = self.data[actual_idx]
        # 提取对应的标签值并构建字典

        label_values = self.labels_matrix[actual_idx]
        label_dict = {key: label_values[i] for i, key in enumerate(self.label_cols)}
        # label_dict = {key: label_values for i, key in enumerate(self.label_cols)}
        # 返回光谱序列 x 以及标签字典 label_dict
        return x, label_dict
if __name__ == '__main__':
    # Original dataset
    # path = '/data/0shared/MIMIC/physionet.org/files/mimic-iv-ecg/1.0/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
    # data = MIMIC_IV_ECG_Dataset(dataset_path=path, resample_length=1024, demo_label=True)

    # VAE encoded dataset
    vae_path = '../prerequisites/LUCAS2015.csv'
    # vae_path = 'mimic_vae.pt'
    data = CustomVAEDataset(vae_path, usage='test')
    # data = DictDataset(vae_path)

    # print(len(data))
    print(data[397][1])
    # print(data[397][1])

    # print(type(data[0][1]['subject_id']))

    # dataloader = DataLoader(data, 512)
    # test reading speed
    for idx, (X, y) in enumerate(tqdm.tqdm(data)):
        pass