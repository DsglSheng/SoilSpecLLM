import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Spectral(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='spectral_data.csv',
                 target=None, scale=False, percent=100,
                 wavelength_range=(400, 2499), resample=None):
        # 设置序列长度和预测长度
        if size == None:
            self.seq_len = 1400
            self.label_len = 0  # 重叠部分长度
            self.pred_len = 700
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # 打印参数信息
        print(f"初始化{flag}数据集: seq_len={self.seq_len}, label_len={self.label_len}, pred_len={self.pred_len}")

        # 初始化数据集类型
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale  # 光谱数据已在0-1范围，不需要缩放
        self.percent = percent
        self.wavelength_range = wavelength_range  # 光谱波长范围
        self.resample = resample  # 可选的重采样间隔

        self.root_path = root_path
        self.data_path = data_path

        # 添加CSV验证步骤
        self.__validate_csv__()

        # 读取数据
        self.__read_data__()

        # 设置输入维度
        self.enc_in = self.data_x.shape[-1]

        # 记录样本总数
        print(f"数据集{flag}初始化完成: 样本数={len(self.data_x)}, 波长点数={self.data_x.shape[1]}")

    def __validate_csv__(self):
        """验证CSV文件的结构和内容"""
        # 检查是否在多进程环境中
        rank = int(os.environ.get('RANK', 0))

        # 只在主进程(rank 0)上进行详细验证，避免多进程竞争
        if rank != 0:
            return

        file_path = os.path.join(self.root_path, self.data_path)
        print(f"验证CSV文件: {file_path}")

        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV文件不存在: {file_path}")

            # 获取文件大小
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            print(f"CSV文件大小: {file_size_mb:.2f} MB")

            # 检查波长范围是否合理
            wl_min, wl_max = self.wavelength_range
            expected_wavelength_count = wl_max - wl_min + 1
            print(f"预期波长范围: {wl_min}-{wl_max}, 预期波长点数: {expected_wavelength_count}")

            # 只读取前几行来确定文件结构
            df_head = pd.read_csv(file_path, nrows=5)
            cols = list(df_head.columns)

            # 检查列数
            print(f"CSV文件列数: {len(cols)}")

            # 检查是否有波长列
            has_wavelength_col = 'wavelength' in cols or 'Wavelength' in cols

            if has_wavelength_col:
                wavelength_col = 'wavelength' if 'wavelength' in cols else 'Wavelength'
                print(f"检测到波长列: {wavelength_col}")

                # 检查文件行数 (对应波长点数)
                # 使用更高效的方式计数行数
                with open(file_path, 'r') as f:
                    # 跳过标题行
                    next(f)
                    line_count = sum(1 for _ in f)

                print(f"CSV文件行数 (不含标题): {line_count}")

                # 验证行数是否接近预期的波长点数
                if abs(line_count - expected_wavelength_count) > expected_wavelength_count * 0.1:  # 允许10%的误差
                    print(f"警告: 文件行数 ({line_count}) 与预期波长点数 ({expected_wavelength_count}) 相差较大")

                # 检查波长范围是否正确
                # 读取第一行和最后一行的波长值进行验证
                # 只读取第一行和最后一行，避免加载整个文件
                first_df = pd.read_csv(file_path, usecols=[wavelength_col], nrows=1)
                first_wavelength = first_df[wavelength_col].iloc[0]
                
                # 使用skiprows跳过中间行，只读最后一行
                # skiprows参数：跳过第1行到倒数第2行（保留标题行和最后一行）
                last_df = pd.read_csv(file_path, usecols=[wavelength_col], skiprows=lambda x: x != 0 and x < line_count)
                last_wavelength = last_df[wavelength_col].iloc[-1]

                print(f"首行波长值: {first_wavelength}, 末行波长值: {last_wavelength}")

                # 检查实际波长范围与预期是否匹配
                if abs(first_wavelength - wl_min) > 10 or abs(last_wavelength - wl_max) > 10:
                    print(
                        f"警告: 实际波长范围 ({first_wavelength}-{last_wavelength}) 与预期 ({wl_min}-{wl_max}) 不匹配")

                # 验证样本数量 (列数 - 1)
                sample_count = len(cols) - 1
                print(f"样本数量: {sample_count}")

                # 验证总序列长度与可用波长点数的关系
                total_required_len = self.seq_len + self.pred_len
                if line_count < total_required_len:
                    print(f"警告: 波长点数 ({line_count}) 小于所需总序列长度 ({total_required_len})")
                else:
                    print(f"验证通过: 波长点数 ({line_count}) >= 所需总序列长度 ({total_required_len})")

            else:
                # 尝试从列名解析波长值
                try:
                    wavelengths = [float(col) for col in cols]
                    min_wl = min(wavelengths)
                    max_wl = max(wavelengths)
                    print(f"从列名检测到的波长范围: {min_wl}-{max_wl}, 波长点数: {len(wavelengths)}")

                    # 检查列名波长范围与预期的匹配度
                    if abs(min_wl - wl_min) > 10 or abs(max_wl - wl_max) > 10:
                        print(f"警告: 实际波长范围 ({min_wl}-{max_wl}) 与预期 ({wl_min}-{wl_max}) 不匹配")

                    # 验证行数 (样本数)
                    # 使用更高效的方式计数行数
                    with open(file_path, 'r') as f:
                        # 跳过标题行
                        next(f)
                        line_count = sum(1 for _ in f)

                    print(f"样本数量 (行数): {line_count}")

                    # 验证总序列长度与可用波长点数的关系
                    total_required_len = self.seq_len + self.pred_len
                    if len(wavelengths) < total_required_len:
                        print(f"警告: 波长点数 ({len(wavelengths)}) 小于所需总序列长度 ({total_required_len})")
                    else:
                        print(f"验证通过: 波长点数 ({len(wavelengths)}) >= 所需总序列长度 ({total_required_len})")

                except ValueError:
                    print("无法从列名解析波长值，请确保数据格式正确")

                    # 检查行列数作为备选验证
                    with open(file_path, 'r') as f:
                        # 跳过标题行
                        next(f)
                        line_count = sum(1 for _ in f)

                    print(f"CSV文件行数 (不含标题): {line_count}, 列数: {len(cols)}")

                    # 验证序列长度与数据形状的关系
                    print(f"配置的序列长度: seq_len={self.seq_len}, pred_len={self.pred_len}")
                    print("请确保数据维度与序列长度设置相匹配")

            print("CSV验证完成")

        except Exception as e:
            print(f"CSV验证出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def __read_data__(self):
        """读取和处理数据"""
        # 使用分块读取代替一次性加载整个文件
        file_path = os.path.join(self.root_path, self.data_path)

        try:
            # 首先使用较小的chunksize读取文件头以获取列信息
            with pd.read_csv(file_path, chunksize=5) as reader:
                df_head = next(reader)
                cols = list(df_head.columns)
                has_wavelength_col = 'wavelength' in cols or 'Wavelength' in cols

                if has_wavelength_col:
                    wavelength_col = 'wavelength' if 'wavelength' in cols else 'Wavelength'
                    sample_cols = [col for col in cols if col != wavelength_col]
                else:
                    # 假设列名是波长值或者没有特定的波长列
                    try:
                        wavelengths = [float(col) for col in df_head.columns]
                        wl_min, wl_max = self.wavelength_range
                        sample_cols = [col for col in df_head.columns
                                       if wl_min <= float(col) <= wl_max]
                        self.wavelengths = np.array([float(col) for col in sample_cols])
                    except:
                        print("警告：列名无法转换为波长值，使用全部数据")
                        sample_cols = list(df_head.columns)
                        if 'sample_id' in sample_cols or 'Sample_ID' in sample_cols:
                            id_col = 'sample_id' if 'sample_id' in sample_cols else 'Sample_ID'
                            sample_cols.remove(id_col)
                        self.wavelengths = np.arange(len(sample_cols))

            print(f"数据文件包含 {len(cols)} 列")

            # 确定要读取的列
            if has_wavelength_col:
                usecols = [wavelength_col] + sample_cols
            else:
                usecols = sample_cols

            # 使用更高效的数据类型
            dtype_dict = {col: 'float32' for col in usecols}

            # 确定总行数，以便进行数据集划分
            print("计算CSV文件行数...")
            with open(file_path, 'r') as f:
                total_rows = sum(1 for _ in f) - 1  # 减去标题行

            print(f"CSV文件总行数: {total_rows}")

            # 确定样本数量 (针对不同格式)
            if has_wavelength_col:
                # 行是波长，列是样本
                num_samples = len(sample_cols)
                print(f"波谱数据格式: 行=波长({total_rows}), 列=样本({num_samples})")
            else:
                # 行是样本，列是波长
                num_samples = total_rows
                print(f"波谱数据格式: 行=样本({num_samples}), 列=波长({len(sample_cols)})")

            # 生成随机索引进行训练/验证/测试分割
            np.random.seed(42)  # 固定随机种子以确保可重复性
            all_indices = np.random.permutation(num_samples)

            # 按比例分配数据
            num_train = int(num_samples * 0.7)
            num_test = int(num_samples * 0.2)
            num_vali = num_samples - num_train - num_test

            train_indices = all_indices[:num_train]
            vali_indices = all_indices[num_train:num_train + num_vali]
            test_indices = all_indices[num_train + num_vali:]

            # 根据集合类型选择相应的索引
            if self.set_type == 0:  # train
                selected_indices = train_indices
            elif self.set_type == 1:  # val
                selected_indices = vali_indices
            else:  # test
                selected_indices = test_indices

            # 应用百分比采样
            if self.set_type == 0 and self.percent < 100:
                num_selected = len(selected_indices) * self.percent // 100
                selected_indices = selected_indices[:num_selected]

            print(f"当前数据集({['train', 'val', 'test'][self.set_type]})样本数: {len(selected_indices)}")

            if has_wavelength_col:
                # 对于波长为行的数据格式，需要分块读取后转置
                use_cols = [wavelength_col] + [sample_cols[i] for i in selected_indices if i < len(sample_cols)]

                # 过滤波长范围
                wl_min, wl_max = self.wavelength_range

                # 首先单独读取完整的波长数据
                print("读取完整波长数据...")
                wavelength_df = pd.read_csv(file_path, usecols=[wavelength_col])
                wavelength_df = wavelength_df[(wavelength_df[wavelength_col] >= wl_min) &
                                              (wavelength_df[wavelength_col] <= wl_max)]
                self.wavelengths = wavelength_df[wavelength_col].values
                print(f"完整波长数据: 范围={self.wavelengths[0]}-{self.wavelengths[-1]}, 点数={len(self.wavelengths)}")

                # 分块读取并处理样本数据
                chunks = []
                for chunk in pd.read_csv(file_path, usecols=use_cols, dtype=dtype_dict, chunksize=1000):
                    # 过滤波长范围
                    chunk = chunk[(chunk[wavelength_col] >= wl_min) & (chunk[wavelength_col] <= wl_max)]

                    # 删除波长列，保留样本数据
                    chunk = chunk.drop(columns=[wavelength_col])

                    chunks.append(chunk)
                    del chunk  # 释放内存

                # 合并所有块并转置
                if chunks:
                    df_data = pd.concat(chunks)
                    data = df_data.values.T  # 转置，使每行代表一个样本
                    self.data_x = data
                    self.data_y = data
                else:
                    raise ValueError("没有数据块被加载，请检查波长范围")

            else:
                # 对于样本为行的数据格式，可以直接选择需要的行索引
                # 这里我们需要分块读取，然后只保留需要的样本行

                # 创建一个集合，用于快速查找所需行索引
                selected_rows = set(selected_indices)

                chunks = []
                current_row = 0

                for chunk in pd.read_csv(file_path, usecols=sample_cols, dtype='float32', chunksize=1000):
                    # 选择当前块中需要的行
                    selected_chunk_rows = []

                    for i, row in enumerate(chunk.itertuples(index=False)):
                        if current_row in selected_rows:
                            selected_chunk_rows.append(row)
                        current_row += 1

                    if selected_chunk_rows:
                        chunks.append(pd.DataFrame(selected_chunk_rows, columns=sample_cols))

                    del chunk  # 释放内存

                # 合并所选行
                if chunks:
                    df_data = pd.concat(chunks)
                    self.data_x = df_data.values
                    self.data_y = self.data_x.copy()
                else:
                    raise ValueError("没有数据块被加载，请检查选择的索引")

            # 应用重采样（如果需要）
            if self.resample is not None and hasattr(self, 'wavelengths'):
                print(f"应用重采样, 间隔={self.resample}")
                # 创建重采样后的波长序列
                wl_min, wl_max = self.wavelength_range
                new_wavelengths = np.arange(wl_min, wl_max + 1, self.resample)

                # 对每个样本进行插值
                resampled_data = np.zeros((len(self.data_x), len(new_wavelengths)))

                for i in range(len(self.data_x)):
                    # 创建插值函数
                    f = interp1d(self.wavelengths, self.data_x[i], kind='linear',
                                 bounds_error=False, fill_value='extrapolate')

                    # 应用插值
                    resampled_data[i] = f(new_wavelengths)

                # 更新数据和波长
                self.data_x = resampled_data
                self.data_y = resampled_data
                self.wavelengths = new_wavelengths

            # 打印波长信息
            if hasattr(self, 'wavelengths'):
                print(f"波长数据: 范围={self.wavelengths[0]}-{self.wavelengths[-1]}, 点数={len(self.wavelengths)}")

                # 验证波长点数是否足够满足序列长度要求
                total_required_len = self.seq_len + self.pred_len
                if len(self.wavelengths) < total_required_len:
                    print(f"警告: 可用波长点数({len(self.wavelengths)})小于所需序列总长度({total_required_len})")
                    # 调整序列长度避免越界
                    avail_len = len(self.wavelengths)
                    self.seq_len = int(avail_len * 2 / 3)  # 分配2/3给输入序列
                    self.pred_len = avail_len - self.seq_len  # 剩余给预测序列
                    print(f"自动调整: seq_len={self.seq_len}, pred_len={self.pred_len}")

            print(f"数据加载完成: 样本形状={self.data_x.shape}")

        except Exception as e:
            print(f"数据加载出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 创建空的替代数据以避免程序崩溃
            self.data_x = np.zeros((10, self.seq_len + self.pred_len))
            self.data_y = np.zeros((10, self.seq_len + self.pred_len))
            self.wavelengths = np.arange(self.seq_len + self.pred_len)

    def __getitem__(self, index):
        """
        获取单个样本的输入和目标序列
        index: 样本索引
        """
        # 直接使用样本索引
        sample_idx = index

        # 获取可用的波长数据长度
        if hasattr(self, 'wavelengths'):
            available_len = len(self.wavelengths)
        else:
            available_len = self.data_x.shape[1]

        # 确保序列长度不超过可用数据长度
        seq_len = min(self.seq_len, available_len)
        pred_len = min(self.pred_len, available_len - seq_len)

        # 固定起始位置为0，即从光谱的起始波长开始
        start_idx = 0

        # 计算结束位置
        s_end = start_idx + seq_len
        r_begin = s_end - self.label_len  # 重叠部分开始位置
        r_end = s_end + pred_len  # 预测部分结束位置

        # 创建固定大小的数组
        seq_x = np.zeros((self.seq_len, 1))
        seq_y = np.zeros((self.label_len + self.pred_len, 1))

        # 填充输入序列数据
        valid_x_len = s_end - start_idx
        if valid_x_len > 0 and valid_x_len <= self.data_x.shape[1]:
            seq_x[:valid_x_len, 0] = self.data_x[sample_idx, start_idx:s_end]

        # 填充目标序列数据
        valid_y_len = r_end - r_begin
        if valid_y_len > 0 and r_begin < self.data_x.shape[1] and r_end <= self.data_x.shape[1]:
            seq_y[:valid_y_len, 0] = self.data_y[sample_idx, r_begin:r_end]

        # 处理波长数据作为标记
        if hasattr(self, 'wavelengths'):
            wavelength_x = np.zeros(self.seq_len)
            wavelength_y = np.zeros(self.label_len + self.pred_len)

            # 计算要使用的波长数据长度
            wl_x_len = min(valid_x_len, len(self.wavelengths) - start_idx)
            if wl_x_len > 0:
                wavelength_x[:wl_x_len] = self.wavelengths[start_idx:start_idx + wl_x_len]

            # 计算目标序列要使用的波长数据长度
            if r_begin < len(self.wavelengths):
                wl_y_len = min(valid_y_len, len(self.wavelengths) - r_begin)
                if wl_y_len > 0:
                    wavelength_y[:wl_y_len] = self.wavelengths[r_begin:r_begin + wl_y_len]

            return seq_x, seq_y, wavelength_x.reshape(-1, 1), wavelength_y.reshape(-1, 1)
        else:
            # 如果没有波长信息，使用零填充的标记
            dummy_mark_x = np.zeros((self.seq_len, 1))
            dummy_mark_y = np.zeros((self.label_len + self.pred_len, 1))
            return seq_x, seq_y, dummy_mark_x, dummy_mark_y

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.data_x)

    def inverse_transform(self, data):
        """直接返回原始数据"""
        return data

class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask

