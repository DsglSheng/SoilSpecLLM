import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from scipy import signal
import ast

class CustomVAEDataset(Dataset):
    def __init__(self, csv_file, num_folds=5, test_fold=None, split='train', resample_length=None):
        # 璇诲彇 CSV 鏂囦欢
        df = pd.read_csv(csv_file)

        # 瑙ｆ瀽 'Processed Data' 鍒楋細灏嗘瘡琛岀殑绌烘牸鍒嗛殧瀛楃涓茶浆涓洪暱搴?100鐨刦loat32鏁扮粍
        sequences = []
        for data_str in df['Processed Data']:
            # 灏嗗瓧绗︿覆鎷嗗垎涓烘诞鐐规暟鍒楄〃骞惰浆鎹负 float32 鏁扮粍
            seq = np.array(data_str.split(), dtype=np.float32)
            # 纭繚搴忓垪闀垮害涓?100锛屽涓嶈冻鍒欏～鍏?锛岃秴鍑哄垯鎴柇
            if len(seq) != 2100:
                if len(seq) < 2100:
                    seq = np.pad(seq, (0, 2100 - len(seq)), mode='constant', constant_values=0)
                else:
                    # print("姝ｅ湪杩涜鎴柇")
                    seq = seq[-2100:].astype(np.float32)
            sequences.append(seq)
        # 灏嗗垪琛ㄨ浆鎹负numpy鏁扮粍 (褰㈢姸: [鏍锋湰鏁? 2100])
        sequences = np.stack(sequences)

        # 濡傛灉鎸囧畾浜嗛噸閲囨牱闀垮害锛屽垯瀵规瘡鏉″簭鍒楄繘琛岄噸閲囨牱鑷虫柊闀垮害
        if resample_length is not None:
            resampled_sequences = []
            for seq in sequences:
                # 浣跨敤 SciPy 鐨勪俊鍙烽噸閲囨牱鍑芥暟璋冩暣搴忓垪闀垮害
                new_seq = signal.resample(seq, resample_length)
                resampled_sequences.append(new_seq.astype(np.float32))
            sequences = np.stack(resampled_sequences)

        # 瀹氫箟闇€瑕佹彁鍙栫殑鏍囩鍒?
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
        #     # 灏嗗瓧绗︿覆鎷嗗垎涓烘诞鐐规暟鍒楄〃骞惰浆鎹负 float32 鏁扮粍
        #     col = np.array(label_cols.split(), dtype=np.float)
        #     label_cols.append(col)
        # labels_matrix = np.stack(label_cols)

        # Processed Data, Coarse, Clay, Sand, Silt, pH(CaCl2), pH(H2O), OC, CaCO3, P, N, K
        # 纭繚 DataFrame 鍖呭惈鎵€鏈夋爣绛惧垪锛堣嫢缂哄け鍒欏～鍏?NaN锛?
        for col in self.label_cols:
            if col not in df.columns:
                df[col] = np.nan
        # 4. 澶勭悊鍖呭惈 Tensor 鏁版嵁鐨勫垪锛屽皢瀛楃涓茶〃绀虹殑 Tensor 杞崲涓?numpy 鏁扮粍

        # def parse_tensor_string(x):
        #
        #     data_str = x.replace("[", "").replace("]", "").replace("...", "").strip()
        #     # 灏嗘暟鍊奸儴鍒嗘媶鍒嗕负鍒楄〃
        #     values = data_str.split()
        #     # 灏嗗垪琛ㄨ浆鎹负娴偣鏁扮被鍨嬬殑 numpy 鏁扮粍
        #     tensor_data = np.array([float(value) for value in values], dtype=np.float32)
        #     print(tensor_data.shape)
        #     return tensor_data
        # for col in self.label_cols:
        #     df[col] = df[col].apply(parse_tensor_string)

        # 鎻愬彇鏍囩鏁版嵁涓?numpy 鐭╅樀锛坒loat32绫诲瀷锛岀己澶卞€间负 NaN锛?
        labels_matrix = df[self.label_cols].to_numpy(dtype=np.float32)
        # labels_matrix = df[self.label_cols]
        # labels_matrix = df[col]
        # 鍑嗗绱㈠紩鍒楄〃锛屾牴鎹?K 鎶樺弬鏁板垝鍒嗚缁?娴嬭瘯闆?
        self.indices = list(range(len(df)))
        if test_fold is not None:
            # 涓烘瘡涓牱鏈寚瀹氭姌鍙?
            if 'Point_ID' in df.columns:
                # 鑻ュ瓨鍦?Point_ID锛屽垯鎸夊敮涓€ Point_ID 鍒嗙粍鍒掑垎鎶樺彿锛岀‘淇濈浉鍚?Point_ID 鐨勬牱鏈湪鍚屼竴鎶?
                unique_ids = df['Point_ID'].unique()
                id_to_fold = {pid: idx % num_folds for idx, pid in enumerate(unique_ids)}
                sample_folds = [id_to_fold[pid] for pid in df['Point_ID']]
            else:
                # 鏃?Point_ID 鏃讹紝鐩存帴鎸夐『搴忓绱㈠紩鍙栨ā鎸囨淳鎶樺彿
                sample_folds = [i % num_folds for i in range(len(df))]
            # 鏍规嵁 split 鍙傛暟杩囨护鍑鸿缁冩垨娴嬭瘯闆嗙殑绱㈠紩
            if split == 'train':
                self.indices = [i for i, fold in enumerate(sample_folds) if fold != test_fold]
            elif split == 'test':
                self.indices = [i for i, fold in enumerate(sample_folds) if fold == test_fold]
            else:
                raise ValueError("split 鍙傛暟蹇呴』鏄?'train' 鎴?'test'")

        # 淇濆瓨鏈€缁堢殑搴忓垪鏁版嵁鍜屾爣绛剧煩闃?
        self.data = sequences  # numpy 鏁扮粍锛屽舰鐘? [N, 搴忓垪闀垮害]
        self.labels_matrix = labels_matrix  # numpy 鏁扮粍锛屽舰鐘? [N, 13]

    def __len__(self):
        # 杩斿洖褰撳墠鏁版嵁闆嗗寘鍚殑鏍锋湰鏁伴噺锛堟牴鎹垝鍒嗗悗鐨?indices 鍒楄〃锛?
        return len(self.indices)

    def __getitem__(self, idx):
        # 鑾峰彇瀹為檯鏍锋湰绱㈠紩
        actual_idx = self.indices[idx]
        # 鎻愬彇鍏夎氨搴忓垪鏁版嵁
        x = self.data[actual_idx]
        # 鎻愬彇瀵瑰簲鐨勬爣绛惧€煎苟鏋勫缓瀛楀吀

        label_values = self.labels_matrix[actual_idx]
        label_dict = {key: label_values[i] for i, key in enumerate(self.label_cols)}
        # label_dict = {key: label_values for i, key in enumerate(self.label_cols)}
        # 杩斿洖鍏夎氨搴忓垪 x 浠ュ強鏍囩瀛楀吀 label_dict
        return x, label_dict
if __name__ == '__main__':
    # Original dataset
    # path = '/data/0shared/MIMIC/physionet.org/files/mimic-iv-ecg/1.0/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
    # data = soil_gen_Dataset(dataset_path=path, resample_length=1024, demo_label=True)

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
