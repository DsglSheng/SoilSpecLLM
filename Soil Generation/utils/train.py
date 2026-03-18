import torch
import torch.nn.functional as F
import time
import os
import numpy as np

def train_epoch_channels(dataloader, 
                         unet, 
                         diffused_model, 
                         condition, 
                         optimizer, 
                         scheduler,
                         device, 
                         number_of_repetition=1):
    loss_list = []
    unet.train()
    for _ in range(number_of_repetition):
        for data, label in dataloader:
            # (1536, B)

            text_embed = label['text_embed']
            text_embed = np.array(text_embed)
            text_embed = np.squeeze(text_embed)
            # text_embed = text_embed.transpose(1, 0)
            # print('text_embed:',text_embed.shape)
            # :text_embed: (2048, 1536)
            # print('text_embed:',text_embed)
            text_embed = np.repeat(text_embed[:, np.newaxis, :], 1, axis=1)
            # print(text_embed.shape)
            text_embed = torch.Tensor(text_embed)
            # print('text_embed:', text_embed.shape)
            # :text_embed: torch.Size([2048, 1, 1536])
            text_embed = text_embed.to(device)

            latent = data.to(device)

            # t = torch.randperm(diffused_model.config.num_train_timesteps-2)[:latent.shape[0]] + 1 
            # compatible with larger batch size
            t = torch.randint(1, diffused_model.config.num_train_timesteps - 1, (latent.shape[0],))

            noise = torch.randn(latent.shape, device=latent.device)
            xt = diffused_model.add_noise(latent, noise, t)

            xt = xt.to(device)
            t = t.to(device)
            noise = noise.to(device)

            if condition:
                # gender = []
                # age = label['age']
                # hr = label['hr']
                # condition_dict = {}
            
                # for ch in label['gender']:
                #     if ch == 'M':
                #         gender.append(1)
                #     else:
                #         gender.append(0)
                # gender = np.array(gender)
                # gender = np.repeat(gender[:, np.newaxis], 1, axis=1)
                # gender = np.repeat(gender[:, :, np.newaxis], 1, axis=2)
                # gender = torch.Tensor(gender)
                # gender = gender.to(device)
                # condition_dict.update({'gender': gender})
                # 从这里开始-------------------------------
                # Clay = label['Clay']
                # Sand = label['Sand']
                # Silt = label['Sand']
                # condition_dict = {}
                # 定义需要处理的变量列表
                # variables = ['Clay', 'Sand', 'Silt']
                # variables = ['pH(CaCl2)', 'pH(H2O)', 'EC', 'OC', 'CaCO3', 'P','N','K']
                variables = ['pH(CaCl2)', 'pH(H2O)', 'OC', 'CaCO3', 'P', 'N', 'K']
                # variables = ['pH(CaCl2)', 'pH(H2O)']
                # variables = ['Clay', 'Sand', 'Silt','pH(CaCl2)', 'pH(H2O)', 'OC', 'CaCO3', 'P', 'N', 'K']
                # Processed Data, Coarse, Clay, Sand, Silt, pH(CaCl2), pH(H2O), OC, CaCO3, P, N, K

                # 初始化条件字典
                condition_dict = {}

                # 遍历每个变量进行处理
                for var in variables:
                    # 从label中获取对应的数据并转换为numpy数组
                    data = np.array(label[var])
                    # 对数据进行维度扩展和重复操作
                    data = np.repeat(data[:, np.newaxis], 1, axis=1)
                    data = np.repeat(data[:, :, np.newaxis], 1, axis=2)
                    # 将numpy数组转换为torch张量并移动到指定设备
                    data = torch.Tensor(data).to(device)
                    # 将处理后的数据添加到条件字典中
                    condition_dict[var] = data
                # （可选）如果需要，再次确认所有张量都在指定设备上

                # Clay = np.array(Clay)
                # Clay = np.repeat(Clay[:, np.newaxis], 1, axis=1)
                # Clay = np.repeat(Clay[:, :, np.newaxis], 1, axis=2)
                # Clay = torch.Tensor(Clay)
                # Clay = Clay.to(device)
                # condition_dict.update({'Clay': Clay})
                #
                # Sand = np.array(Sand)
                # Sand = np.repeat(Sand[:, np.newaxis], 1, axis=1)
                # Sand = np.repeat(Sand[:, :, np.newaxis], 1, axis=2)
                # Sand = torch.Tensor(Sand)
                # Sand = Sand.to(device)
                # condition_dict.update({'Sand': Sand})
                #
                # Silt = np.array(Silt)
                # Silt = np.repeat(Silt[:, np.newaxis], 1, axis=1)
                # Silt = np.repeat(Silt[:, :, np.newaxis], 1, axis=2)
                # Silt = torch.Tensor(Silt)
                # Silt = Silt.to(device)
                # condition_dict.update({'Silt': Silt})

                for key in condition_dict:
                    condition_dict[key] = condition_dict[key].to(device)

                noise_estim = unet(xt, t, text_embed, condition_dict)
            else: 
                noise_estim = unet(xt, t, text_embed)

            # Batchwise MSE loss 
            loss = F.mse_loss(noise_estim, noise, reduction='sum').div(noise.size(0))
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return sum(loss_list) / len(loss_list)


def train_model(meta, 
                save_weights_path, 
                dataloader,  
                diffused_model, 
                unet, 
                h_, 
                logger):
 
    device = torch.device(meta['device'] if torch.cuda.is_available() else "cpu")
    unet = unet.to(device)
    optimizer = torch.optim.AdamW(params=unet.parameters(), lr=h_['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=h_['epochs']*len(dataloader), eta_min=0.1*h_['lr'])

    min_loss = 80
    start_time = time.time()
    for i in range(1, h_['epochs'] + 1):
        s_t = time.time()
        mean_loss = train_epoch_channels(dataloader=dataloader, 
                                         unet=unet, 
                                         diffused_model=diffused_model, 
                                         optimizer=optimizer, 
                                         scheduler=scheduler,
                                         device=device, 
                                         condition=meta['condition'], 
                                         number_of_repetition=1)
        logger.info(f'Epoch: {i}, mean loss: {mean_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}')
        if (mean_loss < min_loss):
            min_loss = mean_loss
            torch.save(unet.state_dict(), os.path.join(save_weights_path, 'unet_best.pth'))
            logger.info(f'epoch {i} unet_best.pth has been saved.')
        if (i % 50 == 0):
            torch.save(unet.state_dict(), os.path.join(save_weights_path, f'unet_{i}.pth'))

        e_t = time.time()
        logger.info(f"Epoch Time Used: {e_t - s_t}s; Total Time Used: {e_t - start_time}s")
