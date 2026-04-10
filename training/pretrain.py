'''
Pretrain the model on the data

'''


from flowvae.ml_operator import ModelOperator, BasicCondAEOperator
from flowvae.ml_operator.config import ModelConfig
from flowvae.dataset import MCFlowDataset
from flowvae.utils import warmup_lr
import matplotlib.pyplot as plt
from torch import nn
import torch

import numpy as np
import os, sys
from flowvae.app.wing import models

if __name__ == '__main__':

    device = "cuda:0"
    folder = '/mnt/ssdraid/yunjia/superwing/'
    portion = 100
    dataset_prefix = ''
    surface_output = True
    aero_lossterm = False

    if surface_output:
        fldata = MCFlowDataset([f'{dataset_prefix}data', f'{dataset_prefix}geom', f'{dataset_prefix}index'], is_ref=False, d_c=2, 
                                split_paras={'indexFileName': folder + 'training_samples_index.txt'}, 
                                aux_channel_take=[2,3], data_base=folder, marker_idx=2)
    else:
        fldata = MCFlowDataset([f'{dataset_prefix}index', f'{dataset_prefix}2ngeom', f'{dataset_prefix}index'], is_ref=False, d_c=2, 
                                split_paras={'indexFileName': 'training_samples_index.txt'}, 
                                aux_channel_take=[2,3], output_channel_take=[-3, -2, -1],data_base=folder, marker_idx=2)
    
    
    batch_size = 64 
    num_epochs = int(25 * batch_size * (100/portion))
    init_lr    = 1e-3
    split_train_ratio = 0.90
    
    if aero_lossterm:
        origingeom = np.load(os.path.join(folder, f'{dataset_prefix}origingeom.npy'))
        fldata.change_to_force(original_geom=origingeom, is_nondim=False, use_save=True)
    
    for run in range(3):
        
        #* U-Net
        # model_args = dict(h_e=[8, 16, 16, 32, 32, 64], h_e1=None, h_e2=None, h_d=[66, 64, 32, 32, 16, 16, 8], h_in=3, h_out=3, de_type='cat', coder_type ='onlycond', coder_kernel=3, nt=128, decoder_layer_sizes=[8, 16, 32, 64, 128, 256], last_size=4, nn_out=256)
        # model_config = ModelConfig(model_name='ounetbedmodel', init_kwargs=model_args)

        #* Transolver
        # model_args = dict(h_in=3, slice_num=64, mlp_ratio=4, n_hidden=512, n_layers=5, is_flatten=True)
        # model_config = ModelConfig(model_name='WingTransolver', init_kwargs=model_args)

        #* ViT
        # model_args = dict(image_size=(128, 256), patch_size=(4, 4),fun_dim=3, out_dim=3, mlp_ratio=4, n_layers=5, pos_embedding='trainable', n_hidden=128)
        # model_config = ModelConfig(model_name='WingViT', init_kwargs=model_args)

        #* AeroTransformer
        # (surface output)
        model_args = dict(patch_size=(4, 4), window_size=(8, 8), fun_dim=3, out_dim=3, mlp_ratio=4, n_layers=5, n_hidden=32, type_cond='inj')
        # (coefficient output)
        # model_args = dict(patch_size=(4,4), fun_dim=3, out_dim=3, mlp_ratio=4, n_layers=5, depth=[2,3,5,3,2], n_hidden=16, device=device, type_cond='inj', output_type='attn_pool')
        model_config = ModelConfig(model_name='AeroTransformer', init_kwargs=model_args)

        op = BasicCondAEOperator('temp' + str(run), model_config, fldata, shuffle=True, 
                                                        batch_size=batch_size, 
                                                        num_epochs=num_epochs,
                                                        # restart=-1,
                                                        split_train_ratio=split_train_ratio,
                                                        device=device)


        op.set_optimizer(optimizer_name='Adam', ema_optimizer=1.0, lr=init_lr)
        op.set_scheduler(scheduler_name='OneCycleLR', max_lr=init_lr, total_steps=(int(len(fldata)*split_train_ratio) // batch_size + 1)*num_epochs)
        if aero_lossterm:
            op.set_lossparas(aero_weight=0.1, aero_epoch=0., conFIG=True)
        op.train_model(save_check=num_epochs, v_tqdm=False, update_lr_batch=True)    

        print('=============================================')
        print('Run %d   Over' % run)

