from flowvae.ml_operator.operator import BasicCondAEOperator, load_model_from_checkpoint
from flowvae.ml_operator.kfold import K_fold, K_fold_evaluate
from flowvae.dataset import MCFlowDataset
from flowvae.ml_operator.config import ModelConfig
import numpy as np
import os, sys, copy
from postprocess import cal_loss

device = "cuda:0"
folder = '/mnt/ssdraid/yunjia/crm/'
dataset_prefix = ''
file_name = 'AeroTransformer_L'
tran_file_name = 'temp'

origingeom = np.load(os.path.join(folder, f'{dataset_prefix}origingeom.npy'))

fldata = MCFlowDataset([f'{dataset_prefix}data', f'{dataset_prefix}geom', f'{dataset_prefix}index'], is_ref=False, d_c=2, 
                       split_paras={'method': 'base'}, aux_channel_take=[2,3], data_base=folder, marker_idx=2)

is_from_scratch = False

batch_size = 64
num_epochs = 1600
init_lr    = 1e-4
trans_epochs = 800
n_flow_per_shape = 1  # max shape amount of flow fields used in training, None means all flow fields of each shape are used


def func_train(irun, itrain, itest):

    model = ModelConfig(config_path=os.path.join('save', file_name, "model_config")).create()
    load_model_from_checkpoint(model, epoch=-1, folder=os.path.join('save', file_name), device=device)
    print("network have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    split_dataset = {
        'train': fldata.subset(itrain, n_flow_per_shape=n_flow_per_shape, rng=irun),
        'val': fldata.subset(itest)
    }
    op = BasicCondAEOperator(file_name, model, fldata=fldata, shuffle=True, 
                             batch_size=batch_size, num_epochs=trans_epochs,
                             split_dataset=split_dataset, device=device)
    
    op.set_optimizer(optimizer_name='Adam', ema_optimizer=1.0 if is_from_scratch else 0.8, lr=init_lr)
    op.set_scheduler('OneCycleLR', max_lr=init_lr, total_steps=(int(len(split_dataset['train'])) // batch_size + 1)*trans_epochs, div_factor=25 if is_from_scratch else 200)
    if not is_from_scratch: op.load_checkpoint(num_epochs-1, load_opt=False)
    op.set_transfer_model(tran_file_name + str(irun), reset_param=False, grad_require_layers=['qkv'], is_lora=False, lora_params={'r':16, 'alpha': '2r', 'is_lora_k': False})#, grad_require_parents=['decoder', 'embedder'])
    op.train_model(save_check=trans_epochs, v_tqdm=False, update_lr_batch=True)    

    return op.model

def func_eval(irun, model, itrain, itest):

    errors = []
    cal_loss(model, errors, fldata, origingeom, ref=0, device=device, output_type=None, is_normal=True)

    absarrors = []
    itrains = []
    itests = []
    ii = 0
    for i_f in range(len(errors[0])):
        for i_c in range(len(errors[0][i_f])):
            absarrors.append(errors[0][i_f][i_c])
            if i_f in itrain:
                itrains.append(ii)
            elif i_f in itest:
                itests.append(ii)
            ii += 1
    absarrors = np.array(absarrors)
    absarrors[:, 3:6] = np.abs(absarrors[:, 6:9] - absarrors[:, 3:6])
    errors_stats = np.vstack((np.mean(np.take(absarrors[:, 0:6], itrains, axis=0), axis=0), np.mean(np.take(absarrors[:, 0:6], itests, axis=0), axis=0)))

    return errors[0], errors_stats

# 
if __name__ == '__main__':

    history_file = os.path.join('save', file_name + '_' + tran_file_name + '_history')
    history = K_fold(fldata.n_shape, func_train, func_eval, k=10, krun=10, num_train=-1, history_file_name=history_file + '_nn')
