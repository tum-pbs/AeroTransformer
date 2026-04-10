from flowvae.ml_operator import load_model_from_checkpoint
from flowvae.ml_operator.config import ModelConfig
from flowvae.dataset import MCFlowDataset
from cfdpost.wing.basic import BasicWing

import os
import numpy as np
import torch

device = "cuda:0"

def cal_loss(vae_model, errors: list, fldata: MCFlowDataset, origingeom, device, bias=0, ref=-1, ref_channels=(None, None), recon_channels=(None, None), input_channels=(None, None), output_type=None, is_normal=True, field_error='L1'):
    errs = []
    
    for i_f in range(fldata.n_shape):
        geom_infos = {}
        geom_infos['ref_area'] = fldata.get_index_info(i_f, 0, 4)
        
        sample_data = fldata.get_series(i_f)
        inputs = torch.from_numpy(sample_data['ref']).float().to(device)
        real_field = sample_data['flowfields']
        # print(i_f in fldata.data_idx)

        if ref > -1:
            auxs   = torch.from_numpy(sample_data['condis']).float().to(device)
            output = vae_model(inputs[:, input_channels[0]: input_channels[1]], code=auxs)[0].cpu().detach().numpy()
            if ref > 0:
                output[ref_channels[0]: ref_channels[1]] += inputs[:, recon_channels[0]: recon_channels[1]].cpu().detach().numpy()
        else:
            output = vae_model(inputs).cpu().detach().numpy()
        # print(output.shape, bias.shape)
        if isinstance(bias, torch.Tensor):
            output = output + bias.cpu().detach().numpy()
        else:
            output = output + bias
        
        err = []
        
        if output_type is None:
            for ic in range(inputs.shape[0]):
                wg1 = BasicWing(paras=geom_infos, aoa=auxs[ic, 0].item(), iscentric=True)
                wg1.read_formatted_surface(geometry=origingeom[i_f], data=real_field[ic], isinitg=False, isnormed=is_normal)
                wg1.aero_force()
                cl_real = wg1.coefficients

                wg2 = BasicWing(paras=geom_infos, aoa=auxs[ic, 0].item(), iscentric=True)
                wg2.read_formatted_surface(geometry=origingeom[i_f], data=output[ic], isinitg=False, isnormed=is_normal)
                wg2.aero_force()
                cl_recon = wg2.coefficients

                if np.abs(cl_real - cl_recon)[0] > 0.005:
                    print(i_f, np.abs(cl_recon - cl_real), cl_real[1], fldata.get_index_info(i_f, ic, 10))

                if field_error == 'L2':
                    err.append(np.concatenate((np.mean((output[ic] - real_field[ic])**2, axis=(1,2)), cl_real, cl_recon)))
                else:
                    err.append(np.concatenate((np.mean(np.abs(output[ic] - real_field[ic]), axis=(1,2)), cl_real, cl_recon)))
        else:
            rngs = np.array([0.82500667, 0.04398663, 0.80370395])
            mins = np.array([-0.02917635,  0.01158103, -0.00831657])
            err = np.concatenate((np.zeros_like(output), real_field[:, [-3, -2, -1]], output * rngs + mins), axis=1)

        errs.append(err)
    errors.append(errs)

if __name__ == '__main__':
    
    folder = '/mnt/ssdraid/yunjia/superwing/'
    dataset_prefix = ''
    surface_output = True
    output_type = None if surface_output else 'attn_pool'
    
    if surface_output:
        fldata = MCFlowDataset([f'{dataset_prefix}data', f'{dataset_prefix}geom', f'{dataset_prefix}index'], is_ref=False, d_c=2, 
                                split_paras={'indexFileName': folder + 'training_samples_index.txt'}, 
                                aux_channel_take=[2,3], data_base=folder, marker_idx=2)
    else:
        fldata = MCFlowDataset([f'{dataset_prefix}index', f'{dataset_prefix}2ngeom', f'{dataset_prefix}index'], is_ref=False, d_c=2, 
                                split_paras={'indexFileName': 'training_samples_index.txt'}, 
                                aux_channel_take=[2,3], output_channel_take=[-3, -2, -1],data_base=folder, marker_idx=2)
    
    origingeom = np.load(os.path.join(folder, f'{dataset_prefix}origingeom.npy'))

    run = 'temp'
    n_runs = 3
    expand_errors = True  # True: load existing *_error_fd and append from next Run index
    error_fd_path = os.path.join('save', '%s_error_fd' % (run))

    if expand_errors and os.path.exists(error_fd_path):
        errors = torch.load(error_fd_path, weights_only=False)
        if not isinstance(errors, list):
            errors = list(errors)
        print(f'Expand mode on: loaded {len(errors)} existing results from {error_fd_path}')
    else:
        errors = []

    start_subrun = len(errors) if expand_errors else 0
    for subrun in range(start_subrun, n_runs):
        file_name = '%s_Run%d' % (run, subrun)        
        model = ModelConfig(config_path=os.path.join('save', file_name, "model_config")).create()
        load_model_from_checkpoint(model, epoch=-1, folder=os.path.join('save', file_name), device=device)

        cal_loss(model, errors, fldata, origingeom, ref=0, device=device, output_type=output_type, is_normal=True)
        torch.save(errors, error_fd_path)
    
    for irun in range(len(errors)):
        err_stat = [[], []]
        ii = 0
        for i_f in range(len(errors[irun])):
            for i_c in range(len(errors[irun][i_f])):
                istrain = ii in fldata.data_idxs
                err_stat[istrain].append(errors[irun][i_f][i_c])
                ii += 1

        err_stat_train = np.array(err_stat[1])
        err_stat_test = np.array(err_stat[0])
        print(err_stat_train.shape)

        print(*np.mean(err_stat_train[:, :3], axis=0), *np.mean(abs(err_stat_train[:, 3:6] - err_stat_train[:, 6:9]), axis=0))
        print(*np.mean(err_stat_test[:, :3], axis=0), *np.mean(abs(err_stat_test[:, 3:6] - err_stat_test[:, 6:9]), axis=0))
    