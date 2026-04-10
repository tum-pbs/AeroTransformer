'''
Yang Yunjia @ Mar. 27 2025

This file is modified from 'gen-mesh-new2.py'
correct the error for root kink adjustment

This is the final version on the cluster which is named 'gen-mesh-new2.py' there.

'''


import os, sys, random, time
import numpy as np
import json
from cst_modeling.section import cst_foil
from cst_modeling.basic import rotation_3d
from cst_modeling.io import read_plot3d, output_plot3d_concat
from scipy.interpolate import CubicSpline
from functools import reduce
from pyhyp import pyHyp
from cgnsutilities.cgnsutilities import readGrid

from mpi4py import MPI

comm = MPI.COMM_WORLD

mesh_point_numbers = {
    'nFoil':    117,
    'nX':       113,
    'nY':       [41, 125],
    # 'nY':       [81, 85],
    'nZ':       9,
    'nFar':     81
}

tip_path = '../../original_tip.xyz'

xx0 = [0., 5.05256963e-04, 1.29435990e-03, 2.30710683e-03, 3.47136074e-03, 4.61166766e-03, 5.96008856e-03\
, 7.55045344e-03, 9.42059830e-03, 1.16133031e-02, 1.41777060e-02, 1.71692487e-02, 2.06514955e-02, 2.46963502e-02, 2.93869138e-02\
, 3.48180094e-02, 4.10985870e-02, 4.83540454e-02, 5.67280608e-02, 6.63881031e-02, 7.69451593e-02, 8.75207785e-02, 9.81108987e-02\
, 1.08712986e-01, 1.19325489e-01, 1.29947375e-01, 1.40577934e-01, 1.51216707e-01, 1.61863310e-01, 1.72517384e-01, 1.83178632e-01\
, 1.93846825e-01, 2.04521670e-01, 2.15202919e-01, 2.25890219e-01, 2.36583338e-01, 2.47281970e-01, 2.57985787e-01, 2.68694449e-01\
, 2.79407717e-01, 2.90125279e-01, 3.00846740e-01, 3.11571696e-01, 3.22299795e-01, 3.33030723e-01, 3.43764039e-01, 3.54499277e-01\
, 3.65235991e-01, 3.75973806e-01, 3.86712274e-01, 3.97450937e-01, 4.08189379e-01, 4.18927144e-01, 4.29663868e-01, 4.40399177e-01\
, 4.51132661e-01, 4.61863901e-01, 4.72592596e-01, 4.83318512e-01, 4.94041363e-01, 5.04760823e-01, 5.15383453e-01, 5.26002270e-01\
, 5.36617182e-01, 5.47228029e-01, 5.57834613e-01, 5.68436802e-01, 5.79034560e-01, 5.89627936e-01, 6.00216864e-01, 6.10801313e-01\
, 6.21381327e-01, 6.31956985e-01, 6.42528420e-01, 6.53095762e-01, 6.63659148e-01, 6.74218723e-01, 6.84774729e-01, 6.95327470e-01\
, 7.05877214e-01, 7.16424300e-01, 7.26969057e-01, 7.37511925e-01, 7.48053471e-01, 7.58594170e-01, 7.69134556e-01, 7.79675279e-01\
, 7.90217107e-01, 8.00760711e-01, 8.11306563e-01, 8.21855670e-01, 8.32408873e-01, 8.42966811e-01, 8.53529992e-01, 8.64099098e-01\
, 8.74674556e-01, 8.85256582e-01, 8.95845260e-01, 9.06440753e-01, 9.17042790e-01, 9.27651068e-01, 9.38265455e-01, 9.48885481e-01\
, 9.59569457e-01, 9.68120400e-01, 9.74963459e-01, 9.80439342e-01, 9.84820908e-01, 9.88326688e-01, 9.91131624e-01, 9.93375756e-01\
, 9.95171174e-01, 9.96607568e-01, 9.97756716e-01, 9.98676054e-01, 9.99411534e-01, 1.]

def redirect_stdout(outFile):
    def decorator(func):
        def wrapper(*args, **kwargs):
            original_stdout_fd = os.dup(sys.stdout.fileno())
            # original_stdout = sys.stdout
            with open(outFile, 'w') as fstdout:

                os.dup2(fstdout.fileno(), sys.stdout.fileno())
                try:
                    result = func(*args, **kwargs)
                finally:
                    os.dup2(original_stdout_fd, sys.stdout.fileno())
                    os.close(original_stdout_fd)
                
            return result
            
        return wrapper
    return decorator

def linear_smooth(block1, block2):
    
    nsm = 3
    # print(block1.shape, block2.shape)
    new_block = np.linspace(block1[:, -nsm], block2[:, nsm-1], 2*nsm-1, axis=1)
    block1[:, -nsm:] = new_block[:, :nsm]
    block2[:, :nsm] = new_block[:, -nsm:]

def scaling_tip(xx1, zz1, wingCoef, rdx=None, rdz=None):
    '''
    Linear interpolate for new tip airfoil
    
    
    '''
    blocks = read_plot3d(tip_path)
    # tip_lower_new, tip_front_new, tip_upper_new, tip_back_new, tip_side
    
    nn0 = len(blocks[0]) + int(len(blocks[1]) / 2)
    nn1 = len(blocks[0])
    nn2 = len(blocks[1])
    
    assert mesh_point_numbers['nFoil'] == nn0, 'Foil direction %d != %d' % (mesh_point_numbers['nFoil'], nn0) 
    assert mesh_point_numbers['nX'] == nn1, 'X direction %d != %d' % (mesh_point_numbers['nX'], nn1)
    assert mesh_point_numbers['nZ'] == nn2, 'Y direction %d != %d' % (mesh_point_numbers['nZ'], nn2)
    
    # target airfoil
    # xu1, yu1, yl1, _, _ = cst_foil(nn=nn0, cst_u=cst_u_tip, cst_l=cst_l_tip, t=t_tip, a1=0.985)
    # xx1 = np.concatenate((np.flip(xu1[1:]), xu1, np.ones((nn2-1,))), axis=0)
    yy1 = np.zeros_like(xx1)
    # zz1 = np.concatenate((np.flip(yl1[1:]), yu1, np.linspace(yu1[-1], yl1[-1], nn2)[1:]), axis=0)
    
    # modify endwall block position based on sweep and dihedral angle 
    tail_avg = np.mean(blocks[4][:, -1, 0], axis=0)
    tip_tail_avg = np.mean(blocks[3][:, 0, 0], axis=0)
    # print(tail_avg, tip_tail_avg)
    dx0, dy, dz0 = tail_avg - tip_tail_avg
    if rdx is None:
        dx = (np.tan(min(35,0.8*wingCoef['SA'])/180*np.pi) + (wingCoef['TR'] - 1) / wingCoef['half_span']) * dy - (tail_avg[0] - 1)
    else:
        dx = rdx * dy * 0.5

    dz = -np.tan(wingCoef['DA']/180*np.pi) * dy * 0.5

    print(np.arctan([rdx, rdz])/np.pi*180)
    blocks[4][:, :, :, 0] += (dx - dx0)
    blocks[4][:, :, :, 2] += (dz - dz0)
    
    # modify endwall block thickness
    low_r = min(zz1) / min(blocks[0][:, 0, 0, 2])
    upp_r = max(zz1) / max(blocks[2][:, 0, 0, 2])
    blocks[4][:, :, :, 2] = np.linspace(blocks[4][0, :, :, 2]*low_r, blocks[4][-1, :, :, 2]*upp_r, blocks[4].shape[0])
    
    ## modify circum blocks to blend
    
    surface_blocks1 = np.concatenate((blocks[0][:-1], blocks[1][:-1], blocks[2][:-1], blocks[3]), axis=0)   # circum blocks of tip
    tip_airfoils1 = surface_blocks1[:, 0, 0]   # airfoil at wing side
    tip_airfoils2 = surface_blocks1[:, -1, 0]   # airfoil at tip side
    side_block_airfoil = np.concatenate((np.flip(blocks[4][0, 1:, 0], axis=0), blocks[4][:-1, 0, 0], blocks[4][-1, :-1, 0], np.flip(blocks[4][:, -1, 0], axis=0)), axis=0)  # endwall block of tip

    diff1 = np.array([xx1, yy1, zz1]).transpose(1, 0) - tip_airfoils1
    diff2 = side_block_airfoil - tip_airfoils2
    
    for block, st, ed in zip(blocks[:-1], [None, nn1-1, nn1+nn2-2, -nn2], [nn1, nn1+nn2-1, -nn2+1, None]):
        block[:, :, 0, :] += (np.linspace(1, 0, block.shape[1])[None, :, None] * diff1[st:ed, None, :] +\
            np.linspace(0, 1, block.shape[1])[None, :, None] * diff2[st:ed, None, :])

    return blocks

def move_block(block, translate, scale, angle, axis):
    
    block_shape = block.shape
    new_block = rotation_3d(block.reshape(-1, 3), origin=np.array([0, 0, 0]), axis=axis, angle=angle)
    # new_block[:, 1] = 0.5 * new_block[:, 1]
    new_block = new_block * scale + translate[None, :]  
    
    return new_block.reshape(*block_shape)

def power_growth0(dx0, L, n):
    '''
    integer ,intent(in ):: n
    real*8  ,intent(in ):: dx0, L
    real*8  ,intent(out):: dxs(n)
    real*8  :: f, a, a1, aa
    integer :: i, is
    '''
    dxs = np.zeros((n,))

    f = L / dx0
    a = 1.2
    a1= 1.0
    
    while abs((a-a1)/a) > 2e-7:
        a1= a
        a =(a*f-f+1.)**(1./n)
    aa = 0.

    for i in range(n):
        dxs[i] = aa+a**i*dx0
        aa = dxs[i]

    return dxs

def power_growth(dx0, dx1, L, n):
    '''

    '''
    k0 = 0
    k = int(n / 2)

    while abs(k0 - k) > 0.5:
        k0 = k
        r = (dx0 / dx1)**(1 / (k-2))
        k = int(n + 1 - (L - dx1 * (1 - r**(k0-1)) / (1 - r)) / dx0)

    yLEs1 = np.concatenate((np.arange(0, n - k) * dx0, np.flip(L - dx1 * (1 - r**np.arange(0, k)) / (1 - r))), axis=0)
    print(yLEs1[n - k - 3], yLEs1[n - k + 2])
    yLEs1[n - k - 3: n - k + 3] = np.linspace(yLEs1[n - k - 3], yLEs1[n - k + 2], 6)    # smoothing
    
    return yLEs1

def surface_meshing(wingCoefs: dict, folder='.'):
    
    #* surface meshing
    
    nn0 = mesh_point_numbers['nFoil']
    nn1 = mesh_point_numbers['nX']
    nn2 = mesh_point_numbers['nZ']
    nny = mesh_point_numbers['nY']
    
    # root baseline airfoil
    xx1, yu1, yl1, _, _ = cst_foil(nn=nn0, x=np.array(xx0), cst_u=wingCoefs['cst_u'], cst_l=wingCoefs['cst_l'], t=wingCoefs['tcroot'], tail=0.008)
    cmb1 = 0.5 * (yu1 + yl1)
    tc1  = yu1 - yl1
    
    # get distributed values
    cabin_ratio = 0.1
    AR = wingCoefs['AR']
    TR = wingCoefs['TR']
    half_span = wingCoefs['half_span'] 
    kink = half_span * wingCoefs['kink']
    cabin = half_span * cabin_ratio
    
    # get spanwise eta distribution
    # inner section
    yLEs0 = np.linspace(cabin, kink, nny[0])
    # outer section (linear) -> 0.0006
    yLEs1 = power_growth(dx0=yLEs0[-1] - yLEs0[-2], dx1=0.002*TR, L=half_span - kink, n=nny[1]) + kink
    yLEs = np.concatenate((yLEs0, yLEs1), axis=0)   # Caution! There is a relunctate point when concatenate, 
                                                    # this is dealed when seperating it to two blocks at line 268
    
    # import matplotlib.pyplot as plt
    # plt.plot(range(nny[0]), yLEs0, '-o', c='r')
    # plt.plot(range(nny[1]), yLEs1, '-o')
    # plt.show()
    
    # leading edge parameters
    tip = half_span + 1e-9
    
    xLEs = np.tan(wingCoefs['SA']/180*np.pi) * yLEs     # sweep direction
    
    zLEs0 = np.tan(wingCoefs['DA_kink']/180*np.pi) * yLEs0  # inner section, linear distribution
    zLE_kink = zLEs0[-1]
    zLE_tip = np.tan(wingCoefs['DA']/180*np.pi) * half_span # zLE at kink
    zLE_cs = CubicSpline([kink, tip], [zLE_kink, zLE_tip], bc_type=((1, np.tan(wingCoefs['DA_kink']/180*np.pi)), 'not-a-knot'), extrapolate=False)
    zLEs1 = zLE_cs(yLEs1)
    zLEs  = np.concatenate((zLEs0, zLEs1), axis=0)
    
    # thickness ratio, chord, camber and twist distribution
    mid_outer = 0.5 * (kink + half_span)
    mid_inner = 0.5 * kink
    trs_cs = CubicSpline([0, kink, mid_outer, tip], [1] + [reduce(lambda x, y: x * y, wingCoefs['rtcs'][:i+1]) for i in range(3)], extrapolate=False)
    trs = trs_cs(yLEs)
    wingCoefs['rtctip'] = trs[-1]
    
    tws_cs = CubicSpline([0, mid_inner, kink, mid_outer, tip], [0] + [reduce(lambda x, y: x + y, wingCoefs['twists'][:i+1]) for i in range(4)], extrapolate=False)
    tws = tws_cs(yLEs)
    
    cmb_cs = CubicSpline([0, mid_inner, kink, mid_outer, tip], [0, wingCoefs['cambers'][0]*wingCoefs['cambers'][1],
                                                                      wingCoefs['cambers'][1], 1, wingCoefs['cambers'][2]], extrapolate=False)
    cmbs = cmb_cs(yLEs)

    chord_tip   = TR
    chord_kink  = TR * wingCoefs['kink'] + 1 * (1 - wingCoefs['kink'])
    root_adj_l  = wingCoefs['rootadj'] * (xLEs[nny[0]] + chord_kink - 1)
    chord_cabin = TR * cabin_ratio + 1 * (1 - cabin_ratio) + root_adj_l * (wingCoefs['kink'] - cabin_ratio) / wingCoefs['kink']
    chord_root = 1 + root_adj_l 
    chords0 = chord_cabin + (chord_kink - chord_cabin) * (yLEs0 - cabin) / (kink - cabin)
    chords1 = chord_kink + (chord_tip  - chord_kink) * (yLEs1 - kink) / (half_span - kink)
    chords  = np.concatenate((chords0, chords1), axis=0)

    wingCoefs['surface_area'] = (chord_cabin + chord_kink) * (kink - cabin) * 0.5 + (chord_kink + chord_tip) * (half_span - kink) * 0.5

    yus = np.outer(cmb1, cmbs) + 0.5 * np.outer(tc1, trs) # nn * ns
    yls = np.outer(cmb1, cmbs) - 0.5 * np.outer(tc1, trs) # nn * ns
    xxs = np.concatenate((np.flip(xx1[1:], axis=0), xx1[:-1], np.ones((nn2,))), axis=0)
    zzs = np.concatenate((np.flip(yls[1:], axis=0), yus[:-1], np.linspace(yus[-1], yls[-1], nn2)), axis=0)
    
    # rotation
    tws_rat = tws / 180 * np.pi
    xxs1  = np.cos(tws_rat[None, :]) * xxs[:, None] +  np.sin(tws_rat[None, :]) * zzs
    zzs1 = -np.sin(tws_rat[None, :]) * xxs[:, None] +  np.cos(tws_rat[None, :]) * zzs

    xxs1 = xxs1 * chords[None, :] + xLEs[None, :]
    zzs1 = zzs1 * chords[None, :] + zLEs[None, :]

    # print(xxs.shape, np.repeat(yLEs[None, :], 2*nn0-2+nn2, axis=0).shape, zzs.shape)
    block = np.stack((xxs1, np.repeat(yLEs[None, :], 2*nn0-2+nn2, axis=0), zzs1)).transpose(1, 2, 0)[:, :, None, :]
    
    nny2 = nny[1] + 1
    output_blocks = [block[:nn1, :nny[0]], block[nn1-1:nn1+nn2-1, :nny[0]], block[nn1+nn2-2: -nn2+1, :nny[0]], block[-nn2:, :nny[0]],
                     block[:nn1, nny[0]:nny[0]+nny2], block[nn1-1:nn1+nn2-1, nny[0]:nny[0]+nny2], block[nn1+nn2-2: -nn2+1, nny[0]:nny[0]+nny2], block[-nn2:, nny[0]:nny[0]+nny2],
                    ] 

    # print(np.mean(output_blocks[7][:, -2, 0], axis=0), np.mean(output_blocks[7][:, -1, 0], axis=0))
    tip_tail_dr = np.mean(output_blocks[7][:, -2, 0] - output_blocks[7][:, -1, 0], axis=0)
    tip_blocks = scaling_tip(xxs, zzs[:, -1], wingCoefs, rdx=tip_tail_dr[0]/tip_tail_dr[1],  rdz=tip_tail_dr[2]/tip_tail_dr[1])
    ntip_blocks = [move_block(blk, translate=np.array([xLEs[-1], yLEs[-1], zLEs[-1]]), scale=chords[-1], angle=tws[-1], axis=np.array([0, 1, 0])) for blk in tip_blocks]

    # rotate block (airfoil at x-z to airfoil at x-y)
    output_blocks = [block[:, :, :, [0, 2, 1]] for block in output_blocks + ntip_blocks]
    for block in output_blocks:
        block[:, :, :, 2] = -block[:, :, :, 2]
    
    for i in range(len(output_blocks)):

        if np.isnan(output_blocks[i]).any():
            print(i)
    
    output_plot3d_concat(output_blocks, fname='wing.xyz', order='ij')
    
    return wingCoefs

def cal_ds(Uinf, cri_yplus, rhoinf, Rex):
    
    miu = 1.8375e-5
        
    # A flat plate with the 1/7 power law
    niu = miu/rhoinf
    Cf = 0.0576*(Rex)**(-0.2)
    
    uT = (0.5 * Cf * Uinf**2)**0.5

    return cri_yplus * niu / uT  # ds*uT/niu

@redirect_stdout('volume_meshing.log')
def volume_meshing(wingCoef: dict, input, ds: float = 1e-5, march_distance: float = 50., 
                   folder: str = '.', output_plot3d: bool = False):
    #*-------------------------------------------------
    #* Wing grid generation
    
    # rst general
    
    input_file = os.path.join(folder, input)
    output_file = os.path.join(folder, "wing_vol.cgns")
    
    symm0 = {'jLow': 'zSymm'}
        
    options = {
        # ---------------------------
        #   General options
        # ---------------------------
        "inputFile": input_file,
        "fileType": "PLOT3D",
        "unattachedEdgesAreSymmetry": True,
        "outerFaceBC": "farfield",
        "autoConnect": True,
        "BC": {1: symm0, 2: symm0, 3: symm0, 4: symm0},
        "families": "wall",
        # rst grid
        # ---------------------------
        #   Grid Parameters
        # ---------------------------
        # "N": 4,    # originally 49
        "N": mesh_point_numbers['nFar'],    # originally 49
        "s0": ds,
        "marchDist": march_distance,
        # rst pseudo
        # ---------------------------
        #   Pseudo Grid Parameters
        # ---------------------------
        "ps0": -1.0,
        "pGridRatio": -1.0,
        "cMax": 1.0,
        # rst smoothing
        # ---------------------------
        #   Smoothing parameters
        # ---------------------------
        "epsE": 1.0,
        "epsI": 2.0,
        "theta": 3.0,
        "volCoef": 0.2,
        "volBlend": 0.0005,
        "volSmoothIter": 20,
        # rst solution
        # ---------------------------
        #   Solution Parameters
        # ---------------------------
        "kspRelTol": 1e-10,
        "kspMaxIts": 1500,
        "kspSubspaceSize": 50,
    }
    # rst run pyHyp

    hyp = pyHyp(comm=comm, options=options)
    hyp.run()
    hyp.writeCGNS(output_file)
    
    if output_plot3d:
        grid = readGrid(fileName=output_file)
        grid.printBlockInfo()
        grid.writePlot3d(os.path.join(folder, 'wing_vol.xyz'))
        
    mesh_dict = {}
    mesh_dict['target_ds'] = ds
  
    return mesh_dict

def read_volumn_meshing_log(mesh_dict, logFile = 'volume_meshing.log'):
    
    with open(logFile, 'r') as f:
        lines = f.readlines()
    
    if not int(lines[16].split()[0]) == 2:
        print(lines[16])
        return mesh_dict
    
    mesh_history = np.loadtxt(logFile, comments='#', skiprows=16)
    # print(mesh_history)
    mesh_dict['min_Quality'] = np.min(mesh_history[:, 8])
    mesh_dict['min_Vol'] = np.min(mesh_history[:, 9])
    mesh_dict['actual_ds'] = mesh_history[0, 10]
    return mesh_dict

if __name__ == "__main__":

    freestream_temperature = 300    # K
    reynold_number = 20.e6          # Million
    renew_mesh = True

    #*--------------------------------------
    #* Read Wing Coefficient
    
    curr_dir = os.getcwd()

    with open('input.json', 'r') as f:
        time.sleep(random.random())     # simutanously reading json may cause error
        wingCoefs = json.load(f)
    
    # AR, TR, ref_area decided by trape shape
    wingCoefs['half_span'] =  wingCoefs['AR'] * (1 + wingCoefs['TR']) * 1/4
    wingCoefs['ref_area'] = (1 + wingCoefs['TR']) * wingCoefs['half_span']  * 1/2
        
    mesh_dict = {}
    funcs_dict = {}
        
    if comm.rank == 0:
        # generating surface mesh
        wingCoefs = surface_meshing(wingCoefs, folder=curr_dir)

    ds = cal_ds(Uinf=np.mean(wingCoefs['M3']) * (1.4 * 287 * freestream_temperature)**0.5, cri_yplus=1.0, rhoinf=1.225, Rex=reynold_number)
    mesh_dict = volume_meshing(wingCoefs, input="wing.xyz", ds=ds, folder=curr_dir, output_plot3d=False)
    
    if comm.rank == 0:
        mesh_dict = read_volumn_meshing_log(mesh_dict)
        
        with open('mesh_info.json', 'w') as f:
            json.dump(mesh_dict, f)
            
        with open('input.json', 'w') as f:
            json.dump(wingCoefs, f)
    