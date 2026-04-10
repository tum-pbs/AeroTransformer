'''
Yang Yunjia @ Nov. 23 2024

Possible error code

1 -> not convergent (err_cl > 0.002)
2 -> mesh not generated
3 -> calculation fail
'''


import os
import time
import json
import subprocess
import random

def log(string: str = '', path: str = '.', init=False):
    print(string)
    if init:
        with open(path+'/running.log', 'w') as f:
            f.write('\n')
    
    with open(path+'/running.log', 'a') as f:
        f.write('[%s] %s \n'%(time.ctime(), string))

if __name__ == "__main__":

    freestream_temperature = 300    # K
    reynold_number = 20.e6          # Million
    err_cl_therhold = 0.002
    renew_mesh = True
    renew_cfd  = False

    log('Start running', init=True)
    #*--------------------------------------
    #* Read Wing Coefficient
    
    curr_dir = os.getcwd()

    with open('input.json', 'r') as f:
        wingCoefs = json.load(f)  

    t0 = time.perf_counter()

    log()
    log('---------------------------------------')
    log('Generate wing mesh')
    
    time.sleep(random.random())     # simutanously reading json may cause error

    if not os.path.exists('wing_vol.cgns') or renew_mesh:
        print('!')
        subprocess.run('mpirun -np 8 python gen-mesh.py', shell=True, check=False)
        print('?')
    
    t1 = time.perf_counter()
    if os.path.exists('mesh_info.json'):
        with open('mesh_info.json', 'r') as f:
            mesh_info = json.load(f)
        log('>  Mesh generated in %.2f (s)' % (t1 - t0))
        log('>  - Minimum quality: %.3f' % (mesh_info['min_Quality']))
    else:
        log('>  Mesh not generated, see volume_meshing.log')
        exit(2)
        
    log()
    log('---------------------------------------')
    log('ADflow simulations')
    
    output_results = []
    
    for ic in range(len(wingCoefs['M3'])):
        
        log('> [%4d] Start running Ma = %.3f  AoA = %.3f' % (ic, wingCoefs['M3'][ic], wingCoefs['AA'][ic]))
        
        subfolder = str(ic)
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
        
        if not os.path.exists(os.path.join(subfolder, 'output.json')) or renew_cfd:
            
            conds = {'M3': wingCoefs['M3'][ic], 'AA': wingCoefs['AA'][ic], 're': reynold_number, 'temp': freestream_temperature}
            
            with open(os.path.join(subfolder, 'conds.json'), 'w') as f:
                json.dump(conds, f)
                
            with open('run-adflow.py', 'r') as fin, open(os.path.join(subfolder, 'run-adflow.py'), 'w') as fout:
                lines = fin.readlines()
                fout.writelines(lines)
            
            result = subprocess.run(f'cd {subfolder} && mpirun -np 8 python run-adflow.py', shell=True, check=False)

    
        if os.path.exists(os.path.join(subfolder, 'output.json')):
            with open(os.path.join(subfolder, 'output.json'), 'r') as f:
                results = json.load(f)
                
            log('> [%4d] Done with CL = %.3f err_CL = %.3f' % (ic, results['results']['wing_cl'], results['results']['err_cl']))
            output_result = [ic, 0 if results['results']['err_cl'] < err_cl_therhold else 1, 
                                results['working_conditions']['AA'], results['working_conditions']['M3'], 
                                results['results']['wing_cl'], results['results']['wing_cd'], results['results']['err_cl'], results['results']['err_cd']]
        else:
            log('> [%4d] Failed' % (ic))
            output_result = [ic, 3]
        
        output_results.append(output_result)
        
    t4 = time.perf_counter()
    log('> CFD done. t= %.2f(s)'%(t4-t1))

    log()
    log('---------------------------------------')
    log('Collect results')
    
    with open('output.txt', 'w') as f:
        f.write('VARIABLES = iCondition  iState  AoA    Ma    CL    CD    errCL  errCD \n')
        for output_result in output_results:
            f.write(' %d  %d  ' % (output_result[0], output_result[1]))
            for output in output_result[2:]:
                f.write(' %.6f '%(output))   
            f.write('\n')  

    
