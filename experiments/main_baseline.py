
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
from tqdm.auto import tqdm

import itertools
import argparse

PARAMETERS = {'max_screen_size': [1000],
              'n_start': [64],
              'batch_size': [64, 32, 16],
              'architecture': ['gcn', 'mlp', 'gin', 'gat', 'rf'],
              'dataset': ['ALDH1', 'PKM2', 'VDR'],
              'seed': list(range(0,5)),
              'bias': ['random', 'small', 'large'],
              'acquisition': ['random', 'exploration', 'exploitation', 'dynamic', 'dynamicbald', 'similarity', 'bald', 'epig', 'pi', 'ei', 'ts']
              }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='The path of the output directory', default='results')
    parser.add_argument('-acq', help="Acquisition function ('random', 'exploration', 'exploitation', 'dynamic', "
                                     "'similarity')", default='random')
    parser.add_argument('-bias', help='The level of bias ("random", "small", "large")', default='random')
    parser.add_argument('-arch', help='The neural network architecture ("gcn", "mlp")', default='mlp')
    parser.add_argument('-dataset', help='The dataset ("ALDH1", "PKM2", "VDR")', default='ALDH1')
    parser.add_argument('-retrain', help='Retrain the model every cycle', default='True')
    parser.add_argument('-batch_size', help='How many molecules we select each cycle', default=64)
    parser.add_argument('-n_start', help='How many molecules we have in our starting set (min=2)', default=64)
    parser.add_argument('-anchored', help='Anchor the weights', default='True')
    parser.add_argument('-scrambledx', help='Scramble the features', default='False')
    parser.add_argument('-scrambledx_seed', help='Seed for scrambling the features', default=1)
    parser.add_argument('-cycle_threshold', help='Seed for scrambling the features', default=-1)
    parser.add_argument('-cluster', help='Seed for scrambling the features', default=8)
    parser.add_argument('-test', help='Seed for scrambling the features', default=0)
    parser.add_argument('-sorted', help='Seed for scrambling the features', default='sorted')
    parser.add_argument('-start', help='Seed for scrambling the features', default=0)
    parser.add_argument('-feature', help='Seed for scrambling the features', default='fp')
    parser.add_argument('-hidden', help='Seed for scrambling the features', default=512)
    parser.add_argument('-at_hidden', help='Seed for scrambling the features', default=64)
    parser.add_argument('-layer', help='Seed for scrambling the features', default='')
    parser.add_argument('-cycle', help='Seed for scrambling the features', default=0)
    parser.add_argument('-beta', help='Seed for scrambling the features', default=0)
    args = parser.parse_args()
    fl = ['fp', 'cb', 'mf', 'um', 'gcn', 'gin', 'gat', 'ba']
    gnn_list = ['gcn', 'gin', 'gat']
    feature_list = ['fp', 'cb', 'mf', 'um', 'fp+cb', 'fp+mf', 'fp+um', 'cb+mf', 'cb+um', 'mf+um', 'fp+cb+mf', 'fp+cb+um', 'fp+mf+um', 'cb+mf+um', 'fp+cb+mf+um']
    feature_map = {'fp+mf+um':65, 'fp+cb+um':64, 'fp+cb+mf':63, 'fp+cb+mf+um':62}
    for feature in ['fp']: #'fp+mf+um', 'fp+cb+um', 'fp+cb+mf', 'fp+cb+mf+um'
        args.feature = feature
        # for start in [7, 9]:
        # for start in [1, 3, 5, 7, 9, 11, 13, 14]:
        # for start in [0, 2, 4, 6, 8, 10, 12]:
        for gnn in gnn_list:
            if gnn in args.feature:
                args.arch = gnn
                break
        else:
            args.arch = 'mlp'
        # for start in range(1):
        for hidn in [256, 512, 1024]:
            for lmda in [0.01, 0.02, 0.005, 0.05]:
                for rround in [0]:
                    args.hidden = hidn
                    for at_hidden in [2]:
                        # args.hidden = 512
                        args.at_hidden = args.hidden // at_hidden
                        for layer in ['']:#_layer2
                            args.layer = layer
                            for cycle in [2, 1, 0]:#2, 4, 12 / 6, 8, 10
                                args.cycle = cycle
                                for dataset in ['ALDH1', 'PKM2', 'VDR']:
                                    for beta in [0]: # , 0.5, 0.75, 1, 2
                                        args.beta = beta
                                        args.dataset = dataset
                                        start = 0
                                        args.start = start
                                        PARAMETERS['acquisition'] = [args.acq]
                                        PARAMETERS['bias'] = [args.bias]
                                        PARAMETERS['dataset'] = [args.dataset]
                                        PARAMETERS['retrain'] = [eval(args.retrain)]
                                        PARAMETERS['architecture'] = [args.arch]
                                        PARAMETERS['batch_size'] = [int(args.batch_size)]
                                        PARAMETERS['n_start'] = [int(args.n_start)]
                                        PARAMETERS['n_start'] = [int(args.n_start)]
                                        PARAMETERS['anchored'] = [eval(args.anchored)]
                                        PARAMETERS['scrambledx'] = [eval(args.scrambledx)]
                                        PARAMETERS['scrambledx_seed'] = [int(args.scrambledx_seed)]
                                        PARAMETERS['cycle_threshold'] = [int(args.cycle_threshold)]
                                        PARAMETERS['cluster'] = [int(args.cluster)]
                                        PARAMETERS['sorted'] =  [args.sorted]
                                        PARAMETERS['start'] =  [int(args.start)]
                                        PARAMETERS['feature'] =  [args.feature]
                                        PARAMETERS['hidden'] =  [int(args.hidden)]
                                        PARAMETERS['at_hidden'] =  [int(args.at_hidden)]
                                        PARAMETERS['layer'] =  [args.layer]
                                        PARAMETERS['cycle'] =  [args.cycle]
                                        PARAMETERS['beta'] =  [args.beta]
                                        PARAMETERS['rround'] =  [rround]
                                        PARAMETERS['lmda'] =  [lmda]
                                        
                                        file_name = f"76. {args.dataset}_{args.acq}_{args.arch}_{args.feature}_{args.start}_nonorm_{args.hidden}"
                                        dir_name = f"./results_1014_base/76. {args.feature}_mlp_cliff_{lmda}_{rround}_cliff{cycle}_sim0.6_128/{file_name}"
                                        
                                        if os.path.exists(dir_name):
                                            continue
                                        os.makedirs(dir_name, exist_ok=True)
                                        print(dir_name)
                                        for s in PARAMETERS['seed']:
                                            os.makedirs(dir_name+f"/{s}", exist_ok=True)
                                        LOG_FILE = f"{dir_name}/{file_name}.csv"

                                        if args.acq in['boundre', 'boundre_ac', 'boundre_bald']:
                                            from active_learning.screening4_0820 import active_learning
                                        elif args.acq in ['boundre_fp', 'cosine']:
                                            # from active_learning.screening4_0820 import active_learning
                                            from active_learning.screening4_0817 import active_learning
                                        else:
                                            if args.start == -1:
                                                from active_learning.screening3 import active_learning
                                            else:
                                                # from active_learning.screening3_chemberta import active_learning
                                                # from active_learning.screening3_chemberta import active_learning
                                                from active_learning.screening3_chemberta_contrastive import active_learning
                                            # from active_learning.screening4_0817 import active_learning
                                            # from active_learning.screening3_ssm import active_learning
                                            # from active_learning.screening3_murko import active_learning
                                            # from active_learning.screening4_0821_cliff import active_learning
                                        
                                        # LOG_FILE = './results/result_ALDH_contrastive_gnn.csv'
                                        # LOG_FILE = args.o

                                        # PARAMETERS['acquisition'] = ['random']
                                        # PARAMETERS['bias'] = ['random']
                                        # PARAMETERS['architecture'] = ['gcn']
                                        # LOG_FILE = f"results/{PARAMETERS['architecture'][0]}_{PARAMETERS['acquisition'][0]}_{PARAMETERS['bias'][0]}_simulation_results.csv"

                                        experiments = [dict(zip(PARAMETERS.keys(), v)) for v in itertools.product(*PARAMETERS.values())]


                                        hit_sum = 0
                                        hit_round = 0
                                        for experiment in tqdm(experiments):
                                            if args.acq in ['exploitation', 'bald']:
                                                print(f'{dir_name} / seed {experiment["seed"]}')
                                                results = active_learning(dir=dir_name, n_start=experiment['n_start'],
                                                                        bias=experiment['bias'],
                                                                        acquisition_method=experiment['acquisition'],
                                                                        max_screen_size=experiment['max_screen_size'],
                                                                        batch_size=experiment['batch_size'],
                                                                        architecture=experiment['architecture'],
                                                                        seed=experiment['seed'],
                                                                        retrain=experiment['retrain'],
                                                                        anchored=experiment['anchored'],
                                                                        dataset=experiment['dataset'],
                                                                        scrambledx=experiment['scrambledx'],
                                                                        scrambledx_seed=experiment['scrambledx_seed'],
                                                                        optimize_hyperparameters=False,
                                                                        cycle_threshold=experiment['rround'],
                                                                        beta = experiment['beta'],
                                                                        start = experiment['start'],
                                                                        feature = experiment['feature'], 
                                                                        hidden=experiment['hidden'],
                                                                        at_hidden=experiment['at_hidden'],
                                                                        layer=experiment['layer'],
                                                                        cycle_rnn=experiment['cycle'],
                                                                        lmda=experiment['lmda'])
                                            else:
                                                results = active_learning(dir=dir_name, n_start=experiment['n_start'],
                                                                        bias=experiment['bias'],
                                                                        acquisition_method=experiment['acquisition'],
                                                                        max_screen_size=experiment['max_screen_size'],
                                                                        batch_size=experiment['batch_size'],
                                                                        architecture=experiment['architecture'],
                                                                        seed=experiment['seed'],
                                                                        retrain=experiment['retrain'],
                                                                        anchored=experiment['anchored'],
                                                                        dataset=experiment['dataset'],
                                                                        scrambledx=experiment['scrambledx'],
                                                                        scrambledx_seed=experiment['scrambledx_seed'],
                                                                        optimize_hyperparameters=False,
                                                                        cycle_threshold=experiment['cycle_threshold'],
                                                                        n_clusters = experiment['cluster'])


                                            # Add the experimental settings to the outfile
                                            results['acquisition_method'] = experiment['acquisition']
                                            results['architecture'] = experiment['architecture']
                                            results['n_start'] = experiment['n_start']
                                            results['batch_size'] = experiment['batch_size']
                                            results['seed'] = experiment['seed']
                                            results['bias'] = experiment['bias']
                                            results['retrain'] = experiment['retrain']
                                            results['scrambledx'] = experiment['scrambledx']
                                            results['scrambledx_seed'] = experiment['scrambledx_seed']
                                            results['dataset'] = experiment['dataset']

                                            results.to_csv(LOG_FILE, mode='a', index=False, header=False if os.path.isfile(LOG_FILE) else True)
                                            hit_sum += results['hits_discovered'].iloc[-1]
                                            hit_round += 1
                                            if dataset == 'ALDH1' and results['hits_discovered'].iloc[-1] < 180:
                                                break
                                            if dataset == 'PKM2' and results['hits_discovered'].iloc[-1] < 7:
                                                break
                                            # if dataset == 'VDR' and results['hits_discovered'].iloc[-1] < 17:
                                            #     break
                                            # if dataset == 'ALDH1' and 270*5 - hit_sum > 340*(5-hit_round):
                                            #     break
                                            # if dataset == 'PKM2' and 30*5 - hit_sum > 40*(5-hit_round):
                                            #     break
                                            # if dataset == 'VDR' and 44*5 - hit_sum > 60*(5-hit_round):
                                            #     break

'''
0. 기본본
1. scaffold
2. scaffold concat
3. scaffold, ecfp를 번갈아가면서
4. exploitation, bald 번갈아
5. bald, exploitation 번갈아
6. exploitation 32개 + bald 32개
7. bald 32개 + exploitation 32개
8. no sampling
9. cliff weight + 1
10. cliff weight + similarity
11. cliff weight + similarity, no_sampler
12. ssm
13. feature dim curriculum
14. cliff weight + similarity, no_sampler 다시
15. feature dim curriculum cycle 8에서 radius3, fp 2048
16. radius3, fp 2048
17. boundre_fp
18. boundre_fp + mlp
19. cluster + mlp
20. finetuning ac boundre
21. seed
22. DrugCLIP
23. Chemberta
24. 이전 DATA 버리기
25. iteration 200
26. anchor False iter 200
27. anchor False 
28. r3 1024
29. r2 256 
30. r3 256 
31. ranking loss
32. boundre_fp, mlp 비교
33.
34.
35. similarity
36. boundre 조정, similarity
37. boundre 조정 
test. boundre 조정 neg 2 
40. boundre 조정 neg log2
41. boundre 조정 neg log2 MLP데이터로 학습
42. gcn, gat, gin
43. residual
44. dropout
45. new mlp
46. new mlp
47. 그 다음라운드 찍어보기
48. gcn, gat, gin
49. cos sim
50. new mlp no ensemble
51. mixup
52. mlp hidden dimension 128
53. chemberta
54. FP + chemberta
55. cycle FP + chemberta
55-1. cycle chemberta
56. molformer
57. cycle FP + molformer
57-1. cycle molformer
58. unimol
59. cycle fp + unimol
59-1. cycle unimol
60. cycle fp + gnn
60-1. cycle gnn
61. brics나 murcko를 활용할 방법
 -> brics를 이용해 (positive aware fp) or (그냥 fp)를 만들고 그걸로??.
 -> brics를 만들 때 0, 1이 아니라 tanimoto similarity로??
61-1. cycle brics_pos
62. 3개 adaptive
65. ensemble mlp를 도입해보자.
84. 128
85. 64
76. cliff module ablation - fp 128
77. no cliff module fp
'''