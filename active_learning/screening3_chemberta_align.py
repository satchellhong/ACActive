"""

This script contains the main active learning loop that runs all experiments.

    Author: Derek van Tilborg, Eindhoven University of Technology, May 2023

"""

from math import ceil
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import WeightedRandomSampler
# from active_learning.nn import RfEnsemble, Ensemble_triple, AcqModel
from active_learning.nn_align import RfEnsemble, Ensemble_triple, AcqModel
from active_learning.data_prep import MasterDataset
from active_learning.data_handler import Handler
from active_learning.utils import Evaluate, to_torch_dataloader, to_torch_dataloader_multi
from active_learning.acquisition import Acquisition, logits_to_pred
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
import logging
from collections import defaultdict
import torch.nn.functional as F

INFERENCE_BATCH_SIZE = 1024
TRAINING_BATCH_SIZE = 64
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def change_scale(x, m):
    if m == 'min':
        if x < 0:
            x = x*1.1
        else:
            x = x*0.9
    else:
        if x < 0:
            x = x*0.9
        else:
            x = x*1.1
    return x


def draw_fig(train_emb, test_emb, screen_emb, pick_emb, y_train, y_test, y_screen, y_pick, dir_name, cycle, emb, dataset, name, mlp_embedding, fp_y1):
    pca_fit = PCA(n_components=2).fit(np.concatenate([train_emb, screen_emb]))

    pca_train = pca_fit.transform(train_emb)
    pca_total = pca_fit.transform(np.concatenate([train_emb, screen_emb]))
    # pca_test = pca_fit.transform(test_emb)
    pca_screen = pca_fit.transform(screen_emb)
    pca_pick = pca_fit.transform(pick_emb)

    os.makedirs(dir_name+f"/{emb}", exist_ok=True)
    
    positive_train = pca_train[y_train == 1]
    negative_train = pca_train[y_train == 0]
    positive_pick = pca_pick[y_pick == 1]
    negative_pick = pca_pick[y_pick == 0]
    positive_screen = pca_screen[y_screen == 1]
    negative_screen = pca_screen[y_screen == 0]
    xmin = change_scale(min(np.concatenate([positive_train[:, 0], negative_train[:, 0], positive_pick[:, 0], negative_pick[:, 0], positive_screen[:, 0], negative_screen[:, 0]])), 'min')
    xmax = change_scale(max(np.concatenate([positive_train[:, 0], negative_train[:, 0], positive_pick[:, 0], negative_pick[:, 0], positive_screen[:, 0], negative_screen[:, 0]])), 'max')
    ymin = change_scale(min(np.concatenate([positive_train[:, 1], negative_train[:, 1], positive_pick[:, 1], negative_pick[:, 1], positive_screen[:, 1], negative_screen[:, 1]])), 'min')
    ymax = change_scale(max(np.concatenate([positive_train[:, 1], negative_train[:, 1], positive_pick[:, 1], negative_pick[:, 1], positive_screen[:, 1], negative_screen[:, 1]])), 'max')
    plt.figure(figsize=(8,6))
    plt.scatter(positive_train[:, 0], positive_train[:, 1], label='positive', alpha=0.7, s=10, c = 'red')
    plt.title(dataset + ' train')
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(dir_name + f"/{emb}/{emb}_train_pos_{cycle}.png")
    plt.show()
    plt.clf()

    plt.scatter(negative_train[:, 0], negative_train[:, 1], label='negative', alpha=0.5, s=10, c = 'grey')
    plt.scatter(positive_train[:, 0], positive_train[:, 1], label='positive', alpha=0.7, s=10, c = 'red')
    plt.title(dataset + ' train')
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(dir_name + f"/{emb}/{emb}_train_{cycle}.png")
    plt.show()
    plt.clf()

    plt.scatter(negative_screen[:, 0], negative_screen[:, 1], label='negative', alpha=0.5, s=10, c = 'grey')
    plt.scatter(positive_screen[:, 0], positive_screen[:, 1], label='positive', alpha=0.7, s=10, c = 'red')
    plt.title(dataset + ' screen')
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(dir_name + f"/{emb}/{emb}_screen_{cycle}.png")
    plt.show()
    plt.clf()

    plt.scatter(negative_pick[:, 0], negative_pick[:, 1], label='negative', alpha=0.5, s=10, c = 'grey')
    plt.scatter(positive_pick[:, 0], positive_pick[:, 1], label='positive', alpha=0.7, s=10, c = 'red')
    plt.title(dataset + ' pick')
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(dir_name + f"/{emb}/{emb}_pick_{cycle}.png")
    plt.show()
    plt.clf()
    if emb == 'emb':
        for i in range(len(name)):
            fp_x = np.squeeze(mlp_embedding[i].detach().cpu().numpy())
            fp_y = fp_y1[i]
            plt.figure(figsize=(8,6))
            if len(fp_x) > 0:
                pca_fp = pca_fit.transform(fp_x)
                negative_mlp = pca_fp[fp_y == 0]
                    
                plt.scatter(negative_screen[:, 0], negative_screen[:, 1], label='negative', alpha=0.3, s=10, c = 'grey')
                plt.scatter(positive_screen[:, 0], positive_screen[:, 1], label='positive', alpha=0.7, s=10, c = 'red')
                plt.scatter(negative_mlp[:, 0], negative_mlp[:, 1], label='mlp_negative', alpha=0.7, s=10, c = 'blue')
                plt.title(dataset + ' compare with mlp')
                plt.xlim((xmin, xmax))
                plt.ylim((ymin, ymax))
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.legend(fontsize=9)
                plt.savefig(dir_name + f"/{emb}/{emb}_{name[i]}_{cycle}.png")
                plt.cla()
                plt.clf()
                plt.close()
                if np.sum(fp_y) > 0:
                    positive_mlp = pca_fp[fp_y == 1]
                    plt.scatter(negative_screen[:, 0], negative_screen[:, 1], label='negative', alpha=0.3, s=10, c = 'grey')
                    plt.scatter(positive_screen[:, 0], positive_screen[:, 1], label='positive', alpha=0.7, s=10, c = 'red')
                    plt.scatter(positive_mlp[:, 0], positive_mlp[:, 1], label='mlp_positive', alpha=0.7, s=10, c = 'blue')
                    plt.title(dataset + ' compare with mlp')
                    plt.xlim((xmin, xmax))
                    plt.ylim((ymin, ymax))
                    plt.xlabel('PC1')
                    plt.ylabel('PC2')
                    plt.legend(fontsize=9)
                    plt.savefig(dir_name + f"/{emb}/{emb}_{name[i]}_pos_{cycle}.png")
                    plt.cla()
                    plt.clf()
                    plt.close()

from sklearn.neighbors import NearestNeighbors
def contrastive_active(smiles, labeled_emb, y_labeled, unlabeled_emb, unlabeled_preds, k=10, n=64):
    labeled_emb = torch.mean(labeled_emb, dim=1)
    unlabeled_emb = torch.mean(unlabeled_emb, dim=1)
    y_labeled = np.array(y_labeled)

    y_hat = torch.mean(torch.exp(unlabeled_preds), dim=1)
    y_hat = y_hat.cpu() if type(y_hat) is torch.Tensor else torch.tensor(y_hat)
    unlabeled_preds = y_hat[:, 1]

    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(labeled_emb.detach().cpu().numpy())

    distances, indices = knn.kneighbors(unlabeled_emb.detach().cpu().numpy())
    scores = []
    for i in range(unlabeled_emb.shape[0]):
        x_pred = unlabeled_preds[i].item()
        neighbor_indices = indices[i]
        neighbor_preds = y_labeled[neighbor_indices]
        diff = np.abs(x_pred - neighbor_preds)
        score = diff.mean()
        scores.append(score)
    picks_idx = torch.argsort(torch.tensor(scores), descending=True)[:n]

    return np.array([smiles[picks_idx.cpu()]]) if n == 1 else smiles[picks_idx.cpu()]

def ndcg_p(logits, y, k=64, p=1.5):
    """
    NDCG with power-scaled discount: d_i = 1 / (log2(i+1))**p

    logits : np.ndarray, 예측 점수 (높을수록 positive)
    y      : np.ndarray, 0/1 label
    k      : 상위 몇 개까지 평가 (None이면 전체)
    p      : 상위 가중 세기 (>1 이면 상위권 더 강조)
    """
    logits, y = np.asarray(logits), np.asarray(y)
    idx = np.argsort(-logits)
    y_sorted = y[idx]
    n = len(y_sorted)
    if k is None or k > n:
        k = n

    ranks = np.arange(1, k+1)
    discounts = 1.0 / (np.log2(ranks + 1) ** p)

    dcg = (y_sorted[:k] * discounts).sum()
    ideal = np.sort(y)[::-1][:k]
    idcg = (ideal * discounts).sum()
    return 0.0 if idcg == 0 else float(dcg / idcg)

def precision_at_ratio(logits, y, ratio=0.25):
    logits, y = np.asarray(logits), np.asarray(y)
    N = len(y)
    if N == 0:
        return 0.0
    K = max(1, int(ratio * N))
    idx = np.argsort(-logits)
    topk_y = y[idx][:K]
    return topk_y.mean()

def ef_at_ratio(logits, y, ratio=0.25):
    logits, y = np.asarray(logits), np.asarray(y)
    N = len(y)
    if N == 0:
        return 0.0
    pi = y.mean()
    if pi == 0:
        return 0.0
    p_at_r = precision_at_ratio(logits, y, ratio)
    return p_at_r / pi

def active_learning(dir, n_start: int = 64, acquisition_method: str = 'exploration', max_screen_size: int = None,
                    batch_size: int = 16, architecture: str = 'gcn', seed: int = 0, bias: str = 'random',
                    optimize_hyperparameters: bool = False, ensemble_size: int = 1, retrain: bool = True,
                    anchored: bool = True, dataset: str = 'ALDH1', scrambledx: bool = False,
                    scrambledx_seed: int = 1, cycle_threshold=1, beta=0, start = 0, feature = '',
                    hidden = 512, at_hidden = 64, layer = '', cycle_rnn = 0, lmda = 0.01) -> pd.DataFrame:
    """
    :param n_start: number of molecules to start out with
    :param acquisition_method: acquisition method, as defined in active_learning.acquisition
    :param max_screen_size: we stop when this number of molecules has been screened
    :param batch_size: number of molecules to add every cycle
    :param architecture: 'gcn', 'mlp', or 'rf'
    :param seed: int 1-20
    :param bias: 'random', 'small', 'large'
    :param optimize_hyperparameters: Bool
    :param ensemble_size: number of models in the ensemble, default is 10
    :param scrambledx: toggles randomizing the features
    :param scrambledx_seed: seed for scrambling the features
    :return: dataframe with results
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="X does not have valid feature names")
    set_seed(seed)

    
    pick_map = {'fp': f'./results_0915/50. {dataset}_exploitation_mlp/{seed}/picked.csv', 
                'fp_new': f'./results_0924/75-1. fp+mf+um_attention_2/75-1. {dataset}_exploitation_mlp_fp+mf+um_0_nonorm_{hidden}_{at_hidden}/{seed}/picked.csv', 
                f'fp1_fp+mf{start}': f'./results_0915/57. fp+molformer/57. {dataset}_exploitation_mlp_1/{seed}/picked.csv'}

    # Load the datasets
    representation = 'ecfp' if architecture in ['mlp', 'rf', 'lgb', 'xgb', 'svm'] else 'graph'
    ds_screen = MasterDataset('screen', representation=representation, feature = feature, dataset=dataset, scramble_x=scrambledx,
                              scramble_x_seed=scrambledx_seed)
    ds_test = MasterDataset('test', representation=representation, feature = feature, dataset=dataset)
    # Initiate evaluation trackers
    eval_test, eval_screen, eval_train = Evaluate(), Evaluate(), Evaluate()
    handler = Handler(n_start=n_start, seed=seed, bias=bias, dataset=dataset)

    dir_name = f"{dir}/{seed}"
    if seed == 0:
        for handlers in logging.root.handlers[:]:
            logging.root.removeHandler(handlers)
            
    logging.basicConfig(
        level=logging.INFO,                  # 로그 레벨 (DEBUG < INFO < WARNING < ERROR < CRITICAL)
        format="%(asctime)s [%(levelname)s] %(message)s",  # 로그 포맷
        handlers=[
            logging.FileHandler(f'{dir_name}/log.log'),  # 파일에 저장
            logging.StreamHandler()          # 콘솔에도 출력
        ]
    )
    # Define some variables
    hits_discovered, total_mols_screened, all_train_smiles = [], [], []
    max_screen_size = len(ds_screen) if max_screen_size is None else max_screen_size

    # build test loader
    x_test, y_test, smiles_test, fp_test = ds_test.all()

    n_cycles = ceil((max_screen_size - n_start) / batch_size)
    # exploration_factor = 1 / lambd^x. To achieve a factor of 1 at the last cycle: lambd = 1 / nth root of 2
    lambd = 1 / (2 ** (1/n_cycles))

    ACQ = Acquisition(method=acquisition_method, seed=seed, lambd=lambd)
    top_result = pd.DataFrame(columns=['cycle', '64', '100', '500', '1000'])
    # While max_screen_size has not been achieved, do some active learning in cycles
    picked_list = pd.DataFrame(columns=[i for i in range(64)])
    prediction_list = defaultdict(list)
    # if cycle_rnn > 0:
    #     picked_df = pd.read_csv(pick_map['fp_new'], index_col=0).T
    #     indexes = np.array([str(i) for i in range(0,min(64*(cycle_rnn+1), 1000))])
    #     mlp_smiles = np.array(picked_df.loc[indexes, 0])
    #     handler.set(mlp_smiles)
    M2 = None
    lamda = 0
    model_list = []
    model_value = []
    picked_model = ""
    label_list = []
    for cycle in tqdm(range(0, n_cycles+1)):
        # if cycle > 5:
        #     ds_screen = MasterDataset('screen', representation=representation, dataset=dataset, scramble_x=scrambledx, scramble_x_seed=scrambledx_seed)
        #     ds_test = MasterDataset('test', representation=representation, dataset=dataset)

        # Get the train and screen data for this cycle
        train_idx, screen_idx = handler()
        x_train, y_train, smiles_train, fp_train = ds_screen[train_idx]

        ds_screen.update_brics(smiles_train, smiles_train[y_train == 1])
        ds_test.update_brics(smiles_train, smiles_train[y_train == 1])

        x_train, y_train, smiles_train, fp_train = ds_screen[train_idx]
        x_screen, y_screen, smiles_screen, fp_screen = ds_screen[screen_idx]
        x_test, y_test, smiles_test, fp_test = ds_test.all()


        # Update some tracking variables
        all_train_smiles.append(';'.join(smiles_train.tolist()))
        hits_discovered.append(sum(y_train).item())
        hits = smiles_train[np.where(y_train == 1)]
        total_mols_screened.append(len(y_train))

        # collected_idx = handler.get_cum()
        # x_collected, y_collected, smiles_collected, fp_collected = ds_screen[collected_idx]
        # all_train_smiles.append(';'.join(smiles_collected.tolist()))
        # hits_discovered.append(sum(y_collected).item())
        # hits = smiles_collected[np.where(y_collected == 1)]
        # total_mols_screened.append(len(y_collected))

        logging.info(f'hit: {cycle}, {len(hits)}, {len(y_train)}')

        if len(train_idx) >= max_screen_size:
            break

        if architecture not in ['rf', 'lgb', 'xgb', 'svm']:
            # Get class weight to build a weighted random sampler to balance out this data
            class_weights = [1 - sum((y_train == 0) * 1) / len(y_train), 1 - sum((y_train == 1) * 1) / len(y_train)]
            weights = [class_weights[i] for i in y_train]
            sampler = WeightedRandomSampler(weights, num_samples=len(y_train), replacement=True)

            model_idx = -1
            model_value_max = 1
            for i in range(max(0, cycle-5), len(model_list)):
                # score_train = -1
                # if i+1 < cycle:
                #     pick_loader = to_torch_dataloader_multi([row[i*64:-64] for row in fp_train], x_train[i*64:-64], y_train[i*64:-64],
                #                                             batch_size=INFERENCE_BATCH_SIZE,
                #                                             shuffle=False, pin_memory=True)
                #     picked_logit = model_list[i].predict(pick_loader, dir_name, '', cycle).cpu()
                #     probs = F.softmax(picked_logit, dim=-1)
                #     picked_logit = torch.mean(probs, dim=1)[:, 1]
                #     sorted_idx = torch.argsort(picked_logit, descending=True)
                #     sorted_y = y_train[sorted_idx].cpu().numpy()
                #     score_train = ndcg_p(picked_logit.numpy(), sorted_y, k=64, p=2)
                # pick_loader = to_torch_dataloader_multi([row[-64:] for row in fp_train], x_train[-64:], y_train[-64:],
                #                                         batch_size=INFERENCE_BATCH_SIZE,
                #                                         shuffle=False, pin_memory=True)        
                # picked_logit = model_list[i].predict(pick_loader, dir_name, '', cycle).cpu()
                # probs = F.softmax(picked_logit, dim=-1)
                # picked_logit = torch.mean(probs, dim=1)[:, 1]
                # sorted_idx = torch.argsort(picked_logit, descending=True)
                # sorted_y = y_train[sorted_idx + cycle*64].cpu().numpy()
                # score = ndcg_p(picked_logit.numpy(), sorted_y, k=64, p=1)\
                
                # model_value[i].append((score, score_train))
                # indent = ' ' * (16 * i)
                # print(indent + ' / '.join(f"{v1:.4f},{v2:.4f}" for v1, v2 in model_value[i]))
                
                pick_loader = to_torch_dataloader_multi([row[(i+1)*64:] for row in fp_train], x_train[(i+1)*64:], y_train[(i+1)*64:],
                                                        batch_size=INFERENCE_BATCH_SIZE,
                                                        shuffle=False, pin_memory=True)        
                picked_logit = model_list[i].predict_kd(pick_loader, dir_name, '', cycle).cpu()
                probs = F.softmax(picked_logit, dim=-1)
                picked_logit = torch.mean(probs, dim=1)[:, 1]
                sorted_idx = torch.argsort(picked_logit, descending=True)
                sorted_y = y_train[sorted_idx+(i+1)*64].cpu().numpy()
                score = ef_at_ratio(picked_logit, sorted_y, ratio=0.25*cycle_rnn)
                # print(sorted_y)
                # print(score)
                
                model_value[i].append(score)
                indent = ' ' * (7 * i)
                if score > model_value_max:
                    model_value_max = score
                    model_idx = i
                # print(indent + ' '.join(f"{v:.4f}" for v in model_value[i]))
            if model_idx != -1:
                # lamda = 0.5
                lamda = max(min(0.5, (model_value_max-1)), 0.1)
            else:
                lamda = 0
            picked_model += f'({cycle}: {model_idx}, {round(lamda,4)})'
            # if cycle >= start:
            
            traj = False
            if not traj or cycle <= cycle_rnn:
                train_loader = to_torch_dataloader_multi(fp_train, x_train, y_train,
                                                batch_size=INFERENCE_BATCH_SIZE,
                                                shuffle=False, pin_memory=True)
                
                train_loader_balanced = to_torch_dataloader_multi(fp_train, x_train, y_train, 
                                                            batch_size=TRAINING_BATCH_SIZE,
                                                            sampler=sampler,
                                                            shuffle=False, pin_memory=True)

                screen_loader = to_torch_dataloader_multi(fp_screen, x_screen, y_screen,
                                                    batch_size=INFERENCE_BATCH_SIZE,
                                                    shuffle=False, pin_memory=True)
                x_test, y_test, smiles_test, fp_test = ds_test.all()
                test_loader = to_torch_dataloader_multi(fp_test, x_test, y_test,
                                                batch_size=INFERENCE_BATCH_SIZE,
                                                shuffle=False, pin_memory=True)
            
            ########################################################################
            else:
                train_prediction_list = []
                screen_prediction_list = []
                for smi in smiles_train:
                    train_prediction_list.append(prediction_list[smi])
                for smi in smiles_screen:
                    screen_prediction_list.append(prediction_list[smi])
                # print(train_prediction_list)
                train_loader = to_torch_dataloader_multi(fp_train, x_train, y_train, train_prediction_list,
                                                batch_size=INFERENCE_BATCH_SIZE,
                                                shuffle=False, pin_memory=True)
                
                train_loader_balanced = to_torch_dataloader_multi(fp_train, x_train, y_train, train_prediction_list, 
                                                            batch_size=TRAINING_BATCH_SIZE,
                                                            sampler=sampler,
                                                            shuffle=False, pin_memory=True)

                screen_loader = to_torch_dataloader_multi(fp_screen, x_screen, y_screen, screen_prediction_list,
                                                    batch_size=INFERENCE_BATCH_SIZE,
                                                    shuffle=False, pin_memory=True)
                x_test, y_test, smiles_test, fp_test = ds_test.all()
                test_loader = to_torch_dataloader_multi(fp_test, x_test, y_test,
                                                batch_size=INFERENCE_BATCH_SIZE,
                                                shuffle=False, pin_memory=True)

            # Initiate and train the model (optimize if specified)
            print("Training model")
            if retrain or cycle == 0:
                n_hidden = 128 if cycle < 0 else 1024
                # if cycle < start:
                M = Ensemble_triple(seed=seed, ensemble_size=ensemble_size, architecture=architecture, hidden = hidden, at_hidden = at_hidden, layer = layer,
                                    in_feats = [len(fp_train[idx][0]) for idx in range(len(fp_train))], 
                                    n_hidden = n_hidden, anchored=anchored, activate=False)
                # if cycle == 0:
                if model_idx == -1:
                    M.train(train_loader_balanced, prev_model = None, lamda = lamda, verbose=False)
                else:
                    M.train(train_loader_balanced, prev_model = model_list[model_idx], lamda = lamda, verbose=False)
                M2 = M
                model_list.append(M)
                model_value.append([])
                # else:
                #     for _ in range(10):
                #         M.train(train_loader_balanced, mode = 'train', verbose=False)
                #         M.train(train_loader_balanced, mode = 'pick', verbose=False)

            # Do inference of the train/test/screen data
            print("Train/test/screen inference")
            train_logits_N_K_C = M.predict(train_loader, dir_name, 'train', cycle)
            # train_embedding = M.embedding(train_loader)
            eval_train.eval(train_logits_N_K_C, y_train)

            # test_logits_N_K_C = M.predict(test_loader, dir_name, '', cycle)
            # # test_embedding = M.embedding(test_loader)
            # eval_test.eval(test_logits_N_K_C, y_test)

            # test_logits_N_K_C = M.predict(train_loader, dir_name, '', cycle)
            # test_embedding = M.embedding(test_loader)
            eval_test.eval(train_logits_N_K_C, y_train)

            screen_logits_N_K_C = M.predict(screen_loader, dir_name, 'screen', cycle)
            eval_screen.eval(screen_logits_N_K_C, y_screen)

            screen_logits_N_K_C_2 = None
            ########## traj ################
            if traj:
                screen_mean_probs_hits = torch.mean(F.softmax(screen_logits_N_K_C, dim=-1), dim=1)[:, 1]
                train_mean_probs_hits = torch.mean(F.softmax(train_logits_N_K_C, dim=-1), dim=1)[:, 1]
                screen_mean_probs_hits = screen_mean_probs_hits.cpu()
                train_mean_probs_hits = train_mean_probs_hits.cpu()
                for smi, prob in zip(smiles_screen, screen_mean_probs_hits):
                    prediction_list[smi].append(prob)
                for smi, prob in zip(smiles_train[-64:], train_mean_probs_hits[-64:]):
                    prediction_list[smi].append(prob)
                if cycle > cycle_rnn:
                    print('round')
                    train_seq_list = []
                    train_label_list = []
                    for i, x in enumerate(smiles_train):
                        if len(prediction_list[x]) > 0:
                            train_seq_list.append(prediction_list[x])
                            train_label_list.append(y_train[i])
                            if len(prediction_list[x]) >= cycle:
                                print(*[f"{t.item():.4f}" for t in prediction_list[x]], "->", int(y_train[i]))
                    # screen_seq_list = []
                    # for i, x in enumerate(smiles_screen):
                    #     screen_seq_list.append(prediction_list[x])
                    # ACQ = Acquisition(method='round_exploitation', seed=seed, lambd=lambd)
                    # traj_model = AcqModel()
                    # traj_model.train(train_seq_list, train_label_list)
                    # screen_logits_N_K_C_2 = traj_model.predict(screen_seq_list)
                    # # idx = torch.argsort(screen_logits_N_K_C, descending=True)
                    # # print(screen_logits_N_K_C[idx[:64]])
                    # if cycle == 15:
                    #     csv = pd.concat([
                    #         pd.DataFrame([[float(f"{t.item():.4f}") for t in seq]])  # tensor → float(4자리)
                    #         for seq in train_seq_list
                    #     ], ignore_index=True)
                    #     csv["label"] = [int(l.item()) for l in train_label_list]
                    #     csv.to_csv(f'{dir_name}/picked_seq.csv', float_format="%.4f")

            ################################

            # for x in screen_logits_N_K_C:

            # screen_loss = M.predict_loss(screen_loader)
            # screen_embedding = M.embedding(screen_loader)


        else:
            print("Training model")
            if retrain or cycle == 0:
                M = RfEnsemble(seed=seed, ensemble_size=ensemble_size, architecture=architecture)
                # if cycle == 0 and optimize_hyperparameters:
                #     M.optimize_hyperparameters(x_train, y_train)
                M.train(x_train, y_train, verbose=False)

            # Do inference of the train/test/screen data
            print("Train/test/screen inference")
            train_logits_N_K_C = M.predict(x_train)
            eval_train.eval(train_logits_N_K_C, y_train)

            test_logits_N_K_C = M.predict(x_test)
            eval_test.eval(test_logits_N_K_C, y_test)

            screen_logits_N_K_C = M.predict(x_screen)
            eval_screen.eval(screen_logits_N_K_C, y_screen)

        # If this is the second to last cycle, update the batch size, so we end at max_screen_size
        if len(train_idx) + batch_size > max_screen_size:
            batch_size = max_screen_size - len(train_idx)

        # Select the molecules to add for the next cycle
        print("Sample acquisition")
        print('hit'+str(cycle)+' : ', len(hits))

        # picks = ACQ.acquire(screen_logits_N_K_C, smiles_screen, hits=hits, n=batch_size, seed=seed, cycle=cycle, y_screen = y_screen, dir_name=dir_name)
        # ACQ = Acquisition(method='loss_prediction', seed=seed, lambd=lambd)
        picks = ACQ.acquire(screen_logits_N_K_C, smiles_screen, screen_loss=screen_logits_N_K_C_2, hits=hits, n=batch_size, seed=seed, cycle=cycle, y_screen = y_screen, dir_name=dir_name, beta=beta)
        # ACQ = Acquisition(method=acquisition_method, seed=seed, lambd=lambd)
        handler.add(picks)
        picked_np = np.stack([t.numpy() for t in ds_screen[handler.get_add(picks)][1]])
        if len(picked_np) < 64:
            picked_np = np.append(picked_np, np.zeros(64-len(picked_np)))
        picked_list.loc[len(picked_list.index)] = picked_np
        top_screen = []
        top_test = []
        top64 = ds_screen[handler.get_add(picks)][1]
        for topN in [100, 500, 1000]:#, 5000]:
            picks1 = ACQ.acquire(screen_logits_N_K_C, smiles_screen, screen_loss=screen_logits_N_K_C_2, hits=hits, n=topN, beta=beta)
            top_screen.append(ds_screen[handler.get_add(picks1)][1])
            picks1 = ACQ.acquire(screen_logits_N_K_C, smiles_screen, screen_loss=screen_logits_N_K_C_2, hits=hits, n=topN, beta=beta)
            top_test.append(ds_screen[handler.get_add(picks1)][1])

        # pick_idx = []
        # for screen_i in range(len(smiles_screen)):
        #     if smiles_screen[screen_i] in picks:
        #         pick_idx.append(screen_i)
        # pick_idx = np.array(pick_idx)
        # y_picked = y_screen[pick_idx]
        
        # pick_loader = to_torch_dataloader_multi([row[pick_idx] for row in fp_screen], x_screen[pick_idx], y_picked,
        #                                         batch_size=INFERENCE_BATCH_SIZE,
        #                                         shuffle=False, pin_memory=True)


        if dataset == 'ALDH1':
            lamda = max(0, sum(top64).item() / 64 - 0.25)
        else:
            lamda = max(0, sum(top64).item()*10 / 64 - 0.2)
        print('top64: '+str(sum(top64).item())+' \ttop100: '+str(sum(top_screen[0]).item())+' \ttop500: '+str(sum(top_screen[1]).item())+' \t top1000: '+str(sum(top_screen[2]).item()))
        # print('top64: '+str(sum(top64).item())+' \ttop100: '+str(sum(top_test[0]).item())+' \ttop500: '+str(sum(top_test[1]).item())+' \t top1000: '+str(sum(top_test[2]).item()))
        top_result.loc[len(top_result.index)] = [cycle, sum(top64).item(), sum(top_screen[0]).item(), sum(top_screen[1]).item(), sum(top_screen[2]).item()]
        eval_train.ef.append([sum(top_test[0]).item(),sum(top_test[1]).item(),sum(top_test[2]).item()])
        eval_test.ef.append([sum(top_test[0]).item(),sum(top_test[1]).item(),sum(top_test[2]).item()])
        eval_screen.ef.append([sum(top_screen[0]).item(),sum(top_screen[1]).item(),sum(top_screen[2]).item()])

        # pick_idx = []
        # for screen_i in range(len(smiles_screen)):
        #     if smiles_screen[screen_i] in picks:
        #         pick_idx.append(screen_i)
        # pick_idx = np.array(pick_idx)
        
        # pick_loader = to_torch_dataloader_multi([row[pick_idx] for row in fp_screen], x_screen[pick_idx], y_screen[pick_idx],
        #                                         batch_size=INFERENCE_BATCH_SIZE,
        #                                         shuffle=False, pin_memory=True)
        # M.predict(pick_loader, dir_name, 'pick', cycle)
        # print('pick_idx : ', len(pick_idx))
        # y_picks = y_screen[pick_idx]
        # pick_embedding = screen_embedding[pick_idx]
        # fp_picks = fp_screen[pick_idx]
        # draw_fig(np.squeeze(train_embedding.detach().cpu().numpy()), np.squeeze(test_embedding.detach().cpu().numpy()), np.squeeze(screen_embedding.detach().cpu().numpy()),
        #         np.squeeze(pick_embedding.detach().cpu().numpy()), y_train, y_test, y_screen, y_picks, dir_name, cycle, 'emb', dataset, [], [], [])
        # draw_fig(np.squeeze(train_embedding.detach().cpu().numpy()), np.squeeze(test_embedding.detach().cpu().numpy()), np.squeeze(screen_embedding.detach().cpu().numpy()),
        #         np.squeeze(pick_embedding.detach().cpu().numpy()), y_train, y_test, y_screen, y_picks, dir_name, cycle, 'emb', dataset, ['mlp1', 'mlp2', 'mlp3', 'cum_mlp'], mlp_embedding, fp_y)
        # draw_fig(fp_train, fp_test, fp_screen, fp_picks, y_train, y_test, y_screen, y_picks, dir_name, cycle, 'fp', dataset)

        # draw_fig2(np.squeeze(train_embedding.detach().cpu().numpy()), np.squeeze(screen_embedding.detach().cpu().numpy()), y_screen, dir_name, cycle, 'emb', dataset, ['mlp1', 'mlp', 'cum_mlp'], mlp_embedding, fp_y)
        picked_list.to_csv(f'{dir_name}/picked_index.csv')

    # Add all results to a dataframe
    train_results = eval_train.to_dataframe("train_")
    test_results = eval_test.to_dataframe("test_")
    screen_results = eval_screen.to_dataframe('screen_')
    results = pd.concat([train_results, test_results, screen_results], axis=1)
    results['hits_discovered'] = hits_discovered
    results['total_mols_screened'] = total_mols_screened
    # results['all_train_smiles'] = all_train_smiles

    cycle = [i//64+1 for i in range(1000)]
    pd.DataFrame([smiles_train.tolist(), y_train.tolist(), cycle]).to_csv(f'{dir_name}/picked.csv')
    
    logging.info(picked_model)

    top_result.to_csv(f"{dir_name}/top_result_{dataset}_{acquisition_method}_{architecture}_ancher{anchored}.csv")

    return results

