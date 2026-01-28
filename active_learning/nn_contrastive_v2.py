"""

This script contains all models:

    - MLP: a simple feed forward multi-layer perceptron. Supports weight anchoring - Pearce et al. (2018)
    - GCN: a simple graph convolutional NN - Kipf & Welling (2016). Supports weight anchoring - Pearce et al. (2018)
    - Model: A wrapper class that contains a train(), and predict() loop
    - Ensemble: Class that ensembles n Model classes. Contains a train() method and an predict() method that outputs
        logits_N_K_C, defined as [N, num_inference_samples, num_classes]. Also has an optimize_hyperparameters() method.

    Author: Derek van Tilborg, Eindhoven University of Technology, May 2023

"""

from copy import deepcopy
import sys
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, BatchNorm, GINConv
from tqdm.auto import trange
from active_learning.hyperopt import optimize_hyperparameters
from typing import Optional, List
import pandas as pd
import os
import random
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import AllChem
from active_learning.model_pretrain import PretrainGIN

sys.path.append('.')

class GIN(torch.nn.Module):
    def __init__(self, mol_emb_dim=130, hidden_dim=1024, output_dim=2, gin_graph_conv_layer=3, gin_x_fc_layer=3, gin_fp_fc_layer=3, classification=True, 
                 seed: int = 42, lr: float = 3e-4, epochs: int = 50, anchored: bool = True, l2_lambda: float = 3e-4, weight_decay=0,  need_pretrain=False, **kwargs):
        super().__init__()

        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay
        self.need_pretrain = need_pretrain
        self.beta = 0.1

        self.atom_embedding = torch.nn.Linear(mol_emb_dim, hidden_dim)

        self.graph_conv = torch.nn.ModuleList()
        self.graph_bn = torch.nn.ModuleList()
        for _ in range(gin_graph_conv_layer):
            gin_mlp = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                          torch.nn.ReLU())
            self.graph_conv.append(GINConv(nn=gin_mlp))
            self.graph_bn.append(BatchNorm(hidden_dim, allow_single_element=True))

        self.x_fc = torch.nn.ModuleList()
        self.x_bn = torch.nn.ModuleList()
        for i in range(gin_x_fc_layer):
            self.x_fc.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.x_bn.append(BatchNorm(hidden_dim, allow_single_element=True))
            
        self.fp_fc = torch.nn.ModuleList()
        self.fp_bn = torch.nn.ModuleList()
        for i in range(gin_fp_fc_layer):
            self.fp_fc.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.fp_bn.append(BatchNorm(hidden_dim, allow_single_element=True))

        if not classification:
            self.out = torch.nn.Linear(hidden_dim, 1)
            self.task = 'regression'
        else:
            self.out = torch.nn.Linear(hidden_dim, output_dim)
            self.task = 'classification'
            
        if need_pretrain:
            self.pretrain = PretrainGIN(emb_dim=300, layer_num=5)
            self.xp_fc = torch.nn.Linear(300, hidden_dim)

    def embed(self, graph):
        x, edge_index, xp, edgep_index, edgep_attr, batch, fp = graph.x, graph.edge_index, graph.xp, graph.edgep_index, graph.edgep_attr, graph.batch, graph.fp

        x = F.elu(self.atom_embedding(x))

        for conv, bn in zip(self.graph_conv, self.graph_bn):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            
        x = global_add_pool(x, batch)

        for fc, bn in zip(self.x_fc, self.x_bn):
            x = fc(x)
            x = bn(x)
            x = F.relu(x)

        for fc, bn in zip(self.fp_fc, self.fp_bn):
            fp = fc(fp)
            fp = bn(fp)
            fp = F.relu(fp)
        
        x = x + fp

        if self.need_pretrain:
            xp = self.pretrain(xp, edgep_index, edgep_attr, batch)
            xp = self.xp_fc(xp)

            x = x + self.beta * xp

        return x
    
    def forward(self, graph, need_emb=True, return_feature=False):
        if need_emb:
            x = self.embed(graph)
        else:
            x = graph

        emb = x

        x = self.out(x)
        if self.task == 'regression':
            if return_feature:
                return x, emb
            return x
        
        x = F.log_softmax(x, 1)

        if return_feature:
            return x, emb

        return x
    
class MLP(torch.nn.Module):
    def __init__(self, hidden_dim=1024, mlp_fc_layer=3, output_dim=2, classification=True, seed: int = 42, lr: float = 3e-4, 
                 epochs: int = 50, anchored: bool = True, l2_lambda: float = 3e-4, weight_decay=0, **kwargs):
        super().__init__()

        self.fc = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay
        for i in range(mlp_fc_layer):
            self.fc.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.bn.append(BatchNorm(hidden_dim, allow_single_element=True))
        if not classification:
            self.out = torch.nn.Linear(hidden_dim, 1)
            self.task = 'regression'
        else:
            self.out = torch.nn.Linear(hidden_dim, output_dim)
            self.task = 'classification'

    def embed(self, fp):
        x = fp

        for fc, bn in zip(self.fc, self.bn):
            x = fc(x)
            x = bn(x)
            x = F.relu(x)

        return x

    def forward(self, fp, need_emb=True, return_feature=False):
        if need_emb:
            x = self.embed(fp)
        else:
            x = fp
        emb = x
        x = self.out(x)
        if self.task == 'regression':
            if return_feature:
                return x, emb
            return x
        x = F.log_softmax(x, 1)

        if return_feature:
            return x, emb
        return x
    
class CliffPredictionModule(torch.nn.Module):
    def __init__(self, in_dims, hidden_dim=128):
        super().__init__()

        self.cliff_head = torch.nn.Sequential(
            torch.nn.Linear(in_dims, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)  # loss scalar 출력
        )

    def forward(self, h):
        # h_i shape: [B, 1, embed_dim]
        h_i = h.unsqueeze(1)
        # h_j shape: [1, B, embed_dim]
        h_j = h.unsqueeze(0)
        abs_diff = torch.abs(h_i - h_j)
        
        # 3-2. AC 예측 헤드 통과
        # pred_cliff_logits shape: [B, B, 1] ---squeeze---> [B, B]
        pred_cliff_logits = self.cliff_head(abs_diff).squeeze(-1)
        
        # 학습 시에는 두 예측 결과를 모두 반환
        return pred_cliff_logits
    
    def forward(self, h1, h2):
        """
        두 개의 임베딩 셋 (h1, h2)을 입력받아
        그 '차이'에 대한 Cliff 예측값을 반환합니다.
        
        - 학습 시: h1=h(B,dim), h2=h(B,dim) -> [B, B] 반환
        - 평가 시: h1=H_U(B,dim), h2=H_L(L,dim) -> [B, L] 반환
        """
        
        # 2. 차이 벡터 계산 로직 (내장)
        # h1_i shape: [B, 1, embed_dim]
        # h2_j shape: [1, L, embed_dim] (L은 B와 같을 수 있음)
        h1_i = h1.unsqueeze(1)
        h2_j = h2.unsqueeze(0)
        
        # abs_diff shape: [B, L, embed_dim]
        abs_diff = torch.abs(h1_i - h2_j)
        
        # 3. MLP 헤드 통과
        # pred_cliff_logits shape: [B, L, 1] ---squeeze---> [B, L]
        pred_cliff_logits = self.cliff_head(abs_diff).squeeze(-1)
        
        return pred_cliff_logits
    def forward_pairs(self, h1, h2):
        """
        이미 선택된 pair들에 대해서만 계산
        
        Args:
            h1: (num_pairs, embed_dim) - 첫 번째 요소들
            h2: (num_pairs, embed_dim) - 두 번째 요소들
        
        Returns:
            pred_cliff_logits: (num_pairs,) - 각 pair의 예측값
        """
        # 이미 대응되는 pair끼리만 계산 (broadcasting 없음!)
        abs_diff = torch.abs(h1 - h2)  # (num_pairs, embed_dim)
        
        # MLP 통과
        pred_cliff_logits = self.cliff_head(abs_diff).squeeze(-1)  # (num_pairs,)
        
        return pred_cliff_logits

class Model(torch.nn.Module):
    def __init__(self, architecture: str, hidden_dim=1024, lmda=0, classification = False, pretrain_file = '', **kwargs):
        super().__init__()
        assert architecture in ['mlp', 'graphMVP', 'gin']
        self.architecture = architecture
        n_hidden = 512
        if architecture == 'mlp':
            self.model = MLP(hidden_dim=hidden_dim, classification=classification, **kwargs)
        elif architecture == 'gin':
            self.model = GIN(hidden_dim=hidden_dim, classification=classification, need_pretrain=False, **kwargs)
        elif architecture == 'graphMVP':
            self.model = GIN(hidden_dim=hidden_dim, classification=classification, need_pretrain=True, **kwargs)
            self.model.pretrain.load_state_dict(torch.load(pretrain_file), strict=True)

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)
        self.classification = classification
        if self.classification:
            self.loss_fn = torch.nn.NLLLoss()
        else:
            self.loss_fn = torch.nn.MSELoss()
        self.lmda = lmda
        self.sim_threshold = 0.6
        self.anchored = True

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)
        self.cliff_prediction_module = CliffPredictionModule(in_dims=hidden_dim, hidden_dim=128).to(self.device)

        self.optimizer = torch.optim.Adam([
            # 그룹 1: 메인 모델의 파라미터
            {'params': self.model.parameters(), 
            'lr': self.model.lr, 
            'weight_decay': self.model.weight_decay},
            
            # 그룹 2: Cliff 모듈의 파라미터
            {'params': self.cliff_prediction_module.parameters(), 
            'lr': self.model.lr,  # (우선 동일한 lr로 시작)
            'weight_decay': self.model.weight_decay}
        ])

        # Save initial weights in the model for the anchored regularization and move them to the gpu
        if self.model.anchored:
            self.model.anchor_weights = deepcopy({i: j for i, j in self.model.named_parameters()})
            self.model.anchor_weights = {i: j.to(self.device) for i, j in self.model.anchor_weights.items()}

        self.train_loss = []
        self.epochs, self.epoch = self.model.epochs, 0

    def calculate_embedding_similarity_matrix(self, h):
        """
        GNN 임베딩(h) 간의 BxB 코사인 유사도 행렬을 계산합니다.
        
        입력:
        - h (Tensor): GNN 인코더의 출력 [B, embed_dim]
        """
        
        # 1. 임베딩 L2 정규화
        #    (코사인 유사도 = 정규화된 벡터의 내적)
        h_norm = F.normalize(h, p=2, dim=1)
        
        # 2. 행렬 곱(matmul)으로 코사인 유사도 행렬 계산
        #    (h_norm @ h_norm.T)
        #    shape: [B, embed_dim] @ [embed_dim, B] -> [B, B]
        sim_matrix = torch.matmul(h_norm, h_norm.t())
        
        # 수치적 안정성을 위해 값을 -1.0과 1.0 사이로 클리핑
        sim_matrix = torch.clamp(sim_matrix, -1.0, 1.0)
        
        return sim_matrix

    def calculate_fp_similarity_matrix(self, fp_batch_U, fp_batch_L):
        """
        (평가/AL용) Unlabeled 배치 [B, dim]와 Labeled 셋 [L, dim]를 입력받아,
        '두 셋 간의' Tanimoto 유사도 행렬 [B, L]를 반환합니다.
        """
        fp_U_float = fp_batch_U.float() # [B, dim]
        fp_L_float = fp_batch_L.float() # [L, dim]
        
        # [B, dim] @ [dim, L] -> [B, L]
        inter = torch.matmul(fp_U_float, fp_L_float.t())
        
        sums_U = fp_U_float.sum(dim=1).unsqueeze(1) # [B, 1]
        sums_L = fp_L_float.sum(dim=1).unsqueeze(0) # [1, L]
        
        # [B, L]
        union = sums_U + sums_L - inter
        
        eps = 1e-6
        sim_matrix = inter / (union + eps)
        
        return sim_matrix
    
    def train(self, dataloader: DataLoader, epochs: int = None, verbose: bool = True) -> None:
        # epochs = 70
        bar = trange(self.epochs if epochs is None else epochs, disable=not verbose)
        self.model.train()
        self.cliff_prediction_module.train()
        scaler = torch.cuda.amp.GradScaler()
        criterion_cliff = torch.nn.BCEWithLogitsLoss()
        
        for epoch in bar:
            running_loss = 0
            items = 0
            self.model.train()
            self.cliff_prediction_module.train()
            for idx, batch in enumerate(dataloader):

                self.optimizer.zero_grad()

                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    y_hat, embeddings = self.model(x, return_feature = True)

                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    if self.classification:
                        loss = self.loss_fn(y_hat, y.squeeze())
                    else:
                        loss = self.loss_fn(y_hat.squeeze(), y.squeeze())

                    if self.anchored:   # Calculate the total anchored L2 loss
                        l2_loss = 0
                        for param_name, params in self.model.named_parameters():
                            anchored_param = self.model.anchor_weights[param_name]
                            l2_loss += (self.model.l2_lambda / len(y)) * torch.mul(params - anchored_param, params - anchored_param).sum()

                        loss = loss + l2_loss   # Add anchored loss to regular loss according to Pearce et al. (2018)

                    

                    if self.architecture == 'mlp':
                        true_sim = self.calculate_fp_similarity_matrix(x, x)
                    else:
                        true_sim = self.calculate_fp_similarity_matrix(x.fp, x.fp)
                    true_activity = y.view(-1, 1)
                    true_act_i = true_activity.unsqueeze(1)
                    true_act_j = true_activity.unsqueeze(0)
                    true_delta_act = torch.abs(true_act_i - true_act_j)
                    true_delta_act = true_delta_act.squeeze(-1)

                    CLIFF_ACT_THRESHOLD = 1.0

                    true_cliff_labels = (true_sim > self.sim_threshold) & \
                                        (true_delta_act >= CLIFF_ACT_THRESHOLD)
                    true_cliff_labels = true_cliff_labels.float()

                    mask = (true_sim > self.sim_threshold)

                    if mask.sum() == 0:
                        loss_cliff = torch.tensor(0.0).to(self.device)
                    else:
                        # 개선: 필요한 pair만 추출
                        i_indices, j_indices = torch.nonzero(mask, as_tuple=True)
                        
                        # 필요한 embedding만 선택
                        emb_i = embeddings[i_indices]  # (num_pairs, embed_dim)
                        emb_j = embeddings[j_indices]  # (num_pairs, embed_dim)
                        
                        # forward_pairs로 필요한 것만 계산
                        y_hat_masked = self.cliff_prediction_module.forward_pairs(emb_i, emb_j)
                        
                        # 정답 레이블도 필요한 것만 추출
                        labels_masked = true_cliff_labels[mask]
                        
                        num_positives = labels_masked.sum()
                        
                        if num_positives == 0:
                            criterion_cliff = torch.nn.BCEWithLogitsLoss()
                        else:
                            num_negatives = labels_masked.numel() - num_positives
                            pos_weight_value = num_negatives / num_positives
                            
                            criterion_cliff = torch.nn.BCEWithLogitsLoss(
                                pos_weight=pos_weight_value.to(self.device)
                            )

                        loss_cliff = criterion_cliff(y_hat_masked, labels_masked)
                    lamda = 0.01

                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    loss = loss + lamda*loss_cliff
                        
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    running_loss += loss.item()
                    items += len(y)

            epoch_loss = running_loss / items
            bar.set_postfix(loss=f'{epoch_loss:.4f}')
            self.train_loss.append(epoch_loss)

            self.epoch += 1

    def predict(self, dataloader, train_dataloader) -> Tensor:
        y_hats = torch.tensor([]).to(self.device)
        
        train_fp_list = []
        train_embeddings_list = []
        train_y_true_list = []
        self.model.eval()
        self.cliff_prediction_module.eval()
        all_final_scores = []
        
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                # Train set의 fingerprint, embedding, label 수집
                for batch in train_dataloader:
                    *xs, y = [t.to(self.device) for t in batch]
                    y_hat, embeddings = self.model(*xs, return_feature=True)
                    
                    if self.architecture == 'mlp':
                        train_fp_list.append(xs[0])
                    else:
                        train_fp_list.append(xs[0].fp)
                    train_embeddings_list.append(embeddings)
                    train_y_true_list.append(y.squeeze())
                
                train_fp = torch.cat(train_fp_list, dim=0).to(self.device)
                train_embeddings = torch.cat(train_embeddings_list, dim=0).to(self.device)
                train_y_true = torch.cat(train_y_true_list, dim=0).to(self.device)

                for batch in dataloader:
                    *xs, y = [t.to(self.device) for t in batch]
                    y_hat, embeddings = self.model(*xs, return_feature=True)
                    
                    # ✅ Similarity matrix 먼저 계산
                    if self.architecture == 'mlp':
                        sim_matrix = self.calculate_fp_similarity_matrix(xs[0], train_fp)
                    else:
                        sim_matrix = self.calculate_fp_similarity_matrix(xs[0].fp, train_fp)
                    mask = (sim_matrix > self.sim_threshold)  # (batch_size, train_size)
                    
                    # 유사한 pair가 하나도 없는 경우 처리
                    if mask.sum() == 0:
                        # 모든 샘플에 대해 0.0 점수 할당
                        final_scores_batch = torch.zeros(embeddings.shape[0]).to(self.device)
                    else:
                        # ✅ 필요한 pair만 추출
                        i_indices, j_indices = torch.nonzero(mask, as_tuple=True)
                        
                        # 필요한 embedding만 선택
                        emb_i = embeddings[i_indices]  # (num_pairs, embed_dim)
                        emb_j = train_embeddings[j_indices]  # (num_pairs, embed_dim)
                        
                        # forward_pairs로 필요한 것만 계산
                        y_hat_cliff_pairs = self.cliff_prediction_module.forward_pairs(emb_i, emb_j)
                        cliff_probs_pairs = torch.sigmoid(y_hat_cliff_pairs)  # (num_pairs,)
                        
                        # train_y_true도 필요한 것만 선택
                        train_y_true_pairs = train_y_true[j_indices]  # (num_pairs,)
                        
                        # Conditional scores 계산 (필요한 pair만)
                        conditional_scores_pairs = torch.where(
                            train_y_true_pairs == 1,
                            1.0 - cliff_probs_pairs,
                            cliff_probs_pairs
                        )
                        
                        # ✅ Sparse matrix로 다시 재구성
                        batch_size = embeddings.shape[0]
                        train_size = train_embeddings.shape[0]
                        
                        # 전체 크기의 sparse tensor 생성 (메모리 효율적)
                        conditional_scores_full = torch.full(
                            (batch_size, train_size), 
                            float('nan'),
                            device=self.device,
                            dtype=torch.bfloat16  # ✅ 명시적으로 지정
                        )
                        conditional_scores_full[i_indices, j_indices] = conditional_scores_pairs
                        
                        # Max 또는 Mean aggregation
                        mx = False
                        if mx:
                            # Max 방식
                            conditional_scores_full[~mask] = -float('inf')
                            final_scores_batch, _ = torch.max(conditional_scores_full, dim=1)
                            final_scores_batch[torch.isinf(final_scores_batch)] = 0.0
                        else:
                            # Mean 방식 (masked)
                            final_scores_batch = torch.nanmean(conditional_scores_full, dim=1)
                            final_scores_batch = torch.nan_to_num(final_scores_batch, nan=0.0)
                    
                    all_final_scores.append(final_scores_batch)

                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)
                
                final_scores_tensor = torch.cat(all_final_scores, dim=0)

        return y_hats, final_scores_tensor

class Ensemble_Model(torch.nn.Module):
    """ Ensemble of GCNs"""
    def __init__(self, ensemble_size: int = 1, seed: int = 0, architecture: str = 'mlp', assay_active = None, pretrain_file = '', **kwargs) -> None:
        self.ensemble_size = ensemble_size
        self.architecture = architecture
        self.seed = seed
        rng = np.random.default_rng(seed=seed)
        self.seeds = rng.integers(0, 1000, 10)
        classification = assay_active is not None
        self.models = {0: Model(seed=self.seeds[0], architecture=architecture, classification = classification, pretrain_file=pretrain_file, **kwargs)}

    def optimize_hyperparameters(self, x, y: DataLoader, **kwargs):
        # raise NotImplementedError
        best_hypers = optimize_hyperparameters(x, y, architecture=self.architecture, **kwargs)
        # # re-init model wrapper with optimal hyperparameters
        self.__init__(ensemble_size=self.ensemble_size, seed=self.seed, **best_hypers)

    def train(self, dataloader: DataLoader) -> None:
        for i, m in self.models.items():
            m.train(dataloader)

    def predict(self, dataloader, train_dataloader) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        results_list = []
        for m in self.models.values():
            # result_tuple = (y_hats_model_i, scores_model_i)
            result_tuple = m.predict(dataloader, train_dataloader)
            results_list.append(result_tuple)

        y_hats_list, scores_list = zip(*results_list)

        ensemble_y_hats = torch.stack(y_hats_list, dim=1)
        ensemble_scores = torch.stack(scores_list, dim=1)

        return ensemble_y_hats, ensemble_scores

    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self) -> str:
        return f"Ensemble of {self.ensemble_size} Classifiers"
    