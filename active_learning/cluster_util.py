from sklearn.cluster import KMeans
import torch
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from sklearn.cluster import DBSCAN
from rdkit.DataStructs import BulkTanimotoSimilarity

def analyze_kmeans_cluster(dataset, x_screen, y_screen, smiles_screen, x_train, y_train):
    cluster_result = pd.DataFrame(columns = ['n_clusters', 'cluster_id', 'positive', 'negative', 'positive_ratio', 'negative_ratio'])
    pick_cluster_result = pd.DataFrame(columns = ['n_clusters', 'cluster_id', 'positive', 'negative', 'positive_ratio', 'negative_ratio','pick_pos','pick_neg', 'initial_pos', 'initial_neg'])
    smiles = pd.read_csv(f'./results_0708/result_{dataset}_exploitation_mlp/0/picked.csv',index_col=0)
    pick_smiles = list(smiles.loc[0])
    pick_y = np.array(smiles.loc[1])
    pick_idx = np.array([0 for _ in range(1000)])
    xx = 0
    for idx, smi in enumerate(pick_smiles):
        for idx2, smi_screen in enumerate(smiles_screen):
            if smi == smi_screen:
                xx += 1
                pick_idx[idx] = idx2
                break
    print(xx)
    for n_clusters in range(2, 65):
        print(n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(x_screen)
        y_kmeans = kmeans.predict(x_screen)
        y_kmeans_initial = kmeans.predict(x_train)
        pick_kmeans = y_kmeans[pick_idx]

        for cluster_id in range(n_clusters):
            y_cluster = y_screen[y_kmeans == cluster_id]
            if len(y_cluster) == 0:
                cluster_result.loc[len(cluster_result.index)] = [n_clusters, cluster_id, 0, 0, 0, 0]
                pick_cluster_result.loc[len(pick_cluster_result.index)] = [n_clusters, cluster_id, 0, 0, 0, 0, 0, 0, 0, 0]
            pos_cluster = torch.sum(y_cluster).item()
            neg_cluster = len(y_cluster) - pos_cluster
            cluster_result.loc[len(cluster_result.index)] = [n_clusters, cluster_id, pos_cluster, neg_cluster, 
                                                             pos_cluster / len(y_cluster), neg_cluster / len(y_cluster)]
            # print(pick_kmeans)
            # print(pick_idx)
            # print(pick_y)
            pick_cluster = pick_y[pick_kmeans == cluster_id]
            # print(pick_cluster)
            y_train_cluster = y_train[y_kmeans_initial == cluster_id]
            pos_train_cluster = torch.sum(y_train_cluster).item()
            neg_train_cluster = len(y_train_cluster) - pos_train_cluster
            pick_cluster_result.loc[len(pick_cluster_result.index)] = [n_clusters, cluster_id, pos_cluster, neg_cluster, 
                                                             pos_cluster / len(y_cluster), neg_cluster / len(y_cluster), (pick_cluster == '1').sum(), (pick_cluster == '0').sum(),
                                                             pos_train_cluster, neg_train_cluster]

        visual_tools = {'pca':PCA(n_components=2), 'umap':UMAP(n_components=2, random_state=0)}
        # for visual in visual_tools:
        #     X_visual = visual_tools[visual].fit_transform(x_screen)
        #     plt.figure(figsize=(8,6))
        #     plt.scatter(X_visual[:, 0], X_visual[:, 1], c=y_kmeans, s=5, alpha=0.1, cmap='viridis')
        #     # centers = visual_tools[visual].transform(kmeans.cluster_centers_)
        #     # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=25, alpha=0.75, marker='X', label='Centroids')
        #     plt.title('KMeans Clustering')
        #     plt.legend()
        #     os.makedirs(f'./cluster_result/KMeans/{dataset}/{visual}', exist_ok=True)
        #     plt.savefig(f'./cluster_result/KMeans/{dataset}/{visual}/{visual}_cluster{n_clusters}.png')
        #     plt.close()
        cluster_result = cluster_result.round(5)
        cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative']] = cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative']].astype(int)
        cluster_result.to_csv(f'./cluster_result/KMeans/{dataset}/cluster_result.csv', index=False)
        pick_cluster_result = pick_cluster_result.round(5)
        pick_cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative','pick_pos','pick_neg', 'initial_pos', 'initial_neg']] = pick_cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative','pick_pos','pick_neg', 'initial_pos', 'initial_neg']].astype(int)
        pick_cluster_result.to_csv(f'./cluster_result/KMeans/{dataset}/pick_cluster_result.csv', index=False)

# from multiprocessing import Pool

# def compute_chunk(start_end):
#     start, end = start_end
#     return pairwise_distances(x_screen[start:end], x_screen, metric='jaccard')
def analyze_DBSCAN_cluster(dataset, x_screen, y_screen, smiles_screen, x_train, y_train):
    cluster_result = pd.DataFrame(columns = ['n_clusters', 'cluster_id', 'positive', 'negative', 'positive_ratio', 'negative_ratio'])
    print('start')
    
    pick_cluster_result = pd.DataFrame(columns = ['n_clusters', 'cluster_id', 'positive', 'negative', 'positive_ratio', 'negative_ratio','pick_pos','pick_neg', 'initial_pos', 'initial_neg'])
    smiles = pd.read_csv(f'./results_0708/result_{dataset}_exploitation_mlp/0/picked.csv',index_col=0)
    pick_smiles = list(smiles.loc[0])
    pick_y = np.array(smiles.loc[1])
    pick_idx = np.array([0 for _ in range(1000)])
    xx = 0
    for idx, smi in enumerate(pick_smiles):
        for idx2, smi_screen in enumerate(smiles_screen):
            if smi == smi_screen:
                xx += 1
                pick_idx[idx] = idx2
                break
    print(xx)

    save_path = f"./cluster_result/dist_matrix_{dataset}.npy"
    if os.path.exists(save_path):
        dist_mat = np.load(save_path)
    else:
        from rdkit.DataStructs.cDataStructs import ExplicitBitVect
        rdkit_fps = []
        for row in x_screen:
            bv = ExplicitBitVect(len(row))
            onbits = np.where(row == 1)[0]
            for bit in onbits:
                bv.SetBit(int(bit))
            rdkit_fps.append(bv)
        dist_mat = np.eye(len(x_screen), dtype=np.float32)
        # dist_mat = pairwise_distances(x_screen, metric='jaccard')
        for mol_i in range(len(rdkit_fps)):
            dist = BulkTanimotoSimilarity(rdkit_fps[mol_i], rdkit_fps[mol_i+1:])
            dist_mat[mol_i, mol_i+1:] = dist
            dist_mat[mol_i+1:, mol_i] = dist
            if mol_i%10000 == 0:
                print(mol_i)
        dist_mat = 1 - dist_mat
        np.save(save_path, dist_mat.astype(np.float32))
    # chunk_size = 1000Wx``
    # num_workers = 10
    # indices = [(i, min(i + chunk_size, len(x_screen))) for i in range(0, len(x_screen), chunk_size)]

    # with Pool(num_workers) as pool:
    #     results = pool.map(compute_chunk, indices)

    # dist_mat = np.vstack(results)
    print('finish')
    dbscan = DBSCAN(eps=0.4, min_samples=5, metric='precomputed')
    y_dbscan = dbscan.fit_predict(dist_mat)
    clusters = set(y_dbscan)
    for cluster_id in sorted(clusters):
        y_cluster = y_screen[y_dbscan == cluster_id]
        if len(y_cluster) == 0:
            cluster_result.loc[len(cluster_result.index)] = [len(clusters), cluster_id, 0, 0, 0, 0]
        pos_cluster = torch.sum(y_cluster).item()
        neg_cluster = len(y_cluster) - pos_cluster
        cluster_result.loc[len(cluster_result.index)] = [len(clusters), cluster_id, pos_cluster, neg_cluster, 
                                                            pos_cluster / len(y_cluster), neg_cluster / len(y_cluster)]
    visual_tools = {'pca':PCA(n_components=2), 'umap':UMAP(n_components=2, random_state=0)}
    # for visual in visual_tools:
    #     X_visual = visual_tools[visual].fit_transform(x_screen)
    #     plt.figure(figsize=(8,6))
    #     plt.scatter(X_visual[:, 0], X_visual[:, 1], c=y_dbscan, s=5, alpha=0.1, cmap='viridis')
        # centers = visual_tools[visual].transform(kmeans.cluster_centers_)
        # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=25, alpha=0.75, marker='X', label='Centroids')
        # plt.title('DBSCAN Clustering')
        # plt.legend()
        # os.makedirs(f'./cluster_result/DBSCAN/{dataset}/{visual}', exist_ok=True)
        # plt.savefig(f'./cluster_result/DBSCAN/{dataset}/{visual}/{visual}_cluster{len(clusters)}.png')
        # plt.close()
    cluster_result = cluster_result.round(5)
    cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative']] = cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative']].astype(int)
    cluster_result.to_csv(f'./cluster_result/DBSCAN/{dataset}/cluster_result.csv', index=False)

    cluster_result = pd.DataFrame(columns = ['n_clusters', 'cluster_id', 'positive', 'negative', 'positive_ratio', 'negative_ratio'])
    pick_cluster_result = pd.DataFrame(columns = ['n_clusters', 'cluster_id', 'positive', 'negative', 'positive_ratio', 'negative_ratio','pick_pos','pick_neg', 'initial_pos', 'initial_neg'])
    dist_mat_train = pairwise_distances(x_train, x_screen, metric='jaccard')
    for n_clusters in range(2, 65):
        print(n_clusters)
        kmeans = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=0)
        kmeans.fit(dist_mat)
        y_kmeans = kmeans.predict(dist_mat)
        y_kmeans_initial = kmeans.predict(dist_mat_train)
        pick_kmeans = y_kmeans[pick_idx]

        for cluster_id in range(n_clusters):
            y_cluster = y_screen[y_kmeans == cluster_id]
            if len(y_cluster) == 0:
                cluster_result.loc[len(cluster_result.index)] = [n_clusters, cluster_id, 0, 0, 0, 0]
                pick_cluster_result.loc[len(pick_cluster_result.index)] = [n_clusters, cluster_id, 0, 0, 0, 0, 0, 0, 0, 0]
            else:
                pos_cluster = torch.sum(y_cluster).item()
                neg_cluster = len(y_cluster) - pos_cluster
                cluster_result.loc[len(cluster_result.index)] = [n_clusters, cluster_id, pos_cluster, neg_cluster, 
                                                             pos_cluster / len(y_cluster), neg_cluster / len(y_cluster)]
                pick_cluster = pick_y[pick_kmeans == cluster_id]
                # print(pick_cluster)
                y_train_cluster = y_train[y_kmeans_initial == cluster_id]
                pos_train_cluster = torch.sum(y_train_cluster).item()
                neg_train_cluster = len(y_train_cluster) - pos_train_cluster
                pick_cluster_result.loc[len(pick_cluster_result.index)] = [n_clusters, cluster_id, pos_cluster, neg_cluster, 
                                                                pos_cluster / len(y_cluster), neg_cluster / len(y_cluster), (pick_cluster == '1').sum(), (pick_cluster == '0').sum(),
                                                                pos_train_cluster, neg_train_cluster]

        cluster_result = cluster_result.round(5)
        cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative']] = cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative']].astype(int)
        cluster_result.to_csv(f'./cluster_result/KMedoids/{dataset}/cluster_result.csv', index=False)
        pick_cluster_result = pick_cluster_result.round(5)
        pick_cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative','pick_pos','pick_neg', 'initial_pos', 'initial_neg']] = pick_cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative','pick_pos','pick_neg', 'initial_pos', 'initial_neg']].astype(int)
        pick_cluster_result.to_csv(f'./cluster_result/KMedoids/{dataset}/pick_cluster_result.csv', index=False)

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances

def analyze_KMedoids_cluster(dataset, x_screen, y_screen):
    dist_mat = pairwise_distances(x_screen, metric="jaccard").astype('float32')
    cluster_result = pd.DataFrame(columns = ['n_clusters', 'cluster_id', 'positive', 'negative', 'positive_ratio', 'negative_ratio'])
    for n_clusters in range(2, 65):
        print(n_clusters)
        kmeans = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=0)
        y_kmeans = kmeans.fit_predict(dist_mat)

        for cluster_id in range(n_clusters):
            y_cluster = y_screen[y_kmeans == cluster_id]
            if len(y_cluster) == 0:
                cluster_result.loc[len(cluster_result.index)] = [n_clusters, cluster_id, 0, 0, 0, 0]
            pos_cluster = torch.sum(y_cluster).item()
            neg_cluster = len(y_cluster) - pos_cluster
            cluster_result.loc[len(cluster_result.index)] = [n_clusters, cluster_id, pos_cluster, neg_cluster, 
                                                             pos_cluster / len(y_cluster), neg_cluster / len(y_cluster)]

        visual_tools = {'pca':PCA(n_components=2), 'umap':UMAP(n_components=2, random_state=0)}
        for visual in visual_tools:
            X_visual = visual_tools[visual].fit_transform(x_screen)
            plt.figure(figsize=(8,6))
            plt.scatter(X_visual[:, 0], X_visual[:, 1], c=y_kmeans, s=5, alpha=0.1, cmap='viridis')
            # centers = visual_tools[visual].transform(kmeans.cluster_centers_)
            # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=25, alpha=0.75, marker='X', label='Centroids')
            plt.title('KMeans Clustering')
            plt.legend()
            os.makedirs(f'./cluster_result/KMedoids/{dataset}/{visual}', exist_ok=True)
            plt.savefig(f'./cluster_result/KMedoids/{dataset}/{visual}/{visual}_cluster{n_clusters}.png')
            plt.close()
        cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative']] = cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative']].astype(int)
        cluster_result.round(5).to_csv(f'./cluster_result/KMedoids/{dataset}/cluster_result.csv', index=False)

import hdbscan
def analyze_HDBSCAN_cluster(dataset, x_screen, y_screen):
    cluster_result = pd.DataFrame(columns = ['n_clusters', 'cluster_id', 'positive', 'negative', 'positive_ratio', 'negative_ratio'])
    dbscan = hdbscan.HDBSCAN(metric='jaccard', min_cluster_size=2, min_samples=5, cluster_selection_method='eom')
    y_dbscan = dbscan.fit_predict(x_screen)
    clusters = set(y_dbscan)
    for cluster_id in sorted(clusters):
        y_cluster = y_screen[y_dbscan == cluster_id]
        if len(y_cluster) == 0:
            cluster_result.loc[len(cluster_result.index)] = [len(clusters), cluster_id, 0, 0, 0, 0]
        pos_cluster = torch.sum(y_cluster).item()
        neg_cluster = len(y_cluster) - pos_cluster
        cluster_result.loc[len(cluster_result.index)] = [len(clusters), cluster_id, pos_cluster, neg_cluster, 
                                                            pos_cluster / len(y_cluster), neg_cluster / len(y_cluster)]
    visual_tools = {'pca':PCA(n_components=2), 'umap':UMAP(n_components=2, random_state=0)}
    for visual in visual_tools:
        X_visual = visual_tools[visual].fit_transform(x_screen)
        plt.figure(figsize=(8,6))
        plt.scatter(X_visual[:, 0], X_visual[:, 1], c=y_dbscan, s=5, alpha=0.1, cmap='viridis')
        plt.title('hdbscan Clustering')
        plt.legend()
        os.makedirs(f'./cluster_result/DBSCAN/{dataset}/{visual}', exist_ok=True)
        plt.savefig(f'./cluster_result/DBSCAN/{dataset}/{visual}/{visual}_cluster{len(clusters)}.png')
        plt.close()
    cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative']] = cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative']].astype(int)
    cluster_result.round(5).to_csv(f'./cluster_result/DBSCAN/{dataset}/cluster_result.csv', index=False)

from sklearn.preprocessing import StandardScaler

def analyze_kmeans_cluster_cliff(dataset, x_screen, y_screen, smiles_screen, x_train, y_train):
    # scaler = StandardScaler()
    # scaler.fit(x_screen)
    # x_screen = scaler.transform(x_screen)
    # x_train = scaler.transform(x_train)
    cluster_result = pd.DataFrame(columns = ['n_clusters', 'cluster_id', 'positive', 'negative', 'positive_ratio', 'negative_ratio'])
    pick_cluster_result = pd.DataFrame(columns = ['n_clusters', 'cluster_id', 'positive', 'negative', 'positive_ratio', 'negative_ratio','pick_pos','pick_neg', 'initial_pos', 'initial_neg'])
    smiles = pd.read_csv(f'./results_0708/result_{dataset}_exploitation_mlp/0/picked.csv',index_col=0)
    pick_smiles = list(smiles.loc[0])
    pick_y = np.array(smiles.loc[1])
    pick_idx = np.array([0 for _ in range(1000)])
    xx = 0
    for idx, smi in enumerate(pick_smiles):
        for idx2, smi_screen in enumerate(smiles_screen):
            if smi == smi_screen:
                xx += 1
                pick_idx[idx] = idx2
                break
    print(xx)
    for n_clusters in range(2, 65):
        print(n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(x_screen)
        y_kmeans = kmeans.predict(x_screen)
        y_kmeans_initial = kmeans.predict(x_train)
        pick_kmeans = y_kmeans[pick_idx]

        for cluster_id in range(n_clusters):
            y_cluster = y_screen[y_kmeans == cluster_id]
            if len(y_cluster) == 0:
                cluster_result.loc[len(cluster_result.index)] = [n_clusters, cluster_id, 0, 0, 0, 0]
                pick_cluster_result.loc[len(pick_cluster_result.index)] = [n_clusters, cluster_id, 0, 0, 0, 0, 0, 0, 0, 0]
            pos_cluster = torch.sum(y_cluster).item()
            neg_cluster = len(y_cluster) - pos_cluster
            cluster_result.loc[len(cluster_result.index)] = [n_clusters, cluster_id, pos_cluster, neg_cluster, 
                                                             pos_cluster / len(y_cluster), neg_cluster / len(y_cluster)]
            # print(pick_kmeans)
            # print(pick_idx)
            # print(pick_y)
            pick_cluster = pick_y[pick_kmeans == cluster_id]
            # print(pick_cluster)
            y_train_cluster = y_train[y_kmeans_initial == cluster_id]
            pos_train_cluster = torch.sum(y_train_cluster).item()
            neg_train_cluster = len(y_train_cluster) - pos_train_cluster
            pick_cluster_result.loc[len(pick_cluster_result.index)] = [n_clusters, cluster_id, pos_cluster, neg_cluster, 
                                                             pos_cluster / len(y_cluster), neg_cluster / len(y_cluster), (pick_cluster == '1').sum(), (pick_cluster == '0').sum(),
                                                             pos_train_cluster, neg_train_cluster]

        visual_tools = {'pca':PCA(n_components=2), 'umap':UMAP(n_components=2, random_state=0)}
        # for visual in visual_tools:
        #     X_visual = visual_tools[visual].fit_transform(x_screen)
        #     plt.figure(figsize=(8,6))
        #     plt.scatter(X_visual[:, 0], X_visual[:, 1], c=y_kmeans, s=5, alpha=0.1, cmap='viridis')
        #     # centers = visual_tools[visual].transform(kmeans.cluster_centers_)
        #     # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=25, alpha=0.75, marker='X', label='Centroids')
        #     plt.title('KMeans Clustering')
        #     plt.legend()
        #     os.makedirs(f'./cluster_result/KMeans/{dataset}/{visual}', exist_ok=True)
        #     plt.savefig(f'./cluster_result/KMeans/{dataset}/{visual}/{visual}_cluster{n_clusters}.png')
        #     plt.close()
        cluster_result = cluster_result.round(5)
        cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative']] = cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative']].astype(int)
        cluster_result.to_csv(f'./cluster_result/KMeans_cliff/{dataset}_noscale/cluster_result.csv', index=False)
        pick_cluster_result = pick_cluster_result.round(5)
        pick_cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative','pick_pos','pick_neg', 'initial_pos', 'initial_neg']] = pick_cluster_result[['n_clusters', 'cluster_id', 'positive', 'negative','pick_pos','pick_neg', 'initial_pos', 'initial_neg']].astype(int)
        pick_cluster_result.to_csv(f'./cluster_result/KMeans_cliff/{dataset}_noscale/pick_cluster_result.csv', index=False)
