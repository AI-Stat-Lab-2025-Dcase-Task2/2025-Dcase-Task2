import numpy as np
import pandas as pd
from scipy.stats import hmean
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

def compute_result(result_train, result_test, test=False, sampling_strategy=None):
    # Data load
    eval_train = pd.read_csv('/home/user/hyunjun/asd/data/dev_train.csv')
    
    if not test:
        eval_test = pd.read_csv('/home/user/hyunjun/asd/data/dev_test.csv')
        eval_test['label'] = eval_test['label'].map({'normal': 0, 'anomaly': 1})
    else:
        eval_test = pd.read_csv('./data/eval_data.csv')

    # Convert CUDA tensor to numpy if necessary
    result_train_np = result_train.cpu().numpy() if hasattr(result_train, 'cpu') else result_train
    result_test_np = result_test.cpu().numpy() if hasattr(result_test, 'cpu') else result_test

    # Normalize result_train per machine and store statistics (computed once)
    machine_stats = {}
    result_train_normalized = np.zeros_like(result_train_np)
    for machine in eval_train['machine'].unique():
        train_indices = np.array(eval_train[eval_train['machine'] == machine].index)
        machine_train = result_train_np[train_indices]
        mean = machine_train.mean(axis=0)
        std = machine_train.std(axis=0) + 1e-8  # Add small epsilon to avoid division by zero
        machine_stats[machine] = {'mean': mean, 'std': std}
        result_train_normalized[train_indices] = (machine_train - mean) / std

    if test:
        # SMOTE on normalized result_train
        result_train_whole = []
        domain_train_whole = []
        machine_train_indices = {}
        start_idx = 0
        for machine in eval_train['machine'].unique():
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            train_indices = np.array(eval_train[eval_train['machine'] == machine].index)
            result_train_res, domain_train = smote.fit_resample(
                result_train_normalized[train_indices],
                eval_train[eval_train['machine'] == machine]['domain'].values
            )
            result_train_whole.append(result_train_res)
            domain_train_whole.append(domain_train)
            smote_samples = result_train_res.shape[0]
            machine_train_indices[machine] = np.arange(start_idx, start_idx + smote_samples)
            start_idx += smote_samples

        result_train_whole = np.concatenate(result_train_whole, axis=0)
        domain_train_whole = np.concatenate(domain_train_whole, axis=0)

        # KNN per machine
        test_scores = np.zeros(len(result_test_np))
        knn_models = {}
        for machine in eval_train['machine'].unique():
            knn = NearestNeighbors(n_neighbors=1, metric='cosine')
            knn.fit(result_train_whole[machine_train_indices[machine]])
            knn_models[machine] = knn

        # Compute anomaly scores for result_test
        for i in range(len(result_test_np)):
            machine = eval_test['machine'].iloc[i]
            test_embedding = (result_test_np[i] - machine_stats[machine]['mean']) / machine_stats[machine]['std']
            distance, _ = knn_models[machine].kneighbors([test_embedding])
            test_scores[i] = distance[0, 0]

        return test_scores
    else:
        # Iterate over sampling strategies from 0.1 to 1.0
        best_score = -float('inf')
        best_results = None
        best_sampling_strategy = None

        for sampling_strategy in np.arange(0.1, 1.1, 0.1):
            # SMOTE on normalized result_train
            result_train_whole = []
            domain_train_whole = []
            machine_train_indices = {}
            start_idx = 0
            for machine in eval_train['machine'].unique():
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                train_indices = np.array(eval_train[eval_train['machine'] == machine].index)
                result_train_res, domain_train = smote.fit_resample(
                    result_train_normalized[train_indices],
                    eval_train[eval_train['machine'] == machine]['domain'].values
                )
                result_train_whole.append(result_train_res)
                domain_train_whole.append(domain_train)
                smote_samples = result_train_res.shape[0]
                machine_train_indices[machine] = np.arange(start_idx, start_idx + smote_samples)
                start_idx += smote_samples

            result_train_whole = np.concatenate(result_train_whole, axis=0)
            domain_train_whole = np.concatenate(domain_train_whole, axis=0)

            # KNN per machine
            test_scores = np.zeros(len(result_test_np))
            knn_models = {}
            for machine in eval_train['machine'].unique():
                knn = NearestNeighbors(n_neighbors=1, metric='cosine')
                knn.fit(result_train_whole[machine_train_indices[machine]])
                knn_models[machine] = knn

            # Compute anomaly scores for result_test
            for i in range(len(result_test_np)):
                machine = eval_test['machine'].iloc[i]
                test_embedding = (result_test_np[i] - machine_stats[machine]['mean']) / machine_stats[machine]['std']
                distance, _ = knn_models[machine].kneighbors([test_embedding])
                test_scores[i] = distance[0, 0]

            # Compute metrics
            eval_test['anomaly_score'] = test_scores
            p_aucs = []
            aucs_source = []
            aucs_target = []
            machine_results = {}

            machine_list = eval_test['machine'].unique()
            for machine in machine_list:
                machine_results[machine] = []
                # pAUC
                temp = eval_test[eval_test['machine'] == machine]
                true = temp['label'].values
                cos = temp['anomaly_score'].values
                p_auc = roc_auc_score(true, cos, max_fpr=0.1)
                p_aucs.append(p_auc)
                machine_results[machine].append(p_auc)
                
                # AUCSource
                temp_source = temp[temp['domain'] == 'source']
                true_source = temp_source['label'].values
                cos_source = temp_source['anomaly_score'].values
                auc_score = roc_auc_score(true_source, cos_source)
                aucs_source.append(auc_score)
                machine_results[machine].append(auc_score)
                
                # AUCTarget
                temp_target = temp[temp['domain'] == 'target']
                true_target = temp_target['label'].values
                cos_target = temp_target['anomaly_score'].values
                auc_score = roc_auc_score(true_target, cos_target)
                aucs_target.append(auc_score)
                machine_results[machine].append(auc_score)

            mean_p_auc = hmean(p_aucs)
            mean_auc_source = hmean(aucs_source)
            mean_auc_target = hmean(aucs_target)
            score = hmean(aucs_source + aucs_target + p_aucs)

            # Update best results if current score is higher
            if score > best_score:
                best_score = score
                best_sampling_strategy = sampling_strategy
                best_results = (machine_results, mean_auc_source, mean_auc_target, mean_p_auc, score)

        # Return the results for the best sampling strategy, including the strategy value
        return best_results + (best_sampling_strategy,)