"""
discriminability_multi_emo.py: Code for computing multi-class emotion discriminability.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def prepare_multiclass_data(
    embeddings_dict: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    emotions: List[str],
    operation: str,
    layer: int,
    return_counts: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], Dict[str, int]]:
    """
    Prepares data for multi-class classification for a specific operation and layer.
    
    Args:
        embeddings_dict: {emotion: {operation: {layer: tensor}}}
        emotions: List of emotions to include
        operation: Operation name
        layer: Layer index
        return_counts: If True, also return per-emotion sample counts
        
    Returns:
        X: Feature matrix [N_total, D]
        y: Label vector [N_total]
        label_map: Dict mapping label index to emotion name
        emotion_counts: Dict mapping emotion name to sample count (if return_counts=True)
    """
    X_list = []
    y_list = []
    label_map = {}
    emotion_counts = {}
    
    for idx, emotion in enumerate(emotions):
        label_map[idx] = emotion
        
        if emotion not in embeddings_dict:
            emotion_counts[emotion] = 0
            continue
        if operation not in embeddings_dict[emotion]:
            emotion_counts[emotion] = 0
            continue
        if layer not in embeddings_dict[emotion][operation]:
            emotion_counts[emotion] = 0
            continue
            
        emb = embeddings_dict[emotion][operation][layer]
        # Ensure numpy
        emb_np = emb.float().cpu().numpy()
        
        X_list.append(emb_np)
        y_list.append(np.full(len(emb_np), idx))
        emotion_counts[emotion] = len(emb_np)
        
    if not X_list:
        if return_counts:
            return np.array([]), np.array([]), label_map, emotion_counts
        return np.array([]), np.array([]), label_map
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    if return_counts:
        return X, y, label_map, emotion_counts
    return X, y, label_map


def train_multiclass_classifier(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    method: str = 'centroid'
) -> Any:
    """
    Trains a multi-class classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        method: 'centroid' or 'linear'
        
    Returns:
        Trained model
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    if method == 'centroid':
        clf = NearestCentroid()
        clf.fit(X_train, y_train)
        return clf
        
    elif method == 'linear':
        # Logistic Regression for multi-class
        # 'multinomial' is standard for multi-class logistic regression
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                multi_class='multinomial', 
                solver='lbfgs', 
                max_iter=2000, 
                random_state=42
            )
        )
        clf.fit(X_train, y_train)
        return clf
        
    else:
        raise ValueError(f"Unknown classification method: {method}")


def compute_multiclass_discriminability(
    train_split: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    test_split: Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    emotions: List[str],
    operations: List[str],
    classifier_type: str = 'centroid',
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Computes multi-class discriminability for specified operations and layers.
    
    Args:
        train_split: Training embeddings {emotion -> op -> layer -> tensor}
        test_split: Test embeddings {emotion -> op -> layer -> tensor}
        emotions: List of emotions
        operations: List of operations
        classifier_type: 'centroid' or 'linear'
        
    Returns:
        results: {op: {layer: {'accuracy': float, 'confusion_matrix': np.ndarray, 
                               'n_train_samples': int, 'n_test_samples': int,
                               'train_emotion_counts': dict, 'test_emotion_counts': dict}}}
    """
    results = {}
    
    for op in operations:
        # Find common layers across all emotions for this op
        # (Assuming all emotions have same layers, but safe to check)
        layers = set()
        first_emo = emotions[0]
        if first_emo in train_split and op in train_split[first_emo]:
            layers = set(train_split[first_emo][op].keys())
        
        if not layers:
            continue
            
        results[op] = {}
        sorted_layers = sorted(list(layers))
        
        for layer in sorted_layers:
            X_train, y_train, label_map, train_emotion_counts = prepare_multiclass_data(
                train_split, emotions, op, layer, return_counts=True
            )
            X_test, y_test, _, test_emotion_counts = prepare_multiclass_data(
                test_split, emotions, op, layer, return_counts=True
            )
            
            if len(X_train) == 0 or len(X_test) == 0:
                continue
            
            clf = train_multiclass_classifier(X_train, y_train, method=classifier_type)
            
            y_train_pred = clf.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)
            
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=list(label_map.keys()))
            with np.errstate(divide='ignore', invalid='ignore'):
                row_sums = cm.sum(axis=1, keepdims=True)
                norm_cm = np.divide(cm, row_sums, where=row_sums != 0)
                norm_cm = np.nan_to_num(norm_cm)
            
            results[op][layer] = {
                'accuracy': acc,
                'train_accuracy': train_acc,
                'confusion_matrix': norm_cm,
                'label_map': label_map,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'train_emotion_counts': train_emotion_counts,
                'test_emotion_counts': test_emotion_counts
            }
            
    return results
