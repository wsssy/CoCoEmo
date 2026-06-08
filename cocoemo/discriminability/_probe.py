"""
discriminability.py: Code for computing and visualizing emotion representation discriminability
in CosyVoice model layers, based on the DISCO method.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import copy
import gc

# matplotlib / seaborn are only needed by the optional plotting helpers below.
# Import them lazily so that steering / extraction (which only need the compute
# functions and HeadGeometry) do not require a plotting stack.
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None
try:
    import seaborn as sns
except Exception:  # pragma: no cover - plotting is optional
    sns = None
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


@dataclass
class LayerDiscriminabilityMetrics:
    """Container describing how separable an emotion pair is at a layer."""

    accuracy: float
    train_accuracy: float
    num_eval: int
    num_train: int
    centroid_gap: float


@dataclass
class HeadDiscriminabilityMetrics:
    """Stores discriminability statistics for a single attention head."""

    accuracy: float
    train_accuracy: float
    num_eval: int
    num_train: int


@dataclass
class HeadGeometry:
    """Describes CosyVoice/Qwen attention geometry for reshaping."""

    num_heads: int
    num_kv_heads: int
    head_dim: int


HEAD_BASED_OPERATIONS = {"attn_output", "q_proj", "k_proj", "v_proj",}
KV_ONLY_OPERATIONS = {"k_proj", "v_proj"}


# def _balance_classes(pos_feats: torch.Tensor, neg_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Returns class-balanced views (min class count) without additional copies."""
#     n = min(len(pos_feats), len(neg_feats))
#     if n == 0:
#         return pos_feats, neg_feats
#     return pos_feats[:n], neg_feats[:n]


def _centroid_prediction(features: torch.Tensor, mu_pos: torch.Tensor, mu_neg: torch.Tensor) -> torch.Tensor:
    """Returns binary predictions using a nearest-centroid classifier."""
    # Ensure CPU tensors for stable numpy <-> torch interop downstream
    feats = features.float()
    dist_pos = torch.norm(feats - mu_pos, dim=1)
    dist_neg = torch.norm(feats - mu_neg, dim=1)
    return (dist_pos <= dist_neg).long()


def _train_classifier(pos_feats: torch.Tensor, neg_feats: torch.Tensor, method: str = 'centroid') -> Any:
    """
    Trains a classifier on the provided positive and negative features.
    
    Args:
        pos_feats: Positive class features [N_pos, D]
        neg_feats: Negative class features [N_neg, D]
        method: 'centroid', 'linear', or 'svm'
        
    Returns:
        Trained model or tuple of centroids
    """
    # Ensure CPU tensors/numpy for sklearn
    pos_np = pos_feats.float().cpu().numpy()
    neg_np = neg_feats.float().cpu().numpy()
    
    if method == 'centroid':
        mu_pos = torch.from_numpy(pos_np.mean(axis=0))
        mu_neg = torch.from_numpy(neg_np.mean(axis=0))
        return (mu_pos, mu_neg)
        
    X = np.concatenate([pos_np, neg_np], axis=0)
    y = np.concatenate([np.ones(len(pos_np)), np.zeros(len(neg_np))], axis=0)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    if method == 'linear':
        # Logistic Regression as a linear probe
        clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', random_state=42, max_iter=8000))
        clf.fit(X, y)
        return clf
        
    elif method == 'svm':
        # Linear SVM
        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', random_state=42, max_iter=8000))
        clf.fit(X, y)
        return clf
        
    else:
        raise ValueError(f"Unknown classification method: {method}")


def _predict_classifier(model: Any, features: torch.Tensor, method: str = 'centroid') -> torch.Tensor:
    """
    Predicts labels using the trained classifier.
    
    Args:
        model: Trained model or centroids returned by _train_classifier
        features: Features to classify [N, D]
        method: 'centroid', 'linear', or 'svm'
        
    Returns:
        Binary predictions (0 or 1) as torch.LongTensor
    """
    feats_np = features.float().cpu().numpy()
    
    if method == 'centroid':
        mu_pos, mu_neg = model
        # Use existing torch-based helper for consistency if desired, 
        # or reimplement. Let's reuse the logic but we need torch tensors for mu
        return _centroid_prediction(features, mu_pos, mu_neg)
        
    elif method in ['linear', 'svm']:
        preds = model.predict(feats_np)
        return torch.from_numpy(preds).long()
        
    else:
        raise ValueError(f"Unknown classification method: {method}")


def _compute_classifier_accuracy(
    pos_train: torch.Tensor, 
    neg_train: torch.Tensor,
    pos_eval: torch.Tensor, 
    neg_eval: torch.Tensor,
    method: str = 'centroid'
) -> Tuple[float, float, int, int, float]:
    """
    Computes accuracy for a train/eval split using the specified method.
    
    Returns:
        (eval_acc, train_acc, n_eval, n_train, centroid_gap/margin)
    """
    # Train
    model = _train_classifier(pos_train, neg_train, method=method)
    
    # Predict Train
    train_feats = torch.cat([neg_train, pos_train], dim=0)
    train_labels = torch.cat([
        torch.zeros(len(neg_train), dtype=torch.long),
        torch.ones(len(pos_train), dtype=torch.long)
    ])
    train_preds = _predict_classifier(model, train_feats, method=method)
    train_acc = (train_preds == train_labels).float().mean().item()
    n_train = len(train_labels)
    
    # Free memory
    del train_feats, train_labels, train_preds
    
    # Predict Eval
    eval_feats = torch.cat([neg_eval, pos_eval], dim=0)
    eval_labels = torch.cat([
        torch.zeros(len(neg_eval), dtype=torch.long),
        torch.ones(len(pos_eval), dtype=torch.long)
    ])
    eval_preds = _predict_classifier(model, eval_feats, method=method)
    eval_acc = (eval_preds == eval_labels).float().mean().item()
    n_eval = len(eval_labels)
    
    # Free memory
    del eval_feats, eval_labels, eval_preds
    
    # Compute "gap" or margin metric
    metric_val = 0.0
    if method == 'centroid':
        mu_pos, mu_neg = model
        metric_val = torch.norm(mu_pos - mu_neg).item()
    elif method in ['linear', 'svm']:
        # For linear models, use margin or coefficient norm as metric
        # Access the classifier from the pipeline
        if isinstance(model, Pipeline):
            # The classifier is the last step
            clf = model.steps[-1][1]
        else:
            clf = model
            
        if hasattr(clf, 'coef_'):
            metric_val = np.linalg.norm(clf.coef_)
        else:
            metric_val = 0.0
        
    return eval_acc, train_acc, n_eval, n_train, metric_val


def _centroid_accuracy(pos_feats: torch.Tensor, neg_feats: torch.Tensor,
                       mu_pos: torch.Tensor, mu_neg: torch.Tensor) -> Tuple[float, int]:
    """Computes centroid-based accuracy for a split. Kept for backward compatibility."""
    if pos_feats.numel() == 0 or neg_feats.numel() == 0:
        return 0.0, 0
    labels = torch.cat(
        [torch.zeros(len(neg_feats), dtype=torch.long),
         torch.ones(len(pos_feats), dtype=torch.long)]
    )
    feats = torch.cat([neg_feats, pos_feats], dim=0)
    preds = _centroid_prediction(feats, mu_pos, mu_neg)
    accuracy = (preds == labels).float().mean().item()
    return accuracy, len(labels)




def _num_elements_for_op(op: str, geometry: HeadGeometry) -> int:
    """Returns number of heads/KV groups for a given operation name."""
    return geometry.num_kv_heads if op in KV_ONLY_OPERATIONS else geometry.num_heads


def _reshape_to_heads(features: torch.Tensor, num_elems: int, head_dim: int) -> torch.Tensor:
    """Reshapes flattened q/k/v tensors into [batch, num_elems, head_dim]."""
    if features.numel() == 0:
        return features.view(0, num_elems, head_dim)
    return features.view(features.size(0), num_elems, head_dim)


def compute_discriminability_for_steering(
    pos_reps: Dict[str, Dict[int, torch.Tensor]],
    neg_reps: Dict[str, Dict[int, torch.Tensor]],
    pos_reps_eval: Dict[str, Dict[int, torch.Tensor]],
    neg_reps_eval: Dict[str, Dict[int, torch.Tensor]],
    operations: Optional[List[str]] = None,
    balance_train: bool = False,
    balance_eval: bool = False,
    return_metrics: bool = False,
    classifier_type: str = 'centroid',
) -> Tuple[Dict[str, Dict[int, float]], Optional[Dict[str, Dict[int, LayerDiscriminabilityMetrics]]]]:
    """
    Computes discriminability (linear separability) for each operation/layer pair
    using the specified classifier ('centroid', 'linear', 'svm').

    Args:
        pos_reps: Positive representations from training split.
        neg_reps: Negative representations from training split.
        pos_reps_eval: Positive representations from evaluation split.
        neg_reps_eval: Negative representations from evaluation split.
        operations: Optional subset of operations to score. Default uses every key.
        balance_train: Whether to balance positive/negative counts before fitting centroids.
        balance_eval: Whether to balance counts before scoring the eval split.
        return_metrics: If True, also return a dict of LayerDiscriminabilityMetrics.

    Returns:
        Tuple where the first entry maps op->layer->accuracy. The second entry (optional)
        maps op->layer->LayerDiscriminabilityMetrics when return_metrics=True.
    """
    operations = operations or list(pos_reps.keys())
    op_to_layer_to_acc: Dict[str, Dict[int, float]] = {}
    op_to_layer_to_metrics: Optional[Dict[str, Dict[int, LayerDiscriminabilityMetrics]]] = {} if return_metrics else None

    for op in operations:
        if op not in pos_reps or op not in neg_reps:
            continue
        op_to_layer_to_acc[op] = {}
        if return_metrics:
            op_to_layer_to_metrics[op] = {}

        layers = sorted(pos_reps[op].keys())
        for layer in layers:
            pos_train = pos_reps[op][layer]
            neg_train = neg_reps[op][layer]
            pos_eval = pos_reps_eval[op].get(layer) if op in pos_reps_eval else None
            neg_eval = neg_reps_eval[op].get(layer) if op in neg_reps_eval else None

            if pos_train is None or neg_train is None:
                continue
            if pos_train.numel() == 0 or neg_train.numel() == 0:
                continue
            if pos_eval is None or neg_eval is None:
                continue
            if pos_eval.numel() == 0 or neg_eval.numel() == 0:
                continue

            balanced_pos_train, balanced_neg_train = (pos_train, neg_train)
            balanced_pos_eval, balanced_neg_eval = (pos_eval, neg_eval)
            # if balance_train:
            #     balanced_pos_train, balanced_neg_train = _balance_classes(pos_train, neg_train)
            # if balance_eval:
            #     balanced_pos_eval, balanced_neg_eval = _balance_classes(pos_eval, neg_eval)

            if balanced_pos_train.numel() == 0 or balanced_neg_train.numel() == 0:
                continue
            if balanced_pos_eval.numel() == 0 or balanced_neg_eval.numel() == 0:
                continue

            eval_acc, train_acc, n_eval, n_train, metric_val = _compute_classifier_accuracy(
                balanced_pos_train, balanced_neg_train, 
                balanced_pos_eval, balanced_neg_eval,
                method=classifier_type
            )

            op_to_layer_to_acc[op][layer] = eval_acc

            if return_metrics and op_to_layer_to_metrics is not None:
                op_to_layer_to_metrics[op][layer] = LayerDiscriminabilityMetrics(
                    accuracy=eval_acc,
                    train_accuracy=train_acc,
                    num_eval=n_eval,
                    num_train=n_train,
                    centroid_gap=metric_val
                )
            
            # Free memory after each layer
            del pos_train, neg_train, pos_eval, neg_eval
            del balanced_pos_train, balanced_neg_train, balanced_pos_eval, balanced_neg_eval
            gc.collect()

    if return_metrics:
        return op_to_layer_to_acc, op_to_layer_to_metrics
    return op_to_layer_to_acc, None


def compute_layer_discriminability(layer_to_rep_pos, layer_to_rep_neg,
                                   layer_to_rep_pos_eval, layer_to_rep_neg_eval):
    """
    Computes mean-difference classifier accuracy for a single component.
    This helper mirrors the DISCO notebook utilities and is kept for backwards
    compatibility. Prefer compute_discriminability_for_steering when possible.
    """
    layers = list(layer_to_rep_pos.keys())
    layer_to_acc = {}
    
    for layer in tqdm(layers, desc="Computing discriminability"):
        # Centroids for classification (computed on train set)
        mu_pos = layer_to_rep_pos[layer].mean(dim=0).numpy()
        mu_neg = layer_to_rep_neg[layer].mean(dim=0).numpy()
        
        # Eval features
        val_features_pos = layer_to_rep_pos_eval[layer].numpy()
        val_features_neg = layer_to_rep_neg_eval[layer].numpy()
        X_val = np.concatenate([val_features_neg, val_features_pos])
        Y_val = [0] * len(val_features_neg) + [1] * len(val_features_pos)
        
        # Score, Predict, Evaluate
        X_val_pos_scores = -1 * np.linalg.norm(X_val - mu_pos, axis=1)
        X_val_neg_scores = -1 * np.linalg.norm(X_val - mu_neg, axis=1)
        Pred_Val_Centroid = (X_val_pos_scores >= X_val_neg_scores).astype(int)
        Acc_Val_Centroid = accuracy_score(Y_val, Pred_Val_Centroid)
        
        layer_to_acc[layer] = Acc_Val_Centroid
    
    return layer_to_acc



def print_discriminability_report(
    op_to_layer_to_acc: Dict[str, Dict[int, float]],
    op_to_layer_to_metrics: Optional[Dict[str, Dict[int, LayerDiscriminabilityMetrics]]] = None,
) -> None:
    """
    Pretty-prints discriminability statistics per operation, optionally enriched
    with LayerDiscriminabilityMetrics if available.
    """
    print("\n" + "=" * 80)
    print("DISCRIMINABILITY REPORT")
    print("=" * 80)

    if not op_to_layer_to_acc:
        print("\n⚠ WARNING: No discriminability data available!")
        print("This usually means no representations were successfully extracted.")
        print("=" * 80)
        return

    for op, layer_to_acc in op_to_layer_to_acc.items():
        if not layer_to_acc:
            print(f"\n{op}:")
            print("  ⚠ No data available for this operation")
            continue

        layers = sorted(layer_to_acc.keys())
        accs = [layer_to_acc[layer] for layer in layers]
        best_layer = max(layer_to_acc, key=layer_to_acc.get)
        best_acc = layer_to_acc[best_layer]

        print(f"\n{op}:")
        print(f"  Best Layer: {best_layer} (Accuracy: {best_acc:.4f})")
        print(f"  Mean Accuracy: {np.mean(accs):.4f}")
        print(f"  Layers > 0.7 acc: {sum(a > 0.7 for a in accs)}/{len(accs)}")

        if op_to_layer_to_metrics and op in op_to_layer_to_metrics:
            metrics = op_to_layer_to_metrics[op].get(best_layer)
            if metrics:
                print(f"  Train Accuracy: {metrics.train_accuracy:.4f}")
                print(f"  Samples (train/eval): {metrics.num_train}/{metrics.num_eval}")
                print(f"  Centroid Gap: {metrics.centroid_gap:.4f}")

    print("\n" + "=" * 80)


def find_best_layers_for_steering(
    op_to_layer_to_acc: Dict[str, Dict[int, float]],
    top_k: int = 5,
    min_accuracy: float = 0.6,
) -> List[Tuple[str, int, float]]:
    """
    Identifies the most discriminative layers across all requested operations.
    """
    all_results: List[Tuple[str, int, float]] = []

    for op, layer_to_acc in op_to_layer_to_acc.items():
        for layer, acc in layer_to_acc.items():
            if acc >= min_accuracy:
                all_results.append((op, layer, acc))

    all_results.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop {top_k} layers for emotion steering (min accuracy {min_accuracy}):")
    print("-" * 60)
    for i, (op, layer, acc) in enumerate(all_results[:top_k], 1):
        print(f"{i}. {op:20s} Layer {layer:2d}: {acc:.4f}")

    return all_results[:top_k]




def create_discriminability_report(component_to_layer_to_acc, save_path=None):
    """
    Creates a detailed report of discriminability across all components and layers.
    
    Args:
        component_to_layer_to_acc: Dict mapping component_name -> layer_to_acc dict
        save_path: If provided, saves report to this path
    
    Returns:
        report_df: Pandas DataFrame with discriminability results
    """
    report_data = []
    
    for component_name, layer_to_acc in component_to_layer_to_acc.items():
        for layer, acc in layer_to_acc.items():
            report_data.append({
                'Component': component_name,
                'Layer': layer,
                'Accuracy': acc
            })
    
    # Handle empty case
    if not report_data:
        print("\n" + "="*80)
        print("DISCRIMINABILITY REPORT")
        print("="*80)
        print("\n⚠ WARNING: No discriminability data available!")
        print("This usually means no representations were successfully extracted.")
        print("Check that frontend_zero_shot() is being called correctly with all required parameters.")
        print("="*80)
        
        # Create empty DataFrame with correct columns
        report_df = pd.DataFrame(columns=['Component', 'Layer', 'Accuracy'])
        
        if save_path:
            report_df.to_csv(save_path, index=False)
            print(f"\nSaved empty report to {save_path}")
        
        return report_df
    
    report_df = pd.DataFrame(report_data)
    
    # Add summary statistics
    print("\n" + "="*80)
    print("DISCRIMINABILITY REPORT")
    print("="*80)
    
    for component_name in component_to_layer_to_acc.keys():
        component_df = report_df[report_df['Component'] == component_name]
        
        if component_df.empty:
            print(f"\n{component_name}:")
            print(f"  ⚠ No data available for this component")
            continue
        
        best_row = component_df.loc[component_df['Accuracy'].idxmax()]
        
        print(f"\n{component_name}:")
        print(f"  Best Layer: {best_row['Layer']}")
        print(f"  Best Accuracy: {best_row['Accuracy']:.4f}")
        print(f"  Mean Accuracy: {component_df['Accuracy'].mean():.4f}")
        print(f"  Std Accuracy: {component_df['Accuracy'].std():.4f}")
    
    print("\n" + "="*80)
    
    if save_path:
        report_df.to_csv(save_path, index=False)
        print(f"\nSaved detailed report to {save_path}")
    
    return report_df



###########################################################################
##               DISCO-Style Visualization Functions                    ##
###########################################################################

def plot_discriminability_heatmap(layer_to_acc, title="Discriminability Heatmap",
                                   save_path=None, figsize=(12, 3)):
    """
    Creates a heatmap visualization of discriminability across layers (DISCO-style).
    
    Args:
        layer_to_acc: Dict mapping layer_id -> accuracy
        title: Plot title
        save_path: If provided, saves plot to this path
        figsize: Figure size
    """
    layers = sorted(layer_to_acc.keys())
    accs = np.array([[layer_to_acc[layer] for layer in layers]])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(accs, cmap='coolwarm', vmin=0.5, vmax=1.0, aspect='auto')
    
    ax.set_yticks([0])
    ax.set_yticklabels(['Accuracy'])
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', fontsize=11, fontweight='bold')
    
    # Add value annotations
    for j, layer in enumerate(layers):
        text = ax.text(j, 0, f'{layer_to_acc[layer]:.3f}',
                      ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
    plt.close()


def plot_discriminability_curve_disco_style(layer_to_acc, pos_emotion, neg_emotion,
                                             title=None, save_path=None):
    """
    Creates a DISCO-style discriminability curve plot.
    
    Args:
        layer_to_acc: Dict mapping layer_id -> accuracy
        pos_emotion: Name of positive emotion
        neg_emotion: Name of negative emotion
        title: Plot title (auto-generated if None)
        save_path: If provided, saves plot to this path
    """
    if title is None:
        title = f'Discriminability Curve\n{pos_emotion} vs {neg_emotion}'
    
    layers = sorted(layer_to_acc.keys())
    accs = np.array([layer_to_acc[layer] for layer in layers])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Main curve
    ax.plot(layers, accs, 'o-', linewidth=2.5, markersize=8, 
           color='steelblue', label='Discriminability')
    
    # Chance level
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
              alpha=0.6, label='Chance Level')
    
    # Highlight best layer
    best_layer = layers[np.argmax(accs)]
    best_acc = accs[np.argmax(accs)]
    ax.scatter([best_layer], [best_acc], color='red', s=300, zorder=5,
              edgecolors='black', linewidth=2, label=f'Best: Layer {best_layer}')
    
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylim([0.4, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved curve to {save_path}")
    
    plt.close()


def create_combined_visualization(layer_to_acc, pos_emotion, neg_emotion,
                                  save_path=None, classifier_type=None):
    """
    Creates a combined visualization with multiple panels (DISCO-style).
    
    Args:
        layer_to_acc: Dict mapping layer_id -> accuracy
        pos_emotion: Name of positive emotion
        neg_emotion: Name of negative emotion
        save_path: If provided, saves plot to this path
        classifier_type: Optional classifier name to include in title
    """
    layers = sorted(layer_to_acc.keys())
    accs = np.array([layer_to_acc[layer] for layer in layers])
    
    fig = plt.figure(figsize=(18, 5))
    
    # Panel 1: Discriminability Curve
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(layers, accs, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Chance')
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Discriminability per Layer', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Heatmap
    ax2 = plt.subplot(1, 3, 2)
    accs_2d = accs.reshape(1, -1)
    im = ax2.imshow(accs_2d, cmap='coolwarm', vmin=0.5, vmax=1.0, aspect='auto')
    ax2.set_yticks([0])
    ax2.set_yticklabels(['Accuracy'])
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers)
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_title('Discriminability Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Accuracy')
    
    # Panel 3: CCDF (Complementary Cumulative Distribution)
    ax3 = plt.subplot(1, 3, 3)
    thresholds = np.linspace(0.5, 1.0, 100)
    fractions = [np.mean(accs >= t) for t in thresholds]
    ax3.plot(thresholds, fractions, linewidth=3, color='steelblue')
    ax3.set_xlabel('Accuracy Threshold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Fraction of Layers', fontsize=12, fontweight='bold')
    ax3.set_title('Discriminability Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0.5, 1.0])
    ax3.set_ylim([0, 1.0])
    
    title = f'Emotion Discriminability Analysis: {pos_emotion} vs {neg_emotion}'
    if classifier_type:
        title += f' ({classifier_type})'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined visualization to {save_path}")
    
    plt.close()


def heatmaps_and_ratios(HS, num_groups, layers, model_choice, dataset_choice, split='test'):
    ''' Plots heatmaps and ccdfs of concept discrimination accuracy in all attention head output, query key and value representation spaces '''
    
    # Use standard DISCO operations
    ops = ['q_proj', 'v_proj', 'k_proj', 'attn_output']
    op_to_name = {"q_proj": "Query", "v_proj": "Value", "attn_output": "Head Output", "k_proj": "Key"}

    fig, axes = plt.subplots(1, 5, figsize=(8 * 5, 8))
    im_list = [] 

    # HS structure: HS['centroid']['accs'][op][layer][head_idx] = {'train': acc, 'test': acc}
    ROW = 0 
    for cls_type in HS.keys(): # e.g., 'centroid'
        accs = copy.deepcopy(HS[cls_type]['accs'])
        
        # Handle GQA (Grouped Query Attention) replication if needed
        # For CosyVoice2 (16 heads), if we have fewer KV heads, we might need this.
        # But if compute_head_discriminability already returns 16 heads for everything, this loop just copies.
        for layer in layers:
            for op in ["v_proj", "k_proj"]:
                if op not in accs: continue
                
                # Check if we need to expand groups (only if heads < num_heads)
                # If data is already full size, this logic might need adjustment or be skipped.
                # Assuming standard DISCO logic here:
                J = 0 
                temp = {} # New dict
                sorted_indices = sorted(accs[op][layer].keys())
                
                # If we already have all heads, num_groups should be 1
                for IDX in sorted_indices:
                    for _ in range(num_groups):
                        temp[J] = accs[op][layer][IDX]
                        J += 1
                accs[op][layer] = temp 

    # Create DataFrames for plotting
    dfs_test = {}
    for op in ops:
        if op in accs:
            # Extract the specific split (e.g., 'test')
            # accs[op][layer][head] is a dict like {'train': 0.9, 'test': 0.8} or just a float
            # We need to handle both cases to be safe, but DISCO expects a dict.
            
            data_dict = accs[op]
            # Convert {layer: {head: {split: acc}}} to DataFrame
            # Rows: heads, Cols: layers
            
            # First, build a matrix or list of dicts
            rows = []
            for layer in layers:
                row = {}
                if layer in data_dict:
                    for head, metrics in data_dict[layer].items():
                        if isinstance(metrics, dict):
                            row[head] = metrics.get(split, 0.5)
                        else:
                            row[head] = metrics # Assume float
                rows.append(row)
            
            df = pd.DataFrame(rows, index=layers)
            dfs_test[op] = df
        else:
            dfs_test[op] = pd.DataFrame()

    COL = 1
    for op in ['q_proj', 'v_proj', 'k_proj', 'attn_output']:
        if op not in dfs_test or dfs_test[op].empty:
            COL += 1
            continue
            
        df_test = dfs_test[op]
        # Transpose for Heatmap: X=Layers, Y=Heads
        im1 = axes[COL].imshow(df_test.T, origin='lower', aspect='auto', cmap='coolwarm', vmin=.5, vmax=1)
        
        if ROW == 0:
            axes[COL].set_title(op_to_name[op], fontweight="bold", fontsize=30)
        im_list.append(im1)
        COL += 1

    # Plot CCDF (Ratio Curve)
    xaxis = np.linspace(.5, 1, num=100)
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

    cidx = 0 
    for op in ops:
        if op not in dfs_test or dfs_test[op].empty:
            cidx += 1
            continue
            
        df_test = dfs_test[op]
        prop_above_T_te = [] 
        
        # Flatten all accuracies for this operation
        flat_te = df_test.values.flatten()
        
        for T in np.linspace(.5, 1, num=100):
            prop_above_T_te.append(len(flat_te[flat_te >= T]) / len(flat_te))
            
        axes[0].plot(xaxis, prop_above_T_te, label=op_to_name[op], color=colors[cidx], linewidth=3)
        cidx += 1
    
    ROW += 1
    axes[1].set_ylabel("Head", fontweight='bold', fontsize=20) # Y-axis is Head

    for COL in range(1, 5):
        axes[COL].set_xlabel("Layer", fontweight='bold', fontsize=20) # X-axis is Layer

    axes[0].set_title("Discriminability Curve", fontweight="bold", fontsize=30)
    axes[0].set_xlabel("Acc. Thresh.", fontweight='bold', fontsize=20)

    for COL in range(0, 5):
        axes[COL].tick_params(axis='both', labelsize=14)
        for label in axes[COL].get_xticklabels():
            label.set_fontweight('bold')
        for label in axes[COL].get_yticklabels():
            label.set_fontweight('bold')

    axes[0].set_ylabel("Fraction of Heads", fontweight='bold', fontsize=20)

    fig.suptitle(f'Representation Space Accuracy\n Model: {model_choice}, Data: {dataset_choice}', fontsize=30, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.92, 0.95]) 

    if im_list:
        cbar = fig.colorbar(im_list[0], ax=axes, location='right', shrink=1)
        for label in cbar.ax.get_yticklabels():
            label.set_fontsize(20)
            label.set_fontweight('bold')
        cbar.ax.set_title("Acc", fontsize=30, fontweight='bold', pad=10)

    axes[0].legend(
        loc='lower left',          
        fontsize=16,
        frameon=True,               
        fancybox=True,              
        framealpha=0.8,             
        edgecolor='gray',           
        title='Representation',     
        title_fontsize=18
    )
    
    return fig


def head_stats(ops, model_use, K, data, layers,
            op_to_layer_to_rep_pos, op_to_layer_to_rep_neg,
            op_to_layer_to_rep_pos_eval, op_to_layer_to_rep_neg_eval, 
            num_heads, head_dim, num_groups, eval_split='test',
            classifier_type='centroid'):
    ''' computes mean-difference classifier accuracy for all representation spaces on head based operations ''' 

    for op in ops:
        assert op in ['k_proj','v_proj','q_proj','attn_output'], 'invalid op'

    # Determine sizes from data
    # data structure: {'train': {'pos_prompts': [...], ...}, ...}
    # But here we might just use the tensors directly if passed
    
    # We will use the tensors passed in op_to_layer_to_rep_* directly
    
    num_kv = num_heads // num_groups
    results = {} 

    accs_mean_centroid = {op: {layer: {} for layer in layers} for op in ops}

    # We iterate through layers and operations
    for layer in layers:
        for op in ops: 
            if op not in op_to_layer_to_rep_pos or layer not in op_to_layer_to_rep_pos[op]:
                continue

            # Determine number of elements (heads)
            N_ELEMS = num_kv if op in ['k_proj', 'v_proj'] else num_heads
            
            # Get tensors
            # Shape expected: [batch, num_heads * head_dim] -> need to reshape
            pos_train = op_to_layer_to_rep_pos[op][layer]
            neg_train = op_to_layer_to_rep_neg[op][layer]
            pos_eval = op_to_layer_to_rep_pos_eval[op][layer]
            neg_eval = op_to_layer_to_rep_neg_eval[op][layer]
            
            # Reshape to [batch, num_heads, head_dim]
            # Note: This assumes the input tensors are [batch, hidden_size]
            def to_heads(t, n_elems):
                return t.view(t.shape[0], n_elems, head_dim)

            try:
                X_pos_train = to_heads(pos_train, N_ELEMS)
                X_neg_train = to_heads(neg_train, N_ELEMS)
                X_pos_eval = to_heads(pos_eval, N_ELEMS)
                X_neg_eval = to_heads(neg_eval, N_ELEMS)
            except Exception as e:
                print(f"Skipping {op} layer {layer}: shape mismatch {e}")
                continue

            for IDX in range(N_ELEMS):
                # Extract head features
                h_pos_train = X_pos_train[:, IDX, :]
                h_neg_train = X_neg_train[:, IDX, :]
                h_pos_eval = X_pos_eval[:, IDX, :]
                h_neg_eval = X_neg_eval[:, IDX, :]
                
                eval_acc, _, _, _, _ = _compute_classifier_accuracy(
                    h_pos_train, h_neg_train,
                    h_pos_eval, h_neg_eval,
                    method=classifier_type
                )
                
                # Store result
                # DISCO format: {'train': val, 'test': val}
                accs_mean_centroid[op][layer][IDX] = {eval_split: eval_acc}

    results['centroid'] = {'accs': accs_mean_centroid}
    return results



###########################################################################
##            Latent Space Visualization (PCA & t-SNE)                  ##
###########################################################################

def visualize_emotion_pca(
    emotion_embeddings: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
    title: str = "Emotion Latent Space (PCA)",
    n_components: int = 2
) -> Tuple[np.ndarray, PCA]:
    """
    Visualize emotion embeddings using PCA.
    
    Args:
        emotion_embeddings: Dict mapping emotion_name -> embeddings tensor [N, D]
        save_path: Optional path to save the plot
        title: Plot title
        n_components: Number of PCA components (2 or 3)
        
    Returns:
        transformed_data: PCA-transformed data
        pca: Fitted PCA object
    """
    # Prepare data
    all_embeddings = []
    all_labels = []
    
    for emotion, emb in emotion_embeddings.items():
        all_embeddings.append(emb.numpy() if isinstance(emb, torch.Tensor) else emb)
        all_labels.extend([emotion.capitalize()] * len(emb))
    
    X = np.vstack(all_embeddings)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Plot
    emotions = list(emotion_embeddings.keys())
    colors = sns.color_palette("husl", len(emotions))
    
    if n_components == 2:
        plt.figure(figsize=(12, 8))
        
        for i, emotion in enumerate(emotions):
            mask = np.array(all_labels) == emotion.capitalize()
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=emotion.capitalize(), 
                       alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', 
                   fontsize=14, fontweight='bold')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', 
                   fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, emotion in enumerate(emotions):
            mask = np.array(all_labels) == emotion.capitalize()
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                      c=[colors[i]], label=emotion.capitalize(), 
                      alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', 
                      fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', 
                      fontsize=12, fontweight='bold')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', 
                      fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best')
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA plot to {save_path}")
    
    plt.close()
    
    return X_pca, pca


def visualize_emotion_tsne(
    emotion_embeddings: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
    title: str = "Emotion Latent Space (t-SNE)",
    perplexity: int = 30,
    n_iter: int = 1000
) -> np.ndarray:
    """
    Visualize emotion embeddings using t-SNE.
    
    Args:
        emotion_embeddings: Dict mapping emotion_name -> embeddings tensor [N, D]
        save_path: Optional path to save the plot
        title: Plot title
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        
    Returns:
        transformed_data: t-SNE-transformed data
    """
    # Prepare data
    all_embeddings = []
    all_labels = []
    
    for emotion, emb in emotion_embeddings.items():
        all_embeddings.append(emb.numpy() if isinstance(emb, torch.Tensor) else emb)
        all_labels.extend([emotion.capitalize()] * len(emb))
    
    X = np.vstack(all_embeddings)
    
    # Apply t-SNE
    print(f"Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    emotions = list(emotion_embeddings.keys())
    colors = sns.color_palette("husl", len(emotions))
    
    for i, emotion in enumerate(emotions):
        mask = np.array(all_labels) == emotion.capitalize()
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[colors[i]], label=emotion.capitalize(), 
                   alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    
    plt.close()
    
    return X_tsne


###########################################################################
##            Steering Vector Masking (DISCO-inspired)                   ##
###########################################################################


def apply_head_mask(ops, mask_criterion, num_kv, num_heads, op_to_df_val, op_to_df_train, K, op_to_layer_to_meandiff, head_dim):
    ''' masks non most-discrimintative heads ''' 

    assert mask_criterion in ['ValAcc', 'TrainAcc']; "Invalid acc criterion for masking"

    for op in ops:
        if op in ['k','v']:
            N_ELEMS = num_kv
        elif op in ['q', 'attn_output']:
            N_ELEMS = num_heads

        if mask_criterion == 'ValAcc':
            stacked = op_to_df_val[op].stack()              # Convert to multi-index series
        elif mask_criterion == 'TrainAcc':
            stacked = op_to_df_train[op].stack()            

        top_k = stacked.nlargest(K)                         # Get top-k values
        AttnHead_IDX, Layer_IDX = zip(*top_k.index)         # Extract row, col indices

        L_TO_KEEP = {L_IDX : [] for L_IDX in Layer_IDX}
        for ATT_HEAD, L_IDX in zip(AttnHead_IDX, Layer_IDX):
            L_TO_KEEP[L_IDX].append(ATT_HEAD)
        
        for L_IDX in L_TO_KEEP.keys():
            HEAD_KEEP = torch.tensor(L_TO_KEEP[L_IDX])
            REPR = op_to_layer_to_meandiff[op][L_IDX].view(1,N_ELEMS,head_dim)
            MASK = torch.zeros_like(REPR)
            MASK[:,HEAD_KEEP,:] = 1
            REPR = REPR * MASK
            REPR = REPR.view(1,-1)
            op_to_layer_to_meandiff[op][L_IDX] = REPR
    
    return op_to_layer_to_meandiff


def keep_most_disc_spaces(op_to_layer_to_meandiff, use_best_heads, use_best_layers, data, model, op_combo, tokenizer, op_to_hookinfo_model, L_TOTAL, eval_split, model_choice, num_kv, num_heads, head_dim, num_groups, op_to_layer_to_rep_pos, op_to_layer_to_rep_neg):
    ''' masks non most discriminative heads or layers in mean-difference vector dictionary (op_to_layer_to_meandiff) '''

    assert not (use_best_heads != -1 and use_best_layers == 1); "cannot select both heads and layers"
    layers = list(range(0,L_TOTAL))

    # compute features for eval split 
    op_to_layer_to_rep_pos_eval, op_to_layer_to_rep_neg_eval = extract_pos_neg(data, model, op_combo, tokenizer, op_dict = op_to_hookinfo_model, layers = L_TOTAL, split = eval_split)

    if use_best_heads != -1: # Mask non top heads for head based methods
        HS = head_stats(op_combo, model_choice, None, data, layers, op_to_layer_to_rep_pos, op_to_layer_to_rep_neg, op_to_layer_to_rep_pos_eval, op_to_layer_to_rep_neg_eval, num_heads=num_heads, head_dim=head_dim, num_groups = num_groups)
                    
        masking_eval_df, masking_train_df =  HS['centroid'][f'op_to_df_{eval_split}'], HS['centroid']['op_to_df_train']

        if type(use_best_heads) == list: # if we are steering multiple operations (i.e., query and value at the same time)
            for op, H in zip(op_combo, use_best_heads):
                op_to_layer_to_meandiff = apply_head_mask([op], mask_criterion = "ValAcc", num_kv = num_kv, num_heads = num_heads, op_to_df_val = masking_eval_df, op_to_df_train = masking_train_df,  K = H, op_to_layer_to_meandiff = op_to_layer_to_meandiff, head_dim = head_dim) 
        else:
            op_to_layer_to_meandiff = apply_head_mask(op_combo, mask_criterion = "ValAcc", num_kv = num_kv, num_heads = num_heads, op_to_df_val = masking_eval_df, op_to_df_train = masking_train_df, K = use_best_heads, op_to_layer_to_meandiff = op_to_layer_to_meandiff, head_dim = head_dim) 

    elif use_best_layers == 1: # mask not top layers
        _, op_to_layer_to_meandiff = get_layer_accs(op_combo, op_to_layer_to_rep_pos,op_to_layer_to_rep_neg, op_to_layer_to_rep_pos_eval, op_to_layer_to_rep_neg_eval, op_to_layer_to_meandiff, mask = True)

    return op_to_layer_to_meandiff


def create_mean_difference(
    operations: List[str],
    pos_reps: Dict[str, Dict[int, torch.Tensor]],
    neg_reps: Dict[str, Dict[int, torch.Tensor]]
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Creates steering vectors as mean difference between positive and negative representations.
    
    Args:
        operations: List of operations to create vectors for
        pos_reps: Positive emotion representations {op: {layer: tensor}}
        neg_reps: Negative emotion representations {op: {layer: tensor}}
    
    Returns:
        steering_vectors: {op: {layer: vector}}
    """
    steering_vectors = {}
    
    for op in operations:
        if op not in pos_reps or op not in neg_reps:
            continue
        
        steering_vectors[op] = {}
        layers = sorted(pos_reps[op].keys())
        
        for layer in layers:
            if layer not in neg_reps[op]:
                continue
            
            pos = pos_reps[op][layer]
            neg = neg_reps[op][layer]
            
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            
            # Mean difference
            mu_pos = pos.mean(dim=0)
            mu_neg = neg.mean(dim=0)
            steering_vectors[op][layer] = mu_pos - mu_neg
    
    return steering_vectors

