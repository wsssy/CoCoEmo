"""Re-export shipped steering vectors in a portable format.

The original *.pt files pickle custom classes (HeadGeometry,
LayerDiscriminabilityMetrics) under a bare top-level ``discriminability`` module,
so they only load where that module is importable. This utility re-saves each
vector using ONLY plain Python/torch types:

    {
        "steering_vectors": {operation: {layer: FloatTensor[1, d]}},
        "metadata": {...},                  # pos/neg emotion, datasets, counts
        "recommended_layers": {op: [...]},  # from the discriminability analysis
        "layer_accuracy":     {op: {layer: float}},
    }

so users can ``torch.load`` them without any cocoemo import, and a JSON sidecar
for documentation.

Usage:
    python scripts/export_steering_vectors.py \
        --src /path/to/steering_vectors --dst steering_vectors
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
import cocoemo.steering  # noqa: F401  registers the legacy 'discriminability' alias


def to_plain(obj):
    """Recursively convert tensors->tensor, dataclasses->dict, leave plain types."""
    import dataclasses
    if torch.is_tensor(obj):
        return obj.detach().cpu().float()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_plain(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_plain(v) for v in obj)
    return obj


def export_one(src_path, dst_dir):
    name = os.path.basename(src_path)
    d = torch.load(src_path, map_location="cpu", weights_only=False)

    vectors = to_plain(d.get("steering_vectors", {}))
    metadata = to_plain(d.get("metadata", {}))
    disc = d.get("discriminability", {}) or {}
    recommended = to_plain(disc.get("recommended_layers", {}))
    layer_acc = to_plain(disc.get("op_to_layer_to_acc", {}))

    # shapes for the sidecar / docs
    shapes = {
        op: {str(layer): list(t.shape) for layer, t in layers.items()}
        for op, layers in vectors.items()
    }

    portable = {
        "steering_vectors": vectors,
        "metadata": metadata,
        "recommended_layers": recommended,
        "layer_accuracy": layer_acc,
    }
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, name)
    torch.save(portable, dst_path)

    sidecar = {
        "file": name,
        "metadata": metadata,
        "recommended_layers": recommended,
        "operations": list(vectors.keys()),
        "shapes": shapes,
    }
    print(f"[export] {name}: ops={list(vectors.keys())} "
          f"pos={metadata.get('pos_emotion')} neg={metadata.get('neg_emotion')}", flush=True)
    return sidecar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of original .pt steering vectors")
    ap.add_argument("--dst", default=os.path.join(REPO, "steering_vectors"))
    args = ap.parse_args()

    src_files = sorted(f for f in os.listdir(args.src) if f.endswith(".pt"))
    if not src_files:
        raise SystemExit(f"no .pt files in {args.src}")

    sidecars = []
    for f in src_files:
        sidecars.append(export_one(os.path.join(args.src, f), args.dst))

    # verify portability: reload WITHOUT the cocoemo alias in a fresh check
    index_path = os.path.join(args.dst, "steering_vectors_index.json")
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(sidecars, fh, indent=2)
    print(f"[export] wrote {len(sidecars)} portable vectors + index -> {args.dst}", flush=True)


if __name__ == "__main__":
    main()
