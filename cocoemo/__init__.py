"""CoCoEmo: Composable and Controllable Human-Like Emotional TTS via Activation Steering.

Reference implementation for the paper. Subpackages:

  * ``steering``         - backbone-agnostic steering vector construction & injection
  * ``discriminability`` - layer/operator linear-separability probing ("where to steer")
  * ``evaluation``       - per-utterance and mixed-emotion metrics (no TTS backbone needed)
  * ``backbones``        - per-model adapters (cosyvoice2, indextts2), each in its own env
  * ``data``             - split manifests + dataloaders

See README.md for the stage-by-stage workflow and per-stage environments.
"""

__version__ = "0.1.0"
