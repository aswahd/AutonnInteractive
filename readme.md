# Auto nnInteractive

**Automatic 3D medical image segmentation from annotated examples.**

Auto nnInteractive adapts [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) to segment a new scan from a small set of annotated support scans. The supports specify the target; the query needs no clicks, boxes, or other manual prompts. Reuse the same supports across queries without updating model parameters.

## How it works

1. **Define the target with examples.** Supply support CT volumes and masks for the same structure.
2. **Transfer the target to a query.** Multiscale cross-attention conditions the query features on the annotated supports, producing a binary segmentation.
3. **Refine the prediction.** Optionally pass the predicted mask to a frozen original nnInteractive model using its native AutoZoom engine.

The model retains nnInteractive's 3D encoder and decoder. Attention operates at three encoder scales, with configurable attention depth. The repository supports flat attention and **hierarchical attention with learned support aggregation**, along with 3D rotary position encoding (RoPE) and interleaved query conditioning. Architecture settings are saved in each checkpoint.

Training uses annotated anatomical structures and optional generated regions. Context dropout trains the model to use different numbers of supports. One request segments one target; supply another support mask set to request a different structure.

## Installation

Use Python 3.10+ and a PyTorch installation appropriate for your hardware. CUDA is used for GPU training and inference. Install this repository from source to use its automatic segmentation module:

```bash
git clone https://github.com/aswahd/AutonnInteractive.git
cd AutonnInteractive
python -m venv .venv
source .venv/bin/activate
pip install -e ./client
pip install -e .
```

Install the client first: the repository builds two distributions sharing the `nnInteractive` namespace, and the full package depends on the client.

**A trained Auto nnInteractive checkpoint is required.** Original nnInteractive weights initialize the backbone and provide optional refinement; they do not contain the learned support-attention modules. The examples below use placeholder paths to your trained checkpoint and prepared arrays.

## Automatic inference

The public API accepts preprocessed arrays aligned with the checkpoint's training axes, spacing, and intensity normalization. Prepare support images and prompts at the checkpoint's input size, typically 192³. Query images can retain their complete preprocessed grid.

| Input | Shape | Content |
|---|---|---|
| Query | `(1, D, H, W)` | Preprocessed image to segment |
| Support images | `(K, 1, d, h, w)` | Annotated examples at the model input size |
| Support prompts | `(K, 7, d, h, w)` | Prompt channels aligned with the support images |

For dense annotations, place each binary target mask in the checkpoint's `prev_seg` channel (`session.source["channel_mapping"]["prev_seg"]`) and leave the other six channels zero. Supply nonempty support masks. Sparse support prompts can be converted to masks by checkpoints configured for frozen prompt conversion.

```python
import numpy as np
from nnInteractive.automatic import AutomaticSegmentationSession

session = AutomaticSegmentationSession("/path/to/checkpoint.pt", device="cuda")
with np.load("supports.npz") as bank:
    session.set_context(bank["images"], bank["prompts"])

query = np.load("query.npy")
result = session.predict(query, refinement="autozoom")
np.save("segmentation.npy", result.mask)
```

`result.mask` has the query's spatial shape. Use `refinement="none"` for the automatic model alone. AutoZoom also requires the original nnInteractive model files recorded in the checkpoint; pass `original_model_dir="/path/to/nnInteractive_v1.0"` to the session if they have moved.

The same workflow is available from the command line:

```bash
nninteractive-automatic predict \
  --checkpoint /path/to/checkpoint.pt \
  --query query.npy \
  --supports supports.npz \
  --refinement autozoom \
  --output prediction.npz
```

The inference API follows the checkpoint's whole-volume or sliding-window configuration and restores predictions to the query grid before optional refinement.

## Training

Start with [hierarchical.json](examples/automatic/hierarchical.json) for learned hierarchical attention, or use the TotalSegmentator template to include generated regions:

```bash
cp examples/automatic/totalsegmentator_supervoxels.json train.json
# Edit checkpoint_dir, data_manifest, and supervoxel_manifest for your data.
nninteractive-automatic train --config train.json --output runs/totalseg
```

The [TotalSegmentator data guide](documentation/totalsegmentator_supervoxels.md) covers manifest preparation, region generation, and mixing real-label and generated-region episodes. Training supports and held-out queries follow the manifest's explicit splits.

For a new learned hierarchical attention run, add these settings under `model` in the training configuration:

```json
{
  "cross_attention_type": "hierarchical",
  "support_aggregation": "learned",
  "rope_3d": true,
  "interleave_query_context": true
}
```

Resume with `--resume runs/totalseg/last.pt`. Architecture changes require a new run. `best.pt` reflects the configured validation metric, which may use query patches rather than complete volumes; keep the preprocessing, support bank, and evaluation cases fixed when comparing checkpoints.

## Documentation

| Resource | Contents |
|---|---|
| [Quick start](documentation/automatic.rst) | Input formats, inference, and training commands |
| [Training configurations](examples/automatic/) | Portable configuration templates |
| [Generated regions](documentation/totalsegmentator_supervoxels.md) | Optional training masks and dataset preparation |

## License

The code is licensed under [Apache-2.0](LICENSE). The original [nnInteractive pretrained weights](https://huggingface.co/MIC-DKFZ/nnInteractive) are licensed under **CC BY-NC-SA 4.0**. Checkpoint licenses are separate from the code license; consult the applicable model terms when using or distributing weights.
