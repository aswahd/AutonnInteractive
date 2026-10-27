# Training with generated regions

Generate additional binary masks from complete, unnormalized CT scans using the original frozen nnInteractive model and native AutoZoom:

```bash
python -m nnInteractive.automatic.generate_supervoxels \
  --manifest /path/to/real_manifest.json \
  --raw-dataset /path/to/raw_dataset \
  --model-dir /path/to/nnInteractive_v1.0 --fold 0 \
  --output /path/to/generated_regions
```

The manifest must index nnU-Net preprocessed arrays and explicit train/validation splits. Only training scans are eligible. Use `--limit 2` for a small initial run, or `--shards N --shard-index I` to divide generation across workers. For MRI-normalized data and matching raw geometry, add `--modality MRI`.

Masks must pass size, overlap, and nearby-point consistency checks. Only the seed-connected component is retained. Each accepted region is stored separately, allowing overlap with anatomical labels. Source hashes, geometry, and generation settings are recorded; incompatible settings require a new output directory.

After all selected cases finish, repeat the command with `--finalize`. This writes `OUTPUT/manifest.json`; cases with no accepted masks remain documented and are excluded from generated-region sampling.

Use [totalsegmentator_supervoxels.json](../examples/automatic/totalsegmentator_supervoxels.json) for training:

- `checkpoint_dir`: original nnInteractive model folder.
- `data_manifest`: real-label dataset manifest.
- `supervoxel_manifest`: generated-region manifest.
- `training.supervoxel_probability`: probability of a generated-region episode; the template uses 0.2.

Real-label episodes use distinct support/query scans. Generated-region episodes use independently augmented views of one source scan; region IDs do not imply correspondence across patients. Validation uses real labels only. Configure `validation_targets` for structures with positive held-out cases.

```bash
nninteractive-automatic train --config train.json --output runs/model
```

See [the quick start](automatic.rst) for inference and [configuration options](../nnInteractive/automatic/config.py) for sampling, augmentation, and validation settings.
