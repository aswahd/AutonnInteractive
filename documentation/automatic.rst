Auto nnInteractive
==================

Segment a query volume from annotated support images, without query prompts.
Requires a trained Auto nnInteractive checkpoint.

Inference
---------

Use preprocessed images matching the checkpoint's normalization, orientation,
and spacing. Supports must match its model input size, typically 192 cubed:

* Query: ``(1, D, H, W)``.
* Support images: ``(K, 1, d, h, w)``.
* Support prompts: ``(K, 7, d, h, w)``.

For dense annotations, put each nonempty binary target mask in the checkpoint's
``prev_seg`` channel (``session.source["channel_mapping"]["prev_seg"]``) and
zero the remaining channels.

.. code-block:: python

    import numpy as np
    from nnInteractive.automatic import AutomaticSegmentationSession

    session = AutomaticSegmentationSession("checkpoint.pt", device="cuda")
    with np.load("supports.npz") as bank:
        session.set_context(bank["images"], bank["prompts"])
    result = session.predict(np.load("query.npy"), refinement="autozoom")
    np.save("mask.npy", result.mask)

Reuse the context for subsequent queries. Output masks retain the query grid.
AutoZoom requires the original nnInteractive model files; supply
``original_model_dir`` when constructing the session if they have moved.
Use ``refinement="none"`` to skip refinement.

Training
--------

Prepare a manifest from an nnU-Net preprocessed dataset::

    nninteractive-automatic prepare --dataset /path/to/dataset \
      --configuration nnUNetPlans_3d_fullres --output data.json

Copy an `example configuration <../examples/automatic/>`_ to ``train.json``;
set its checkpoint and manifest paths, then train::

    nninteractive-automatic train --config train.json --output runs/model

Resume with ``--resume runs/model/last.pt``. Architecture changes require a new
run. See `configuration options <../nnInteractive/automatic/config.py>`_ for
model and training settings.
