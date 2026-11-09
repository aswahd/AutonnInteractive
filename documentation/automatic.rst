Auto nnInteractive
==================

Segment a query volume from annotated support images, without query prompts.
Requires a trained Auto nnInteractive checkpoint.

Inference
---------

Use preprocessed images matching the checkpoint's normalization, orientation,
and spacing. Use four supports. Dense supports receive foreground-centered
384-cubed crops, clipped to the scan and expanded to retain the target, then
resized to the model input (typically 192 cubed). The whole query is resized
for prediction, restored to its supplied grid, and refined once with AutoZoom.

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
    result = session.predict(np.load("query.npy"))
    np.save("mask.npy", result.mask)

``set_context`` supplies the annotated examples that define the target.
``predict`` finds that target in a new scan and refines it with AutoZoom,
without retraining or manual query prompts. The saved mask retains the query grid.

By default, we crop a foreground-centered 384-cubed context from each support
(clipped to the scan), resize it to 192 cubed, and downsample the entire query
to 192 cubed before prediction and AutoZoom. These settings may not be best for
your dataset. Try other ``support_context`` sizes or sliding-window query
inference (``full_volume=False``), which may preserve useful detail for smaller
structures. Compare the resulting masks on your own data to choose what works best.

AutoZoom requires the original nnInteractive model files; supply
``original_model_dir`` when constructing the session if they have moved.
Use ``refinement="none"`` to skip refinement.
Pass lists for differently shaped support scans, or ``support_context=None``
to ``set_context`` for supports already prepared at the model input size.
Use ``full_volume=False`` when constructing the session for sliding windows
(``--sliding-window`` in the CLI).

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
