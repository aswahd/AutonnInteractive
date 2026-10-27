Figure and volume-viewer source
===============================

This directory contains code only. Supply measured arrays and write generated figures,
HTML, meshes and reports to an external output directory. Model training/inference is
provided by ``nnInteractive.automatic``; no checkpoint or dataset is bundled here.

``render_figures.py`` renders CT cut planes, mask surfaces, attention and measured
curves from a supplied report/array bundle. It retains the recorded paper-study schema
and requires its named baseline/KD/Backprop and historical comparison arrays. It does
not run inference or manufacture measurements. Run rendering inside a compute job::

    python visualization/render_figures.py --data /path/to/measurements \
        --output /path/to/figures

``viewer/export.py`` and ``viewer/viewer.html`` build portable offline viewers from
saved full-volume predictions and recorded raw-window/support predictions::

    python visualization/viewer/export.py /path/to/full_volumes /path/to/patches \
        /path/to/viewers --assets /path/to/javascript_assets

The existing bundle schema has four 192-cubed supports/windows, support counts 1--4,
and separate native-AutoZoom results. The viewer's schema has not been changed into a
large-context data collector in this packaging cleanup. Older measured bundles remain
available in the local archive. Update collection/export together when preparing new
large-context figures; do not relabel old predictions as new-model results.

Provide ``three.min.js`` and ``fflate.min.js`` in the assets directory; they are embedded
in generated HTML. Runtime rendering uses NumPy, matplotlib, Pillow and scikit-image.
The optional ``viewer/test_viewer.py`` browser check additionally uses Playwright and
Chromium. It compares displayed mask pixels with embedded voxels and checks controls.
No browser assets, generated HTML or medical-image arrays are version controlled.
