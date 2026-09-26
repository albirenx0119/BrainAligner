# Local implementation notes

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Open http://127.0.0.1:8000.

## Processing

- TIFF/OME-TIFF first series is normalized to `Z,C,Y,X`; explicit `C` or RGB `S` axis is channel, and remaining non-spatial axes are folded into slices.
- Folder import sorts `.tif` / `.tiff` paths and concatenates each file as serial slices. Channel counts must match. Image dimensions are centered on a common canvas and data types promoted to one shared dtype.
- The selected channel is registered by multiresolution normalized cross-correlation. It estimates rotation and translation only, pairwise from the reference outward. Composed transforms are rendered once for every channel.
- Output ZIP contains an OME-TIFF and `transforms.json` with matrices and correlation scores.

## Limitations

This is a classical rigid intensity-registration baseline, not a learned AI model or brain-region segmentation. The transform contains rotation and translation only, with no local deformation. It does not support non-rigid tissue deformation, atlas registration, manual mask editing, physical pixel-size preservation, or tiled viewing of very large images. It downsamples registration inputs to at most the selected proxy size (512, 768, 1024, or 1536 px) on the long side. Inspect results before analysis.

The hosted Render version receives uploaded TIFF data for processing. This prototype stores uploads in process memory and does not write them to disk. The free hosting plan may sleep when idle; the first visit can take time to wake up.


