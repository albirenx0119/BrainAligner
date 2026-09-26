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
- Selected channel is registered with OpenCV ECC rotation + translation against the chosen reference slice. The same per-slice transform is applied across every channel.
- Output ZIP contains an OME-TIFF and `transforms.json` with matrices and ECC scores.

## Limitations

This is a classical rigid intensity-registration baseline, not a learned AI model or brain-region segmentation. It does not support non-rigid tissue deformation, atlas registration, manual mask editing, physical pixel-size preservation, or tiled viewing of very large images. It downsamples registration inputs to at most 1400 px on the long side. Inspect results before analysis.

The hosted Render version receives uploaded TIFF data for processing. This prototype stores uploads in process memory and does not write them to disk. The free hosting plan may sleep when idle; the first visit can take time to wake up.

