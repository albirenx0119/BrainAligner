# BrainAligner

Multi-channel TIFF / OME-TIFF serial section alignment in a browser. The review workflow is inspired by [AlignRef](https://github.com/SatoruMuro/AlignRef).

## Live app

[Open BrainAligner](https://brain-aligner.onrender.com/)

> Prototype: registration currently uses OpenCV ECC rigid alignment. It does not include AI brain segmentation or non-rigid registration. Review results before research use.

## Quick start

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>.

## Deploy

A [Render Blueprint](render.yaml) is included. After pushing this repository, click to create the web service in your Render account:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/albirenx0119/BrainAligner)

Render assigns an `onrender.com` URL after deployment. Uploading TIFFs to a hosted version sends them to that hosting service for processing. This prototype keeps uploads in application memory and does not write them to disk.

## Folder import

Choose a directory containing `.tif` / `.tiff` files. Files are sorted by relative path and concatenated as serial slices. Different image dimensions are centered on a common canvas; differing data types are promoted to a shared type. Channel counts must match.

See [local implementation notes](README.local.md) for details and limitations.