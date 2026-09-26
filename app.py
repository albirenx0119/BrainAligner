from __future__ import annotations
import io, json, re, uuid, zipfile
import base64
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import tifffile
from registration import register_stack
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
DATA: dict[str, dict[str, Any]] = {}
app = FastAPI(title='BrainAligner')

def natural_key(value: str):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r'(\d+)', value)]

def to_zcyx(arr: np.ndarray, axes: str) -> np.ndarray:
    axes = axes.upper()
    if 'Y' not in axes or 'X' not in axes: raise ValueError(f'Y/X axes not found (axes={axes})')
    c_axis = axes.find('C') if 'C' in axes else (axes.find('S') if 'S' in axes else -1)
    y_axis, x_axis = axes.index('Y'), axes.index('X')
    keep = {y_axis, x_axis}
    if c_axis >= 0: keep.add(c_axis)
    other_axes = [i for i in range(arr.ndim) if i not in keep]
    perm = other_axes + ([c_axis] if c_axis >= 0 else []) + [y_axis, x_axis]
    a = np.transpose(arr, perm)
    z = int(np.prod([arr.shape[i] for i in other_axes])) if other_axes else 1
    c = arr.shape[c_axis] if c_axis >= 0 else 1
    return np.ascontiguousarray(a.reshape(z, c, arr.shape[y_axis], arr.shape[x_axis]))

def display8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32); lo, hi = np.percentile(a, [1, 99.8])
    if hi <= lo: hi = lo + 1
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

def png_data(a: np.ndarray, max_side: int = 1200) -> str:
    im = display8(a); h, w = im.shape; scale = min(1, max_side / max(h, w))
    if scale < 1: im = cv2.resize(im, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.png', im)
    return 'data:image/png;base64,' + __import__('base64').b64encode(buf).decode()

def info(s: dict[str, Any]) -> dict[str, Any]:
    a = s['data']
    return {'id': s['id'], 'name': s['name'], 'slices': a.shape[0], 'channels': a.shape[1], 'height': a.shape[2], 'width': a.shape[3], 'dtype': s['dtype'], 'axes': s['axes'], 'channelPreviews': [png_data(a[0, c]) for c in range(a.shape[1])], 'referencePreview': png_data(a[0, 0])}

@app.post('/api/upload')
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.tif', '.tiff')): raise HTTPException(400, 'TIFFファイルを選択してください')
    raw = await file.read()
    if len(raw) > 2_000_000_000: raise HTTPException(413, 'ファイルは2GB以下にしてください')
    try:
        with tifffile.TiffFile(io.BytesIO(raw)) as tf:
            series = tf.series[0]; data = to_zcyx(series.asarray(), series.axes); axes = series.axes
    except Exception as e: raise HTTPException(400, f'TIFFを読み込めません: {e}')
    if data.shape[0] * data.shape[1] > 1000: raise HTTPException(400, 'スライス数×チャンネル数が大きすぎます')
    h, w = data.shape[2:]
    masks = np.ones((data.shape[0], h, w), dtype=np.uint8)
    sid = str(uuid.uuid4()); s = {'id': sid, 'name': Path(file.filename).name, 'data': data, 'masks': masks, 'dtype': str(data.dtype), 'sliceDtypes': [str(data.dtype)] * data.shape[0], 'axes': axes, 'originalShape': list(data.shape)}; DATA[sid] = s
    return info(s)

@app.post('/api/upload-folder')
async def upload_folder(files: list[UploadFile] = File(...)):
    """Load TIFFs chosen from one directory and concatenate them as serial sections."""
    if not files: raise HTTPException(400, 'フォルダ内にTIFFファイルがありません')
    files = sorted((f for f in files if f.filename.lower().endswith(('.tif', '.tiff'))), key=lambda f: natural_key(f.filename.replace('\\', '/')))
    if not files: raise HTTPException(400, '選択したフォルダにTIFFファイルがありません')
    if len(files) > 1000: raise HTTPException(413, 'フォルダ内のTIFFは1000ファイル以下にしてください')
    arrays = []; axes_seen = []
    try:
        for f in files:
            raw = await f.read()
            if len(raw) > 2_000_000_000: raise HTTPException(413, f'{Path(f.filename).name} は2GBを超えています')
            with tifffile.TiffFile(io.BytesIO(raw)) as tf:
                series = tf.series[0]
                arrays.append(to_zcyx(series.asarray(), series.axes))
                axes_seen.append(series.axes)
    except HTTPException: raise
    except Exception as e: raise HTTPException(400, f'TIFFを読み込めません: {e}')
    channel_counts = {a.shape[1] for a in arrays}
    if len(channel_counts) != 1:
        raise HTTPException(400, f'チャンネル数が異なるTIFFがあります: {sorted(channel_counts)}。同じチャンネル構成の画像を選択してください')
    # Different section dimensions and bit depths are common in microscope exports.
    # Promote to a shared dtype and center each image on a common canvas.
    max_h = max(a.shape[2] for a in arrays); max_w = max(a.shape[3] for a in arrays)
    dtype = np.result_type(*(a.dtype for a in arrays))
    padded = []; masks = []
    for a in arrays:
        canvas = np.zeros((a.shape[0], a.shape[1], max_h, max_w), dtype=dtype)
        mask = np.zeros((a.shape[0], max_h, max_w), dtype=np.uint8)
        top = (max_h - a.shape[2]) // 2; left = (max_w - a.shape[3]) // 2
        canvas[:, :, top:top+a.shape[2], left:left+a.shape[3]] = a
        mask[:, top:top+a.shape[2], left:left+a.shape[3]] = 1
        padded.append(canvas)
        masks.append(mask)
    data = np.concatenate(padded, axis=0)
    support = np.concatenate(masks, axis=0)
    if data.shape[0] * data.shape[1] > 1000: raise HTTPException(400, 'スライス数×チャンネル数が大きすぎます')
    name = Path(files[0].filename.replace('\\', '/')).parent.name or 'folder-stack'
    sid = str(uuid.uuid4())
    slice_dtypes = [str(a.dtype) for a in arrays for _ in range(a.shape[0])]
    s = {'id': sid, 'name': f'{name} ({len(files)} files)', 'data': data, 'masks': support, 'dtype': str(data.dtype), 'sliceDtypes': slice_dtypes,
         'axes': 'folder:' + ','.join(axes_seen), 'originalShape': list(data.shape),
         'sourceFiles': [Path(f.filename.replace('\\', '/')).name for f in files]}
    DATA[sid] = s
    return info(s)

class AlignRequest(BaseModel):
    id: str
    reference_slice: int = 0
    channel: int = 0
    max_dimension: int = 768

@app.post('/api/align')
def align(req: AlignRequest):
    if req.id not in DATA: raise HTTPException(404, '画像セッションがありません')
    s = DATA[req.id]; src = s['data']; z, c, h, w = src.shape
    if not (0 <= req.reference_slice < z and 0 <= req.channel < c): raise HTTPException(400, 'スライスまたはチャンネルが範囲外です')
    if req.max_dimension not in (512, 768, 1024, 1536): raise HTTPException(400, 'Proxy size must be 512, 768, 1024, or 1536.')
    ref_idx = req.reference_slice
    matrices, transforms, failed, factor = register_stack(src[:, req.channel], s['masks'], ref_idx, req.max_dimension, s['sliceDtypes'])
    aligned = np.empty_like(src)
    for i, matrix in enumerate(matrices):
        affine = np.array([[matrix[0], matrix[2], matrix[4]], [matrix[1], matrix[3], matrix[5]]], dtype=np.float32)
        for ch in range(c):
            aligned[i, ch] = cv2.warpAffine(src[i, ch], affine, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    s['aligned'] = aligned; s['transforms'] = transforms; s['reference'] = ref_idx; s['channel'] = req.channel
    picks = sorted(set([0, ref_idx, min(z-1, ref_idx+1)]))
    return {'id': req.id, 'transforms': transforms, 'failedSlices': failed, 'proxyScale': factor, 'alignedPreviews': [{'slice': i, 'channels': [png_data(aligned[i, ch]) for ch in range(c)]} for i in picks], 'reference': ref_idx, 'channel': req.channel}

@app.get('/api/preview/{sid}/{slice_index}')
def preview(sid: str, slice_index: int, channel: int = 0, aligned: bool = False):
    if sid not in DATA: raise HTTPException(404, '画像セッションがありません')
    s = DATA[sid]; data = s.get('aligned') if aligned else s['data']
    if data is None: raise HTTPException(400, '先に位置合わせを実行してください')
    if not (0 <= slice_index < data.shape[0] and 0 <= channel < data.shape[1]): raise HTTPException(400, 'スライスまたはチャンネルが範囲外です')
    encoded = png_data(data[slice_index, channel]).split(',', 1)[1]
    return Response(base64.b64decode(encoded), media_type='image/png')

@app.get('/api/export/{sid}')
def export(sid: str):
    if sid not in DATA: raise HTTPException(404, '画像セッションがありません')
    s = DATA[sid]
    if 'aligned' not in s: raise HTTPException(400, '先に位置合わせを実行してください')
    buf = io.BytesIO(); metadata = {'source': s['name'], 'source_axes': s['axes'], 'source_shape_zcyx': s['originalShape'], 'reference_slice': s['reference'], 'reference_channel': s['channel'], 'transforms': s['transforms'], 'note': 'Applied as one rigid transform per slice to all channels.'}
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('transforms.json', json.dumps(metadata, ensure_ascii=False, indent=2))
        out = io.BytesIO(); tifffile.imwrite(out, s['aligned'], ome=True, metadata={'axes':'ZCYX'})
        zf.writestr(Path(s['name']).stem + '_aligned.ome.tif', out.getvalue())
    return Response(buf.getvalue(), media_type='application/zip', headers={'Content-Disposition': f'attachment; filename="{Path(s["name"]).stem}_aligned.zip"'})

app.mount('/', StaticFiles(directory=ROOT / 'static', html=True), name='static')

