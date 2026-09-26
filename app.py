from __future__ import annotations
import io, json, uuid, zipfile
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import tifffile
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
DATA: dict[str, dict[str, Any]] = {}
app = FastAPI(title='BrainAligner')

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
    sid = str(uuid.uuid4()); s = {'id': sid, 'name': Path(file.filename).name, 'data': data, 'dtype': str(data.dtype), 'axes': axes, 'originalShape': list(data.shape)}; DATA[sid] = s
    return info(s)

@app.post('/api/upload-folder')
async def upload_folder(files: list[UploadFile] = File(...)):
    """Load TIFFs chosen from one directory and concatenate them as serial sections."""
    if not files: raise HTTPException(400, 'フォルダ内にTIFFファイルがありません')
    files = sorted((f for f in files if f.filename.lower().endswith(('.tif', '.tiff'))), key=lambda f: f.filename.lower())
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
    padded = []
    for a in arrays:
        canvas = np.zeros((a.shape[0], a.shape[1], max_h, max_w), dtype=dtype)
        top = (max_h - a.shape[2]) // 2; left = (max_w - a.shape[3]) // 2
        canvas[:, :, top:top+a.shape[2], left:left+a.shape[3]] = a
        padded.append(canvas)
    data = np.concatenate(padded, axis=0)
    if data.shape[0] * data.shape[1] > 1000: raise HTTPException(400, 'スライス数×チャンネル数が大きすぎます')
    name = Path(files[0].filename.replace('\\', '/')).parent.name or 'folder-stack'
    sid = str(uuid.uuid4())
    s = {'id': sid, 'name': f'{name} ({len(files)} files)', 'data': data, 'dtype': str(data.dtype),
         'axes': 'folder:' + ','.join(axes_seen), 'originalShape': list(data.shape),
         'sourceFiles': [Path(f.filename.replace('\\', '/')).name for f in files]}
    DATA[sid] = s
    return info(s)

class AlignRequest(BaseModel):
    id: str
    reference_slice: int = 0
    channel: int = 0
    max_dimension: int = 1400

@app.post('/api/align')
def align(req: AlignRequest):
    if req.id not in DATA: raise HTTPException(404, '画像セッションがありません')
    s = DATA[req.id]; src = s['data']; z, c, h, w = src.shape
    if not (0 <= req.reference_slice < z and 0 <= req.channel < c): raise HTTPException(400, 'スライスまたはチャンネルが範囲外です')
    factor = min(1, req.max_dimension / max(h, w)); nh, nw = max(8, round(h*factor)), max(8, round(w*factor))
    def prep(im):
        out = display8(im)
        if (nh, nw) != out.shape: out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(out.astype(np.float32) / 255, (0, 0), 1.2)
    ref_idx = req.reference_slice; ref = prep(src[ref_idx, req.channel]); aligned = np.empty_like(src); transforms = []
    for i in range(z):
        if i == ref_idx: mat = np.eye(2, 3, dtype=np.float32); score = 1.0
        else:
            mat = np.eye(2, 3, dtype=np.float32)
            try: score, mat = cv2.findTransformECC(ref, prep(src[i, req.channel]), mat, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 250, 1e-6), None, 5)
            except cv2.error: score = 0.0
        fullmat = mat.copy(); fullmat[0, 2] /= factor; fullmat[1, 2] /= factor
        transforms.append({'slice': i, 'matrix': fullmat.tolist(), 'score': float(score)})
        for ch in range(c): aligned[i, ch] = cv2.warpAffine(src[i, ch], fullmat, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    s['aligned'] = aligned; s['transforms'] = transforms; s['reference'] = ref_idx; s['channel'] = req.channel
    picks = sorted(set([0, ref_idx, min(z-1, ref_idx+1)]))
    return {'id': req.id, 'transforms': transforms, 'alignedPreviews': [{'slice': i, 'channels': [png_data(aligned[i, ch]) for ch in range(c)]} for i in picks], 'reference': ref_idx, 'channel': req.channel}

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

