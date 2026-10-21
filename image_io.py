"""Single-page image import and rendering from unchanged source pixels."""
from __future__ import annotations
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
import cv2
import numpy as np
import tifffile
from PIL import Image
from registration import Proxy, as_matrix, check_cancel

EXTENSIONS = {".tif",".tiff",".png",".jpg",".jpeg"}

def natural_key(name):
    return [(1,int(s)) if s.isdigit() else (0,s.casefold()) for s in re.split(r"(\d+)",str(name))]

@dataclass(frozen=True)
class Source:
    path: Path
    name: str
    width: int
    height: int
    dtype: str
    white_is_zero: bool = False

def validate_canvas(w,h):
    if w<1 or h<1 or max(w,h)>32767 or w*h>100_000_000:
        raise ValueError("画像サイズの上限は1辺32767画素・合計100MPです")

def inspect_source(path):
    path = Path(path).resolve()
    white_is_zero = False
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError(f"{path.name}: PNG、JPEG、単一ページTIFFを選んでください")
    if path.suffix.lower() in {".tif",".tiff"}:
        with tifffile.TiffFile(path) as tf:
            if len(tf.pages)!=1 or tf.is_ome:
                raise ValueError(f"{path.name}: 複数ページTIFF・OME-TIFFは読み込めません。1切片1ファイルにしてください")
            page = tf.pages[0]
            if page.subifds or page.samplesperpixel not in (1,3) or page.dtype.kind!="u" or page.dtype.itemsize not in (1,2):
                raise ValueError(f"{path.name}: 符号なし8/16-bitのグレーまたはRGBのみ対応しています")
            orientation = page.tags['Orientation'].value if 'Orientation' in page.tags else 1
            if int(page.photometric) not in (0,1,2) or int(orientation)!=1:
                raise ValueError(f"{path.name}: 画素形式または画像の向きが非対応です")
            if page.is_tiled or (page.planarconfig is not None and int(page.planarconfig)!=1):
                raise ValueError(f"{path.name}: strip形式・interleavedのTIFFを使用してください")
            w,h,dtype = page.imagewidth,page.imagelength,str(page.dtype)
            white_is_zero = int(page.photometric)==0
    else:
        with Image.open(path) as image:
            if getattr(image,"n_frames",1)!=1:
                raise ValueError(f"{path.name}: 複数フレーム画像は読み込めません")
            w,h = image.size
            dtype = "uint16" if image.mode in ("I;16","I;16B","I") else "uint8"
    validate_canvas(w,h)
    return Source(path,path.name,int(w),int(h),dtype,white_is_zero)

class ImageStore:
    def __init__(self,sources):
        self.sources,self.cache,self.lock = sources,OrderedDict(),RLock()

    def get(self,index):
        with self.lock:
            if index in self.cache:
                self.cache.move_to_end(index)
                return self.cache[index]
            source = self.sources[index]
            raw = cv2.imdecode(np.fromfile(source.path,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f"{source.name}: 画像をデコードできません")
            if raw.shape[:2]!=(source.height,source.width):
                raise ValueError(f"{source.name}: 元画像のサイズが変わりました")
            if raw.dtype not in (np.uint8,np.uint16):
                raise ValueError(f"{source.name}: 符号なし8/16-bit画像を使用してください")
            if raw.dtype==np.uint16:
                # The 16-bit TIFF decoder preserves MINISWHITE sample values;
                # its 8-bit path has already inverted them to display intensity.
                if source.white_is_zero:
                    raw = np.uint16(65535)-raw
                raw = np.floor(raw.astype(np.float64)/257.+.5).astype(np.uint8)
            if raw.ndim==2:
                rgb = np.repeat(raw[:,:,None],3,axis=2)
            elif raw.shape[2]==3:
                rgb = cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
            elif raw.shape[2]==4 and source.path.suffix.lower()=='.png':
                alpha = raw[:,:,3:4].astype(np.float32)/255
                rgb = np.rint(cv2.cvtColor(raw[:,:,:3],cv2.COLOR_BGR2RGB)*alpha+255*(1-alpha)).astype(np.uint8)
            else:
                raise ValueError(f"{source.name}: グレーまたはRGB画像を使用してください")
            self.cache[index] = rgb
            while len(self.cache)>3:
                self.cache.popitem(last=False)
            return rgb

def load_sources(paths,progress=None,cancel=None):
    paths = sorted(set(Path(p).resolve() for p in paths),key=lambda p:natural_key(p.as_posix()))
    if not paths:
        raise ValueError("対応する画像がありません")
    sources = []
    for index,path in enumerate(paths):
        check_cancel(cancel)
        sources.append(inspect_source(path))
        if progress: progress(index+1,len(paths),path.name)
    store = ImageStore(sources)
    for index in range(len(sources)):
        check_cancel(cancel)
        store.get(index)
        if progress: progress(index+1,len(sources),f"デコード確認: {sources[index].name}")
    validate_canvas(max(s.width for s in sources),max(s.height for s in sources))
    return sources,store

def make_proxy(store,index,canvas,limit,channel,background):
    source = store.sources[index]
    width,height = canvas
    scale = min(1.,limit/max(width,height))
    pw,ph = int(np.ceil(width*scale)),int(np.ceil(height*scale))
    left,top = (width-source.width)//2,(height-source.height)//2
    yy,xx = np.mgrid[:ph,:pw]
    sx,sy = (xx+.5)/scale-left-.5,(yy+.5)/scale-top-.5
    x0,y0 = np.floor(sx).astype(int),np.floor(sy).astype(int)
    fx,fy = sx-x0,sy-y0
    rgb = store.get(index)
    def tap(x,y):
        return rgb[np.clip(y,0,source.height-1),np.clip(x,0,source.width-1)].astype(np.float64)
    sampled = ((tap(x0,y0)*(1-fx[:,:,None])+tap(x0+1,y0)*fx[:,:,None])*(1-fy[:,:,None])
               +(tap(x0,y0+1)*(1-fx[:,:,None])+tap(x0+1,y0+1)*fx[:,:,None])*fy[:,:,None])
    outside = (sx<-.5)|(sy<-.5)|(sx>=source.width-.5)|(sy>=source.height-.5)
    sampled[outside] = 255 if background=='white' else 0
    sampled = np.floor(sampled+.5)
    if channel=='Gray':
        values = (sampled[:,:,0]*.2126+sampled[:,:,1]*.7152+sampled[:,:,2]*.0722)/255
    else:
        values = sampled[:,:,{'Red':0,'Green':1,'Blue':2}[channel]]/255
    mask = ((xx+.5)/scale>=left+1)&((xx+.5)/scale<left+source.width-1)
    mask &= ((yy+.5)/scale>=top+1)&((yy+.5)/scale<top+source.height-1)
    return Proxy(values.astype(np.float32),mask),scale

def render(store,index,canvas,transform,viewport,background,scale=1.):
    vx,vy,vw,vh = viewport
    source = store.sources[index]
    pad = np.eye(3)
    pad[:2,2] = [(canvas[0]-source.width)//2,(canvas[1]-source.height)//2]
    output = np.array([[scale,0,-vx*scale],[0,scale,-vy*scale],[0,0,1.]])
    edge_matrix = output @ as_matrix(transform) @ pad
    pixel_matrix = edge_matrix.copy()
    pixel_matrix[:2,2] += edge_matrix[:2,:2] @ np.array([.5,.5])-.5
    size = (max(1,round(vw*scale)),max(1,round(vh*scale)))
    fill = 255 if background=='white' else 0
    return cv2.warpAffine(store.get(index),pixel_matrix[:2],size,flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(fill,fill,fill))
