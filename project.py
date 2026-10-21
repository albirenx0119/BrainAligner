"""Transform state, strict JSON persistence and streamed local export."""
from __future__ import annotations
import copy
import io
import json
import re
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from image_io import render, validate_canvas
from registration import check_cancel, compose, identity


def rigid_value(value):
    matrix = np.asarray(value,dtype=np.float64)
    if matrix.shape!=(6,) or not np.all(np.isfinite(matrix)):
        raise ValueError("変換行列の形式が不正です")
    rotation = np.array([[matrix[0],matrix[2]],[matrix[1],matrix[3]]])
    if not np.allclose(rotation.T @ rotation,np.eye(2),atol=1e-6,rtol=0) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6):
        raise ValueError("平行移動・回転以外を含む変換は読み込めません")
    return matrix


class Project:
    def __init__(self,sources,store):
        self.sources,self.store = sources,store
        self.canvas = (max(s.width for s in sources),max(s.height for s in sources))
        self.viewport = (0,0,*self.canvas)
        self.background = 'white'
        self.automatic = [identity() for _ in sources]
        self.manual = [identity() for _ in sources]
        self.qc = [None for _ in sources]
        self.settings = {}
        self.history = []
        self.crop_history = []

    def final(self,index):
        return compose(self.manual[index],self.automatic[index])

    def snapshot(self):
        return copy.deepcopy((self.viewport,self.background,self.automatic,self.manual,self.qc,self.settings,self.crop_history))

    def checkpoint(self):
        self.history.append(self.snapshot())
        self.history = self.history[-50:]

    def undo(self):
        if not self.history:
            return False
        (self.viewport,self.background,self.automatic,self.manual,self.qc,self.settings,self.crop_history) = self.history.pop()
        return True

    def apply_manual(self,matrix,start,end):
        if not 0<=start<len(self.sources) or not 0<=end<len(self.sources):
            raise ValueError("手動補正の適用範囲が不正です")
        matrix = rigid_value(matrix)
        self.checkpoint()
        for index in range(min(start,end),max(start,end)+1):
            self.manual[index] = compose(matrix,self.manual[index])

    def crop(self,viewport):
        x,y,w,h = map(int,viewport)
        validate_canvas(w,h)
        self.checkpoint()
        self.crop_history.append(self.viewport)
        self.viewport = (x,y,w,h)

    def undo_crop(self):
        if self.crop_history:
            self.checkpoint()
            self.viewport = self.crop_history.pop()
            return True
        return False

    def expand(self):
        x,y,w,h = self.viewport
        validate_canvas(w+200,h+200)
        self.checkpoint()
        self.viewport = (x-100,y-100,w+200,h+200)

    def to_dict(self):
        return {'format':'BrainAligner Desktop','version':1,'canvas':list(self.canvas),
                'viewport':list(self.viewport),'background':self.background,'settings':self.settings,
                'slices':[{'index':i,'name':s.name,'width':s.width,'height':s.height,
                           'automaticTransform':self.automatic[i].tolist(),
                           'manualTransform':self.manual[i].tolist(),
                           'finalTransform':self.final(i).tolist(),'qc':self.qc[i]}
                          for i,s in enumerate(self.sources)]}

    def load_dict(self,data):
        if not isinstance(data,dict) or data.get('format')!='BrainAligner Desktop' or data.get('version')!=1:
            raise ValueError("対応していない変換ファイルです")
        slices = data.get('slices')
        if data.get('canvas')!=list(self.canvas) or not isinstance(slices,list) or len(slices)!=len(self.sources):
            raise ValueError("画像枚数または共通キャンバスが一致しません")
        viewport = data.get('viewport')
        if not isinstance(viewport,list) or len(viewport)!=4 or any(type(v) is not int for v in viewport):
            raise ValueError("出力範囲が不正です")
        validate_canvas(viewport[2],viewport[3])
        if any(abs(v)>1_000_000 for v in viewport[:2]):
            raise ValueError("出力範囲の座標が大きすぎます")
        if data.get('background') not in ('white','black'):
            raise ValueError("背景色が不正です")
        automatic,manual,qc = [],[],[]
        for i,(source,row) in enumerate(zip(self.sources,slices)):
            if not isinstance(row,dict) or row.get('index')!=i or row.get('width')!=source.width or row.get('height')!=source.height:
                raise ValueError(f"切片 {i+1} の順番または画像サイズが一致しません")
            a,m = rigid_value(row.get('automaticTransform')),rigid_value(row.get('manualTransform'))
            if not np.allclose(compose(m,a),rigid_value(row.get('finalTransform')),atol=1e-6,rtol=0):
                raise ValueError("変換の合成が一致しません")
            record = row.get('qc')
            if record is not None:
                if not isinstance(record,dict) or not isinstance(record.get('score'),(int,float)) or not np.isfinite(record['score']):
                    raise ValueError("QC情報が不正です")
                if record.get('warning') is not None and not isinstance(record['warning'],str):
                    raise ValueError("QC警告が不正です")
            automatic.append(a); manual.append(m); qc.append(copy.deepcopy(record))
        settings = data.get('settings',{})
        if not isinstance(settings,dict):
            raise ValueError("計算設定が不正です")
        self.checkpoint()
        self.viewport = tuple(viewport)
        self.background = data['background']
        self.automatic,self.manual,self.qc,self.settings = automatic,manual,qc,copy.deepcopy(settings)
        self.crop_history = []


def export_names(sources,format):
    used = set()
    names = []
    for source in sources:
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',Path(source.name).stem).rstrip('. ') or 'image'
        if stem.upper() in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}:
            stem = '_'+stem
        name,n = f'{stem}.{format}',1
        while name.casefold() in used:
            n += 1
            name = f'{stem}__{n:03d}.{format}'
        used.add(name.casefold())
        names.append(name)
    return names


def export_project(project,destination,format='png',as_zip=False,progress=None,cancel=None):
    if format not in ('png','jpg'):
        raise ValueError("PNGまたはJPEGを指定してください")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    stem = 'BrainAligner_'+time.strftime('%Y%m%d_%H%M%S')
    target = destination/(stem+('.zip' if as_zip else ''))
    suffix = 2
    while target.exists() or target.with_name(target.name+'.partial').exists():
        target = destination/(f'{stem}_{suffix}'+('.zip' if as_zip else ''))
        suffix += 1
    manifest = project.to_dict()
    names = export_names(project.sources,format)
    manifest['export'] = {'format':format,'bitDepth':8,'files':names}
    payload = json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
    archive = None
    temporary = target.with_name(target.name+'.partial')
    if as_zip:
        archive = zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True)
    else:
        target.mkdir()
        (target/'INCOMPLETE.txt').write_text('書き出し中です。完了時にこのファイルは削除されます。',encoding='utf-8')
    try:
        for index,name in enumerate(names):
            check_cancel(cancel)
            rgb = render(project.store,index,project.canvas,project.final(index),project.viewport,project.background)
            image = Image.fromarray(rgb)
            output = io.BytesIO()
            image.save(output,format='PNG' if format=='png' else 'JPEG',**({'quality':95} if format=='jpg' else {}))
            check_cancel(cancel)
            if archive:
                archive.writestr(name,output.getvalue())
            else:
                (target/name).write_bytes(output.getvalue())
            if progress: progress(index+1,len(names),name)
        check_cancel(cancel)
        if archive:
            archive.writestr('transforms.json',payload)
            archive.close()
            archive = None
            temporary.rename(target)
        else:
            (target/'transforms.json').write_bytes(payload)
            (target/'INCOMPLETE.txt').unlink()
        return target
    finally:
        if archive is not None:
            archive.close()
        if as_zip and temporary.exists():
            temporary.unlink()
