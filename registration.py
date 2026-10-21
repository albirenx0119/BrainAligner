"""Masked, multiresolution intensity correlation for rigid registration."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

class Cancelled(Exception):
    pass

class RegistrationFailure(ValueError):
    def __init__(self, message, score=-1.):
        super().__init__(message)
        self.score = float(score)

def check_cancel(event=None):
    if event is not None and event.is_set():
        raise Cancelled("処理を中止しました")

def identity():
    return np.array([1., 0., 0., 1., 0., 0.])

def as_matrix(m):
    a,b,c,d,e,f = m
    return np.array([[a,c,e],[b,d,f],[0.,0.,1.]])

def from_matrix(m):
    return np.asarray(m,dtype=np.float64)[:2,:].T.reshape(-1)

def compose(a,b):
    return from_matrix(as_matrix(a) @ as_matrix(b))

def inverse(m):
    return from_matrix(np.linalg.inv(as_matrix(m)))

def rigid(dx,dy,degrees,cx,cy):
    theta = math.radians(degrees)
    c,s = math.cos(theta), math.sin(theta)
    return np.array([c,s,-s,c,cx+dx-c*cx+s*cy,cy+dy-s*cx-c*cy])

@dataclass
class Proxy:
    data: np.ndarray
    mask: np.ndarray

def half(image):
    h,w = image.data.shape
    nh,nw = (h+1)//2,(w+1)//2
    sums = np.zeros((nh,nw),np.float64)
    counts = np.zeros((nh,nw),np.uint8)
    support = np.ones((nh,nw),bool)
    for dy in (0,1):
        for dx in (0,1):
            block = image.data[dy::2,dx::2]
            bh,bw = block.shape
            sums[:bh,:bw] += block
            counts[:bh,:bw] += 1
            support[:bh,:bw] &= image.mask[dy::2,dx::2].astype(bool)
    return Proxy((sums/counts).astype(np.float32),support)

def centroid(image):
    d = image.data.astype(np.float64)
    border = np.sort(np.concatenate((d[0],d[-1])))
    weights = np.abs(d-border[len(border)//2])*image.mask
    total = weights.sum()
    h,w = d.shape
    if total <= 1e-5:
        return w/2,h/2
    return float(weights.sum(axis=0) @ (np.arange(w)+.5)/total), float(weights.sum(axis=1) @ (np.arange(h)+.5)/total)

class Correlation:
    def __init__(self,fixed,moving):
        self.moving = moving
        h,w = fixed.data.shape
        step = max(1,int(math.sqrt(h*w/14000)))
        yy,xx = np.mgrid[1:h-1:step,1:w-1:step]
        keep = fixed.mask[yy,xx].astype(bool)
        self.x,self.y = xx[keep]+.5,yy[keep]+.5
        self.values = fixed.data[yy[keep],xx[keep]].astype(np.float64)

    def __call__(self,transform):
        a,b,c,d,e,f = inverse(transform)
        mx = a*self.x+c*self.y+e-.5
        my = b*self.x+d*self.y+f-.5
        image = self.moving.data
        h,w = image.shape
        valid = (mx>=0)&(my>=0)&(mx<w-1)&(my<h-1)
        fixed_values = self.values[valid]
        mx,my = mx[valid],my[valid]
        ix,iy = np.floor(mx).astype(int),np.floor(my).astype(int)
        mask = self.moving.mask
        support = (mask[iy,ix]&mask[iy,ix+1]&mask[iy+1,ix]&mask[iy+1,ix+1]).astype(bool)
        n = int(support.sum())
        if n<32 or n<self.values.size*.45:
            return -1.
        fvals = fixed_values[support]
        ix,iy = ix[support],iy[support]
        fx,fy = mx[support]-ix,my[support]-iy
        mvals = ((image[iy,ix]*(1-fx)+image[iy,ix+1]*fx)*(1-fy)
                 +(image[iy+1,ix]*(1-fx)+image[iy+1,ix+1]*fx)*fy)
        fvals = fvals-fvals.mean()
        mvals = mvals-mvals.mean()
        vf,vm = float(np.sum(fvals*fvals)),float(np.sum(mvals*mvals))
        if vf/n<1e-7 or vm/n<1e-7:
            return -1.
        return float(np.sum(fvals*mvals)/math.sqrt(vf*vm))

def optimize(fixed,moving,initial,steps,cancel=None):
    h,w = fixed.data.shape
    metric = Correlation(fixed,moving)
    p = list(initial)
    best = metric(rigid(*p,w/2,h/2))
    for translation_step,angle_step in steps:
        for _ in range(35):
            check_cancel(cancel)
            changed = False
            for axis in range(3):
                candidate,value = p,best
                for sign in (-1,1):
                    q = p.copy()
                    q[axis] += sign*(angle_step if axis==2 else translation_step)
                    if abs(q[2])>40 or abs(q[0])>w*.4 or abs(q[1])>h*.4:
                        continue
                    score = metric(rigid(*q,w/2,h/2))
                    if score>value+1e-9:
                        candidate,value = q,score
                if candidate is not p:
                    p,best,changed = candidate,value,True
            if not changed:
                break
    return p,best

def register_rigid(fixed,moving,cancel=None):
    if fixed.data.shape!=moving.data.shape or min(fixed.data.shape)<16:
        raise ValueError("計算用画像は同じサイズで縦横とも16画素以上必要です")
    fp,mp = [fixed],[moving]
    while max(fp[-1].data.shape)>80 and min(fp[-1].data.shape)>32:
        fp.append(half(fp[-1])); mp.append(half(mp[-1]))
    f,m = fp[-1],mp[-1]
    h,w = f.data.shape
    fc,mc = centroid(f),centroid(m)
    cx,cy = w/2,h/2
    metric = Correlation(f,m)
    seeds = [([0.,0.,0.],metric(identity()))]
    for angle in range(-30,31,5):
        check_cancel(cancel)
        r = math.radians(angle)
        c,s = math.cos(r),math.sin(r)
        dx = fc[0]-(c*(mc[0]-cx)-s*(mc[1]-cy)+cx)
        dy = fc[1]-(s*(mc[0]-cx)+c*(mc[1]-cy)+cy)
        for ox in (-6,0,6):
            for oy in (-6,0,6):
                p = [dx+ox,dy+oy,float(angle)]
                seeds.append((p,metric(rigid(*p,cx,cy))))
    seeds.sort(key=lambda item:item[1],reverse=True)
    candidates = [optimize(f,m,seed[0],[(2,2),(1,1),(.5,.5),(.25,.2)],cancel) for seed in seeds[:4]]
    p,score = max(candidates,key=lambda item:item[1])
    for level in range(len(fp)-2,-1,-1):
        ph,pw = fp[level+1].data.shape
        nh,nw = fp[level].data.shape
        matrix = rigid(*p,pw/2,ph/2)
        c,s = math.cos(math.radians(p[2])),math.sin(math.radians(p[2]))
        nx,ny = nw/2,nh/2
        initial = [matrix[4]*2-nx+c*nx-s*ny,matrix[5]*2-ny+s*nx+c*ny,p[2]]
        p,score = optimize(fp[level],mp[level],initial,[(1,.5),(.5,.2),(.25,.08),(.125,.025)],cancel)
    if score<.35:
        raise RegistrationFailure("画像の類似度または模様が不足しています",score)
    h,w = fixed.data.shape
    return rigid(*p,w/2,h/2),score

def register_series(count,reference,get_proxy,progress=None,cancel=None):
    transforms = [identity() for _ in range(count)]
    records = [None]*count
    records[reference] = {"score":1.,"warning":None,"parent":None}
    pairs = [(i,i+1) for i in range(reference-1,-1,-1)]+[(i,i-1) for i in range(reference+1,count)]
    failed = set()
    successes = 0
    parent_cache = {}
    for number,(child,parent) in enumerate(pairs,1):
        check_cancel(cancel)
        if progress:
            progress(number-1,len(pairs),f"切片 {child+1} → {parent+1}")
        fixed,_ = parent_cache.get(parent) or get_proxy(parent)
        moving,scale = get_proxy(child)
        parent_cache = {child:(moving,scale)}
        score,warning = -1.,None
        try:
            pair,score = register_rigid(fixed,moving,cancel)
            pair[4:6] /= scale
            transforms[child] = compose(transforms[parent],pair)
            successes += 1
            if score<.7:
                warning = "相関が低いため重ね合わせを確認してください"
        except RegistrationFailure as exc:
            score,warning = exc.score,str(exc)
            failed.add(child)
            transforms[child] = transforms[parent].copy()
        if parent in failed:
            failed.add(child)
            warning = (warning+" / " if warning else "")+"先行する隣接ペアで失敗しています"
        records[child] = {"score":score,"warning":warning,"parent":parent}
    if pairs and not successes:
        raise RegistrationFailure("全ペアの位置合わせに失敗しました。画像やチャンネルを確認してください")
    check_cancel(cancel)
    return transforms,records
