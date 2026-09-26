"""Multiresolution rigid normalized-correlation registration."""
from __future__ import annotations
import math
import numpy as np


def identity(): return np.array([1., 0., 0., 1., 0., 0.])

def compose(a, b):
    return np.array([
        a[0]*b[0]+a[2]*b[1], a[1]*b[0]+a[3]*b[1],
        a[0]*b[2]+a[2]*b[3], a[1]*b[2]+a[3]*b[3],
        a[0]*b[4]+a[2]*b[5]+a[4], a[1]*b[4]+a[3]*b[5]+a[5]
    ], dtype=np.float64)

def inverse(m):
    det=m[0]*m[3]-m[1]*m[2]
    return np.array([m[3]/det,-m[1]/det,-m[2]/det,m[0]/det,
        (m[2]*m[5]-m[3]*m[4])/det,(m[1]*m[4]-m[0]*m[5])/det])

def rigid(dx, dy, degrees, cx, cy):
    t=math.radians(degrees); c=math.cos(t); s=math.sin(t)
    return np.array([c,s,-s,c,cx+dx-c*cx+s*cy,cy+dy-s*cx-c*cy])

def sample(image, x, y):
    h,w=image['data'].shape
    if x < 0 or y < 0 or x >= w-1 or y >= h-1: return np.nan
    ix=int(math.floor(x)); iy=int(math.floor(y)); fx=x-ix; fy=y-iy; d=image['data']
    return (d[iy,ix]*(1-fx)+d[iy,ix+1]*fx)*(1-fy)+(d[iy+1,ix]*(1-fx)+d[iy+1,ix+1]*fx)*fy

def half(image):
    d=image['data']; h,w=d.shape; nh=(h+1)//2; nw=(w+1)//2
    out=np.zeros((nh,nw),np.float32); mask=np.zeros((nh,nw),np.uint8)
    source=image['mask']
    for y in range(nh):
        for x in range(nw):
            block=d[y*2:min(y*2+2,h),x*2:min(x*2+2,w)]
            mb=source[y*2:min(y*2+2,h),x*2:min(x*2+2,w)]
            out[y,x]=float(block.mean()); mask[y,x]=1 if np.all(mb) else 0
    return {'data':out,'mask':mask}

def centroid(image):
    d=image['data']; h,w=d.shape; border=np.concatenate((d[0,:],d[-1,:]))
    bg=float(np.median(border)); weights=np.abs(d-bg)*image['mask']; total=float(weights.sum())
    if total<=1e-5: return (w/2,h/2)
    yy,xx=np.indices((h,w),dtype=np.float32)
    return (float(((xx+.5)*weights).sum()/total),float(((yy+.5)*weights).sum()/total))

def correlation(fixed,moving,matrix,max_samples=14000):
    h,w=fixed['data'].shape; stride=max(1,int(math.floor(math.sqrt((h*w)/max_samples))))
    yy,xx=np.mgrid[1:h-1:stride,1:w-1:stride]; kf=fixed['mask'][yy,xx]>0
    eligible=int(kf.sum())
    if eligible==0:return -1.
    fixed_x=xx[kf]; fixed_y=yy[kf]; x=fixed_x+.5; y=fixed_y+.5
    inv=inverse(matrix); mx=inv[0]*x+inv[2]*y+inv[4]-.5; my=inv[1]*x+inv[3]*y+inv[5]-.5
    valid=(mx>=0)&(my>=0)&(mx<moving['data'].shape[1]-1)&(my<moving['data'].shape[0]-1)
    if not np.any(valid):return -1.
    x=x[valid]; y=y[valid]; mx=mx[valid]; my=my[valid]
    ix=np.floor(mx).astype(int); iy=np.floor(my).astype(int)
    mm=moving['mask']
    valid=mm[iy,ix].astype(bool)&mm[iy,ix+1].astype(bool)&mm[iy+1,ix].astype(bool)&mm[iy+1,ix+1].astype(bool)
    n=int(valid.sum())
    if n<32 or n<eligible*.45:return -1.
    x=x[valid]; y=y[valid]; mx=mx[valid]; my=my[valid]; ix=ix[valid]; iy=iy[valid]
    fx=mx-ix; fy=my-iy; d=moving['data']
    mv=(d[iy,ix]*(1-fx)+d[iy,ix+1]*fx)*(1-fy)+(d[iy+1,ix]*(1-fx)+d[iy+1,ix+1]*fx)*fy
    fvals=fixed['data'][fixed_y[valid],fixed_x[valid]]
    sf=float(fvals.sum()); sm=float(mv.sum()); sff=float(np.dot(fvals,fvals)); smm=float(np.dot(mv,mv)); sfm=float(np.dot(fvals,mv))
    vf=sff-sf*sf/n; vm=smm-sm*sm/n
    if vf/n<1e-7 or vm/n<1e-7:return -1.
    return float((sfm-sf*sm/n)/math.sqrt(vf*vm))

def optimize(fixed,moving,p,steps,max_iterations=35):
    h,w=fixed['data'].shape
    def score(q):return correlation(fixed,moving,rigid(q[0],q[1],q[2],w/2,h/2))
    best=score(p)
    for translation,angle in steps:
        for _ in range(max_iterations):
            changed=False
            for axis in range(3):
                candidate=p; value=best
                for sign in (-1,1):
                    q=list(p); q[axis]+=sign*(angle if axis==2 else translation)
                    if abs(q[2])>40 or abs(q[0])>w*.4 or abs(q[1])>h*.4:continue
                    v=score(q)
                    if v>value+1e-9:candidate=q; value=v
                if candidate is not p:p=candidate; best=value; changed=True
            if not changed:break
    return {'p':p,'score':best}

def register_rigid(fixed,moving):
    if fixed['data'].shape!=moving['data'].shape:raise ValueError('Proxy dimensions must match.')
    full_h,full_w=fixed['data'].shape
    h,w=full_h,full_w
    if min(w,h)<16:raise ValueError('Registration needs images at least 16 pixels across.')
    fp=[fixed]; mp=[moving]
    while max(fp[-1]['data'].shape)>80 and min(fp[-1]['data'].shape)>32:
        fp.append(half(fp[-1])); mp.append(half(mp[-1]))
    f=fp[-1]; m=mp[-1]; h,w=f['data'].shape; fc=centroid(f); mc=centroid(m); cx=w/2; cy=h/2
    seeds=[{'p':[0.,0.,0.],'score':correlation(f,m,identity())}]
    for angle in range(-30,31,5):
        r=math.radians(angle); c=math.cos(r); s=math.sin(r)
        dx=fc[0]-(c*(mc[0]-cx)-s*(mc[1]-cy)+cx)
        dy=fc[1]-(s*(mc[0]-cx)+c*(mc[1]-cy)+cy)
        for ox in (-6,0,6):
            for oy in (-6,0,6):
                p=[dx+ox,dy+oy,float(angle)]
                seeds.append({'p':p,'score':correlation(f,m,rigid(*p,cx,cy))})
    seeds.sort(key=lambda q:q['score'],reverse=True)
    best=sorted([optimize(f,m,s['p'],[(2,2),(1,1),(.5,.5),(.25,.2)]) for s in seeds[:4]],key=lambda q:q['score'],reverse=True)[0]
    for level in range(len(fp)-2,-1,-1):
        previous=fp[level+1]['data']; next_image=fp[level]['data']; ph,pw=previous.shape; nh,nw=next_image.shape
        matrix=rigid(*best['p'],pw/2,ph/2); angle=math.radians(best['p'][2]); c=math.cos(angle); s=math.sin(angle)
        nx=nw/2; ny=nh/2
        p=[matrix[4]*2-nx+c*nx-s*ny,matrix[5]*2-ny+s*nx+c*ny,best['p'][2]]
        best=optimize(fp[level],mp[level],p,[(1,.5),(.5,.2),(.25,.08),(.125,.025)])
    matrix=rigid(*best['p'],full_w/2,full_h/2)
    return {'matrix':matrix,'score':best['score'],'warning':'Low correlation — inspect overlay.' if best['score']<.7 else None}

def register_stack(images, masks, reference, max_side=768, source_dtypes=None):
    """Register adjacent slices outward from reference and accumulate rigid transforms."""
    z,h,w=images.shape; scale=min(1.,float(max_side)/max(h,w)); ph=max(1,math.ceil(h*scale)); pw=max(1,math.ceil(w*scale))
    proxies=[]
    for i in range(z):
        image=images[i].astype(np.float32)
        # TIFF unsigned 16-bit display conversion is fixed-range, with no per-image contrast stretch.
        source_dtype=np.dtype(source_dtypes[i]) if source_dtypes else images.dtype
        if source_dtype==np.uint16:image=np.rint(image/257.)
        elif source_dtype!=np.uint8:
            lo,hi=float(np.nanmin(image)),float(np.nanmax(image))
            image=np.clip(image-lo,0,hi-lo if hi>lo else 1)*(255./(hi-lo if hi>lo else 1))
        import cv2
        support=cv2.erode(masks[i].astype(np.uint8),np.ones((3,3),np.uint8),iterations=1)
        if (ph,pw)!=(h,w):
            proxy=cv2.resize(image,(pw,ph),interpolation=cv2.INTER_LINEAR)
            proxy_mask=cv2.resize(support,(pw,ph),interpolation=cv2.INTER_NEAREST)
        else:proxy=image; proxy_mask=support
        # Match the valid source support used for correlation and exclude the one-pixel source edge.
        proxies.append({'data':np.asarray(proxy,dtype=np.float32)/255.,'mask':proxy_mask.astype(np.uint8)})
    transforms=[None]*z; transforms[reference]=identity(); pairs=[]
    for i in range(reference-1,-1,-1):pairs.append((i,i+1))
    for i in range(reference+1,z):pairs.append((i,i-1))
    records=[]; failed=[]; failed_parent=set()
    for child,parent in pairs:
        score=-1.; warning=None
        try:
            result=register_rigid(proxies[parent],proxies[child]); pair=result['matrix'].copy(); score=result['score']; warning=result['warning']
            pair[4]/=scale; pair[5]/=scale
            if score<.35:raise ValueError('Insufficient image similarity or texture.')
            transforms[child]=compose(transforms[parent],pair)
            if parent in failed_parent:warning='Parent slice had a failed registration; inspect this slice.'
        except Exception as e:
            score=float(getattr(e,'score',score)); pair=identity(); transforms[child]=transforms[parent].copy(); warning=str(e)
            failed.append(child); failed_parent.add(child)
        if parent in failed_parent:failed_parent.add(child)
        records.append({'slice':child,'parent':parent,'matrix':transforms[child].tolist(),'pairMatrix':pair.tolist(),'score':float(score),'warning':warning})
    indexed={record['slice']:record for record in records}
    indexed[reference]={'slice':reference,'parent':None,'matrix':transforms[reference].tolist(),'pairMatrix':identity().tolist(),'score':1.,'warning':None}
    records=[indexed[i] for i in range(z)]
    return transforms,records,failed,scale

