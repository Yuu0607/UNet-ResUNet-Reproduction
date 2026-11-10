import os, gc, copy, random, math
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

ROOT='/kaggle/input/competitions/2026task3/detection'
WORK='/kaggle/working'
OUT=f'{WORK}/fovea_v12_unet'
MODEL_DIR=f'{OUT}/models'
os.makedirs(MODEL_DIR,exist_ok=True)
FOLDS_CSV=f'{WORK}/fovea_v4_v8_fold_diagnostics.csv'
GT_CSV=f'{ROOT}/fovea_localization_train_GT.csv'
SEEDS=[42,2026,3407]
EPOCHS=80
SNAP_EPOCHS=(60,70,80)
BATCH_SIZE=4
NUM_WORKERS=2
LR=3e-4
EMA_DECAY=0.995
SIZE=512
HM=128
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
cv2.setNumThreads(0)

# These columns and 5-fold assignments were verified in earlier experiments.
fold_df=pd.read_csv(FOLDS_CSV)[['ID','fold','True_X','True_Y']].copy()
assert len(fold_df)==80 and fold_df.ID.nunique()==80 and fold_df.fold.nunique()==5
fold_df.ID=fold_df.ID.astype(int)
fold_df.fold=fold_df.fold.astype(int)
assert fold_df.fold.value_counts().min()==16

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_rgb(path):
    bgr=cv2.imread(path)
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

CLAHE=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
YY,XX=np.meshgrid(np.linspace(0,1,SIZE,dtype=np.float32),np.linspace(0,1,SIZE,dtype=np.float32),indexing='ij')

def preprocess(img):
    img=cv2.resize(img,(SIZE,SIZE),interpolation=cv2.INTER_AREA)
    rgb=img.astype(np.float32)/255.0
    g=CLAHE.apply(img[:,:,1]).astype(np.float32)/255.0
    six=np.concatenate([rgb,g[...,None],XX[...,None],YY[...,None]],axis=2)
    six[:,:,:3]=(six[:,:,:3]-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    six[:,:,3]=(six[:,:,3]-.5)/.25
    six[:,:,4:]=(six[:,:,4:]-.5)/.5
    return np.ascontiguousarray(six.transpose(2,0,1),dtype=np.float32)

class FoveaDS(Dataset):
    def __init__(self,rows,train=False):
        self.rows=rows.reset_index(drop=True); self.train=train
        self.cache=[]
        for r in self.rows.itertuples():
            p=f'{ROOT}/train/{int(r.ID):04d}.jpg'
            img=read_rgb(p); h,w=img.shape[:2]
            self.cache.append((preprocess(img),w,h))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows.iloc[i]; arr,w,h=self.cache[i]
        x=float(row.True_X)/w; y=float(row.True_Y)/h
        arr=arr.copy()
        if self.train:
            if random.random()<.5:
                arr=np.ascontiguousarray(arr[:,:,::-1]); x=1-x
                # Flip X positional channel to match new spatial positions.
                arr[4]=-arr[4]
            if random.random()<.5:
                # Mild photometric augmentation, only RGB/CLAHE channel.
                alpha=random.uniform(.9,1.1)
                arr[:4]=arr[:4]*alpha
        return torch.from_numpy(np.ascontiguousarray(arr)),torch.tensor([x,y],dtype=torch.float32),int(row.ID)

class ConvBlock(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.GroupNorm(min(8,co),co),nn.SiLU(inplace=True),nn.Conv2d(co,co,3,padding=1,bias=False),nn.GroupNorm(min(8,co),co),nn.SiLU(inplace=True))
    def forward(self,x):return self.net(x)

class V12UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0=ConvBlock(6,16) # 512
        self.e1=ConvBlock(16,32) # 256
        self.e2=ConvBlock(32,64) # 128
        self.e3=ConvBlock(64,96) # 64
        self.mid=ConvBlock(96,128) # 32
        self.d3=ConvBlock(128+96,96)
        self.d2=ConvBlock(96+64,64) # 128
        self.heat=nn.Conv2d(64,1,1)
        self.global_head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2),nn.Sigmoid())
    def forward(self,x):
        a=self.e0(x)
        b=self.e1(F.avg_pool2d(a,2))
        c=self.e2(F.avg_pool2d(b,2))
        d=self.e3(F.avg_pool2d(c,2))
        m=self.mid(F.avg_pool2d(d,2))
        z=self.d3(torch.cat([F.interpolate(m,size=d.shape[-2:],mode='bilinear',align_corners=False),d],dim=1))
        z=self.d2(torch.cat([F.interpolate(z,size=c.shape[-2:],mode='bilinear',align_corners=False),c],dim=1))
        logits=self.heat(z).flatten(1)
        probs=F.softmax(logits,dim=1)
        grid=torch.linspace(0,1,HM,device=x.device,dtype=x.dtype)
        yy,xx=torch.meshgrid(grid,grid,indexing='ij')
        heat_xy=torch.stack([(probs*xx.flatten()).sum(1),(probs*yy.flatten()).sum(1)],dim=1)
        global_xy=self.global_head(m)
        final_xy=.8*heat_xy+.2*global_xy
        return final_xy,heat_xy,global_xy,logits

def objective(outputs,truth):
    final,heat,glob,logits=outputs
    grid=torch.linspace(0,1,HM,device=truth.device)
    yy,xx=torch.meshgrid(grid,grid,indexing='ij')
    dist=(xx[None]-truth[:,0,None,None])**2+(yy[None]-truth[:,1,None,None])**2
    target=F.softmax((-dist/(2*(2.5/HM)**2)).flatten(1),dim=1)
    logp=F.log_softmax(logits,dim=1)
    heat_loss=-(target*logp).sum(dim=1).mean()
    # Coordinate losses are dimensionless; heatmap cross entropy supplies spatial supervision.
    coord=F.smooth_l1_loss(final,truth,beta=.02)
    aux=F.smooth_l1_loss(glob,truth,beta=.02)
    return heat_loss + 12*coord + 3*aux

class EMA:
    def __init__(self,model):
        self.model=copy.deepcopy(model).eval()
        for p in self.model.parameters():p.requires_grad_(False)
    @torch.no_grad()
    def update(self,model):
        for a,b in zip(self.model.parameters(),model.parameters()):a.lerp_(b.detach(),1-EMA_DECAY)
        # GroupNorm has no running statistics; copy any buffers for completeness.
        for a,b in zip(self.model.buffers(),model.buffers()):a.copy_(b)

def predict(model,loader,tta=True):
    model.eval(); records=[]
    with torch.inference_mode():
        for xx,yy,ids in loader:
            xx=xx.to(DEVICE)
            with torch.autocast(device_type='cuda',enabled=(DEVICE=='cuda')):
                p=model(xx)[0].float()
                if tta:
                    flipped=torch.flip(xx,dims=[3]).clone()
                    flipped[:,4]=-flipped[:,4]  # X position channel
                    q=model(flipped)[0].float()
                    q[:,0]=1-q[:,0]
                    p=(p+q)/2
            for idx,xy in zip(ids.numpy(),p.cpu().numpy()):
                records.append((int(idx),float(xy[0]),float(xy[1])))
    return pd.DataFrame(records,columns=['ID','norm_x','norm_y'])

def make_loader(ds,shuffle=False):
    return DataLoader(ds,batch_size=BATCH_SIZE,shuffle=shuffle,num_workers=NUM_WORKERS,pin_memory=(DEVICE=='cuda'),persistent_workers=NUM_WORKERS>0)

# Load test images and dimensions before training, fail early on missing files.
TEST_FILES=sorted([x for x in os.listdir(f'{ROOT}/test') if x.lower().endswith(('.jpg','.jpeg','.png'))])
assert len(TEST_FILES)==20
TEST_IMAGES={}; DIMS={}
for f in TEST_FILES:
    i=int(os.path.splitext(f)[0]); image=read_rgb(f'{ROOT}/test/{f}')
    h,w=image.shape[:2]; DIMS[i]=(w,h); TEST_IMAGES[i]=preprocess(image)
assert len(TEST_IMAGES)==20

class TestDS(Dataset):
    def __init__(self):self.ids=sorted(TEST_IMAGES)
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        idx=self.ids[i]
        return torch.from_numpy(TEST_IMAGES[idx]),torch.zeros(2),idx

test_loader=make_loader(TestDS())
train_dims={}
for row in fold_df.itertuples():
    im=cv2.imread(f'{ROOT}/train/{int(row.ID):04d}.jpg')
    if im is None:raise RuntimeError(f'Missing train {row.ID}')
    train_dims[int(row.ID)]=(im.shape[1],im.shape[0])

all_oof=[]; all_test=[]; histories=[]
print('V12',DEVICE,'Seeds',SEEDS,'Epochs',EPOCHS,'15 models, EMA snapshots',SNAP_EPOCHS,flush=True)

for seed in SEEDS:
  for fold in sorted(fold_df.fold.unique()):
    seed_all(seed*10+int(fold))
    train_df=fold_df[fold_df.fold!=fold].copy()
    val_df=fold_df[fold_df.fold==fold].copy()
    train_loader=make_loader(FoveaDS(train_df,train=True),shuffle=True)
    val_loader=make_loader(FoveaDS(val_df,train=False))
    model=V12UNet().to(DEVICE)
    ema=EMA(model)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=LR*.05)
    scaler=torch.amp.GradScaler('cuda',enabled=DEVICE=='cuda')
    snapshots=[]
    for epoch in range(1,EPOCHS+1):
        model.train(); losses=[]
        for xx,yy,_ in train_loader:
            xx=xx.to(DEVICE,non_blocking=True);yy=yy.to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=DEVICE=='cuda'):
                loss=objective(model(xx),yy)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt);scaler.update()
            ema.update(model)
            losses.append(float(loss.item()))
        sched.step()
        histories.append(dict(seed=seed,fold=int(fold),epoch=epoch,loss=np.mean(losses)))
        if epoch in SNAP_EPOCHS:
            snapshots.append({k:v.detach().cpu().clone() for k,v in ema.model.state_dict().items()})
        if epoch%10==0:
            print(f'Seed {seed} Fold {fold} Epoch {epoch}/{EPOCHS} loss {np.mean(losses):.4f}',flush=True)
    assert len(snapshots)==3
    averaged={}
    for key in snapshots[0]:
        if torch.is_floating_point(snapshots[0][key]):
            averaged[key]=torch.stack([s[key].float() for s in snapshots]).mean(0).to(snapshots[0][key].dtype)
        else:averaged[key]=snapshots[-1][key].clone()
    ema.model.load_state_dict(averaged,strict=True)
    path=f'{MODEL_DIR}/v12_seed{seed}_fold{fold}.pth'
    torch.save({'model_state_dict':averaged,'seed':seed,'fold':int(fold),'epochs':EPOCHS,'snapshots':SNAP_EPOCHS,'input_channels':6,'image_size':SIZE},path)
    val_pred=predict(ema.model,val_loader,tta=True)
    val_pred['seed']=seed;val_pred['fold']=int(fold)
    for i,row in val_pred.iterrows():
        w,h=train_dims[int(row.ID)]
        val_pred.loc[i,'Pred_X']=row.norm_x*w
        val_pred.loc[i,'Pred_Y']=row.norm_y*h
    all_oof.append(val_pred)
    test_pred=predict(ema.model,test_loader,tta=True)
    test_pred['seed']=seed;test_pred['fold']=int(fold)
    for i,row in test_pred.iterrows():
        w,h=DIMS[int(row.ID)]
        test_pred.loc[i,'Pred_X']=row.norm_x*w
        test_pred.loc[i,'Pred_Y']=row.norm_y*h
    all_test.append(test_pred)
    pd.concat(all_oof,ignore_index=True).to_csv(f'{OUT}/oof_partial.csv',index=False)
    pd.concat(all_test,ignore_index=True).to_csv(f'{OUT}/test_partial.csv',index=False)
    pd.DataFrame(histories).to_csv(f'{OUT}/training_history.csv',index=False)
    print('SAVED',path,flush=True)
    del model,ema,opt,sched,scaler,snapshots,averaged,train_loader,val_loader
    gc.collect()
    if DEVICE=='cuda':torch.cuda.empty_cache()

# OOF: each image has one held-out prediction per seed (3 total), never average other folds.
oof_all=pd.concat(all_oof,ignore_index=True)
assert oof_all.groupby('ID').size().eq(len(SEEDS)).all()
oof=oof_all.groupby('ID',as_index=False)[['Pred_X','Pred_Y']].mean().merge(fold_df,on='ID',validate='one_to_one')
oof_rmse=np.sqrt(np.mean(np.square(oof[['Pred_X','Pred_Y']].to_numpy()-oof[['True_X','True_Y']].to_numpy())))
oof.to_csv(f'{OUT}/v12_oof.csv',index=False)
oof_all.to_csv(f'{OUT}/v12_oof_all_seeds.csv',index=False)

# Test: all 15 models averaged, unlike OOF.
test_all=pd.concat(all_test,ignore_index=True)
assert test_all.groupby('ID').size().eq(15).all()
test=test_all.groupby('ID',as_index=False)[['Pred_X','Pred_Y']].mean().sort_values('ID')
test.to_csv(f'{OUT}/v12_test_predictions.csv',index=False)
test_all.to_csv(f'{OUT}/v12_test_all_models.csv',index=False)
rows=[]
for r in test.itertuples():
    w,h=DIMS[int(r.ID)]
    x=float(np.clip(r.Pred_X,0,w-1));y=float(np.clip(r.Pred_Y,0,h-1))
    rows.extend([{'ImageID':f'{r.ID}_Fovea_X','value':x},{'ImageID':f'{r.ID}_Fovea_Y','value':y}])
submission=pd.DataFrame(rows)
assert submission.shape==(40,2) and submission.ImageID.is_unique and np.isfinite(submission.value).all()
submission.to_csv(f'{WORK}/submission_fovea_v12_unet.csv',index=False)
print('\n'+'='*65)
print('V12 OOF RMSE:',round(float(oof_rmse),4))
print('Previous V4+V8 OOF baseline: 23.5195')
print('OOF:',f'{OUT}/v12_oof.csv')
print('Submission:',f'{WORK}/submission_fovea_v12_unet.csv')
print(submission.head(6).to_string(index=False))
