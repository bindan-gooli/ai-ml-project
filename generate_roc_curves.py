"""
Regenerate ROC curves for Datasets 1, 2, and 3.
Uses the same model configs as the main pipeline.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import (IsolationForest, HistGradientBoostingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PREPROCESSED_DIR    = "preprocessed_data"
BATCH_SIZE          = 256
EPOCHS              = 1
PRIMARY_SAMPLE_SIZE = 200_000
DATASETS            = [1, 2, 3]   # only generate for these

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── Same model definitions as pipeline ──────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(16,32),nn.ReLU(),nn.Linear(32,d))
    def forward(self,x): return self.dec(self.enc(x))

class VAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1=nn.Linear(d,32); self.mu=nn.Linear(32,16); self.lv=nn.Linear(32,16)
        self.fc3=nn.Linear(16,32); self.fc4=nn.Linear(32,d)
    def encode(self,x):
        h=torch.relu(self.fc1(x)); return self.mu(h),self.lv(h)
    def reparameterize(self,mu,lv):
        return mu+torch.randn_like(torch.exp(0.5*lv))*torch.exp(0.5*lv)
    def decode(self,z): return self.fc4(torch.relu(self.fc3(z)))
    def forward(self,x):
        mu,lv=self.encode(x); return self.decode(self.reparameterize(mu,lv)),mu,lv

class DAGMM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(d,32),nn.Tanh(),nn.Linear(32,16),nn.Tanh(),nn.Linear(16,8))
        self.dec=nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,32),nn.Tanh(),nn.Linear(32,d))
        self.est=nn.Sequential(nn.Linear(10,16),nn.Tanh(),nn.Dropout(0.5),nn.Linear(16,4),nn.Softmax(dim=1))
    def forward(self,x):
        zc=self.enc(x); xp=self.dec(zc)
        ed=torch.nn.functional.pairwise_distance(x,xp,p=2).unsqueeze(1)
        cs=torch.nn.functional.cosine_similarity(x,xp,dim=1).unsqueeze(1)
        z=torch.cat([zc,ed,cs],dim=1); return zc,xp,self.est(z),z

class DeepSVDD(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,16))
    def forward(self,x): return self.enc(x)

def train_model(model, X, mtype='ae'):
    model=model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    tx=torch.Tensor(X).to(device)
    loader=DataLoader(TensorDataset(tx),batch_size=BATCH_SIZE,shuffle=True)
    c=None
    if mtype=='deep_svdd':
        model.eval()
        with torch.no_grad(): c=torch.mean(model(tx[:50000]),dim=0).detach()
        model.train()
    for _ in range(EPOCHS):
        for (bx,) in loader:
            opt.zero_grad()
            if mtype=='vae':
                rx,mu,lv=model(bx)
                loss=(nn.functional.mse_loss(rx,bx,reduction='sum')
                      -0.5*torch.sum(1+lv-mu.pow(2)-lv.exp()))
            elif mtype=='dagmm':
                zc,xp,gamma,z=model(bx)
                loss_r=torch.mean(torch.sum((bx-xp)**2,dim=1))
                gs=torch.sum(gamma,dim=0); phi=gs/gamma.size(0)
                mu_g=torch.sum(gamma.unsqueeze(-1)*z.unsqueeze(1),dim=0)/gs.unsqueeze(-1)
                zm=z.unsqueeze(1)-mu_g.unsqueeze(0)
                cov=(torch.sum(gamma.unsqueeze(-1).unsqueeze(-1)*zm.unsqueeze(-1)*zm.unsqueeze(-2),dim=0)
                     /gs.unsqueeze(-1).unsqueeze(-1))
                cov+=torch.eye(z.size(1)).to(device)*1e-6
                try: ci=torch.inverse(cov); dc=torch.det(cov)
                except: ci=torch.pinverse(cov); dc=torch.ones(cov.shape[0]).to(device)
                et=torch.exp(-0.5*torch.sum(torch.sum(zm.unsqueeze(-1)*ci.unsqueeze(0),dim=-2)*zm,dim=-1))
                dc=torch.clamp(dc,min=1e-12)
                pz=torch.clamp(torch.sum(phi.unsqueeze(0)*et/torch.sqrt((2*np.pi)**z.size(1)*dc.unsqueeze(0)),dim=1),min=1e-12)
                loss=loss_r+0.1*torch.mean(-torch.log(pz))+0.005*torch.sum(1.0/torch.diagonal(cov,dim1=1,dim2=2))
            elif mtype=='deep_svdd':
                loss=torch.mean(torch.sum((model(bx)-c)**2,dim=1))
            else:
                rx=model(bx); loss=nn.MSELoss()(rx,bx)
            loss.backward(); opt.step()
    if mtype=='dagmm':
        model.eval()
        with torch.no_grad():
            sub=tx[:50000]; zc,xp,gamma,z=model(sub)
            gs=torch.sum(gamma,dim=0); phi=gs/gamma.size(0)
            mu_g=torch.sum(gamma.unsqueeze(-1)*z.unsqueeze(1),dim=0)/gs.unsqueeze(-1)
            zm=z.unsqueeze(1)-mu_g.unsqueeze(0)
            cov=(torch.sum(gamma.unsqueeze(-1).unsqueeze(-1)*zm.unsqueeze(-1)*zm.unsqueeze(-2),dim=0)
                 /gs.unsqueeze(-1).unsqueeze(-1))+torch.eye(z.size(1)).to(device)*1e-6
        return model,(phi,mu_g,cov)
    if mtype=='deep_svdd': return model,c
    return model

def get_scores(model, X, mtype='ae', extra=None):
    model.eval()
    tx=torch.Tensor(X).to(device)
    loader=DataLoader(TensorDataset(tx),batch_size=BATCH_SIZE*4,shuffle=False)
    errs=[]
    with torch.no_grad():
        for (bx,) in loader:
            if mtype=='vae':
                rx,mu,lv=model(bx); e=torch.mean((bx-rx)**2,dim=1)
            elif mtype=='dagmm':
                phi,mu_g,cov=extra; zc,xp,gamma,z=model(bx)
                try: ci=torch.inverse(cov); dc=torch.det(cov)
                except: ci=torch.pinverse(cov); dc=torch.ones(cov.shape[0]).to(device)
                zm=z.unsqueeze(1)-mu_g.unsqueeze(0); dc=torch.clamp(dc,min=1e-12)
                et=torch.exp(-0.5*torch.sum(torch.sum(zm.unsqueeze(-1)*ci.unsqueeze(0),dim=-2)*zm,dim=-1))
                pz=torch.clamp(torch.sum(phi.unsqueeze(0)*et/torch.sqrt((2*np.pi)**z.size(1)*dc.unsqueeze(0)),dim=1),min=1e-12)
                e=-torch.log(pz)
            elif mtype=='deep_svdd':
                c=extra; e=torch.sum((model(bx)-c)**2,dim=1)
            else:
                rx=model(bx); e=torch.mean((bx-rx)**2,dim=1)
            errs.append(e.cpu().numpy())
    return np.concatenate(errs)

COLORS = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
          '#42d4f4','#f032e6','#bfef45','#fabed4','#469990',
          '#dcbeff','#9A6324','#fffac8','#800000','#aaffc3',
          '#808000','#ffd8b1','#000075','#a9a9a9','#ffffff',
          '#000000','#e6beff','#ffe119','#4169E1','#FF6347']

sec_configs = [
    ('M6','Isolation Forest','if'),
    ('M7','Autoencoder','ae'),
    ('M8','VAE','vae'),
    ('M9','DAGMM','dagmm'),
    ('M10','Deep SVDD','deep_svdd'),
]
prim_configs = [
    ('M1','HistGradBoost', lambda: HistGradientBoostingClassifier(max_iter=100,random_state=42)),
    ('M2','ExtraTrees',    lambda: ExtraTreesClassifier(n_estimators=100,random_state=42,n_jobs=4)),
    ('M3','GradBoost',     lambda: GradientBoostingClassifier(n_estimators=100,random_state=42)),
    ('M4','RandomForest',  lambda: RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=4)),
    ('M5','MLP',           lambda: MLPClassifier(hidden_layer_sizes=(128,64),max_iter=100,random_state=42)),
]

os.makedirs("roc_curves", exist_ok=True)

for ds_idx in DATASETS:
    npz = os.path.join(PREPROCESSED_DIR, f"dataset_{ds_idx}.npz")
    if not os.path.exists(npz):
        print(f"[SKIP] {npz} not found"); continue

    data = np.load(npz)
    X_train,X_test,y_train,y_test = data['X_train'],data['X_test'],data['y_train'],data['y_test']
    d = X_train.shape[1]
    print(f"\n=== Dataset {ds_idx}  train={X_train.shape}  test={X_test.shape} ===")

    # Stratified subsample for primary models
    rng = np.random.default_rng(42)
    if len(X_train) > PRIMARY_SAMPLE_SIZE:
        ip=np.where(y_train==1)[0]; in_=np.where(y_train==0)[0]
        np_=min(len(ip),int(PRIMARY_SAMPLE_SIZE*len(ip)/len(y_train)))
        nn_=min(PRIMARY_SAMPLE_SIZE-np_,len(in_))
        ch=np.concatenate([rng.choice(ip,np_,replace=False),rng.choice(in_,nn_,replace=False)])
        rng.shuffle(ch)
        Xtr_sub,ytr_sub=X_train[ch].astype(np.float32),y_train[ch]
    else:
        Xtr_sub,ytr_sub=X_train.astype(np.float32),y_train

    # Train secondary models
    print("  Training secondary models...")
    secondary = {}
    m6=IsolationForest(n_estimators=100,random_state=42,n_jobs=-1); m6.fit(X_train)
    secondary['M6']=('Isolation Forest',{'train':-m6.score_samples(X_train),'test':-m6.score_samples(X_test)})
    print("    M6 done")

    m7=train_model(Autoencoder(d),X_train,'ae')
    secondary['M7']=('Autoencoder',{'train':get_scores(m7,X_train,'ae'),'test':get_scores(m7,X_test,'ae')})
    print("    M7 done")

    m8=train_model(VAE(d),X_train,'vae')
    secondary['M8']=('VAE',{'train':get_scores(m8,X_train,'vae'),'test':get_scores(m8,X_test,'vae')})
    print("    M8 done")

    m9,dp=train_model(DAGMM(d),X_train,'dagmm')
    secondary['M9']=('DAGMM',{'train':get_scores(m9,X_train,'dagmm',dp),'test':get_scores(m9,X_test,'dagmm',dp)})
    print("    M9 done")

    m10,sc=train_model(DeepSVDD(d),X_train,'deep_svdd')
    secondary['M10']=('Deep SVDD',{'train':get_scores(m10,X_train,'deep_svdd',sc),'test':get_scores(m10,X_test,'deep_svdd',sc)})
    print("    M10 done")

    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(14,11))
    color_idx = 0
    for sec_key,scores in secondary.items():
        sec_label,score_dict=scores
        Xtr_h=np.column_stack((Xtr_sub,score_dict['train'][:len(Xtr_sub)])).astype(np.float32)
        Xte_h=np.column_stack((X_test,score_dict['test'])).astype(np.float32)
        for pm,plabel,pfn in prim_configs:
            print(f"    [{pm}+{sec_key}] fitting...", end=' ', flush=True)
            try:
                model=pfn(); model.fit(Xtr_h,ytr_sub)
                yprob=model.predict_proba(Xte_h)[:,1]
                auc=roc_auc_score(y_test,yprob)
                fpr,tpr,_=roc_curve(y_test,yprob)
                ax.plot(fpr,tpr,lw=1.2,color=COLORS[color_idx%len(COLORS)],
                        label=f'{pm}+{sec_key} ({plabel}+{sec_label}) AUC={auc:.3f}')
                print(f"AUC={auc:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
            color_idx+=1

    ax.plot([0,1],[0,1],'k--',lw=1.5,label='Random')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('False Positive Rate',fontsize=13)
    ax.set_ylabel('True Positive Rate',fontsize=13)
    ax.set_title(f'ROC Curves – Dataset {ds_idx} (All 25 Hybrid Combinations)',fontsize=15,fontweight='bold')
    ax.legend(loc='lower right',fontsize=7,ncol=2)
    ax.grid(True,alpha=0.3)
    plt.tight_layout()
    for fmt in ['pdf','tiff','png']:
        path=f"roc_curves/dataset_{ds_idx}_roc.{fmt}"
        kw={'dpi':200} if fmt in ('tiff','png') else {}
        plt.savefig(path,format=fmt,**kw)
        print(f"  Saved: {path}")
    plt.close()
    print(f"  ✅ Dataset {ds_idx} ROC curves saved.")

print("\n🎉 All done! Check roc_curves/ for PNG, PDF, and TIFF files.")
