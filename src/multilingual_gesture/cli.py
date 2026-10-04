from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(prog="multigesture"); s=p.add_subparsers(dest="cmd",required=True)
    e=s.add_parser("extract-units"); e.add_argument("--motion",required=True); e.add_argument("--output",required=True); e.add_argument("--variance",type=float,required=True); e.add_argument("--closure",type=float,required=True); e.add_argument("--fps",type=int,default=15)
    t=s.add_parser("train"); t.add_argument("--pairs",required=True); t.add_argument("--output",required=True); t.add_argument("--epochs",type=int,default=50); t.add_argument("--batch-size",type=int,default=64); t.add_argument("--seed",type=int,default=0)
    m=s.add_parser("mine"); m.add_argument("--wild",required=True); m.add_argument("--units",required=True); m.add_argument("--checkpoint",required=True); m.add_argument("--output-prefix",required=True); m.add_argument("--clusters",type=int,default=100); m.add_argument("--sbert",default="all-MiniLM-L6-v2")
    r=s.add_parser("retrieve"); r.add_argument("--rules",required=True); r.add_argument("--clusters",required=True); r.add_argument("--text",required=True); r.add_argument("--source-language",default="en"); r.add_argument("--translations"); r.add_argument("--output",required=True); r.add_argument("--seed",type=int,default=0); r.add_argument("--sbert",default="all-MiniLM-L6-v2")
    a=p.parse_args(); {"extract-units":extract,"train":train,"mine":mine,"retrieve":get}[a.cmd](a)

def extract(a):
    from .pipeline import extract_units
    x=np.load(a.motion); units=extract_units(x,a.fps,variance_threshold=a.variance,closure_threshold=a.closure); length=3*a.fps
    if not units: raise ValueError("no units passed the variance and closure thresholds")
    padded=np.stack([np.pad(u,((0,length-len(u)),(0,0)),mode="edge") for u in units]); ids=np.asarray([f"unit_{i:06d}" for i in range(len(units))]); output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); np.savez(output,motion3d=padded,lengths=np.asarray([len(u) for u in units]),ids=ids)

def train(a):
    import torch
    from .model import GestureCLR,ntxent
    from .pipeline import augment_2d
    d=np.load(a.pairs); x2=d["pose2d"].astype("float32"); x3=d["motion3d"].astype("float32")
    if len(x2)!=len(x3) or len(x2)<2: raise ValueError("training requires at least two aligned 2D/3D pairs")
    if a.epochs<1 or a.batch_size<2: raise ValueError("epochs must be positive and batch size must be at least two")
    torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed); model=GestureCLR(x2.shape[-1],x3.shape[-1]); opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    for epoch in range(a.epochs):
        order=rng.permutation(len(x2)); total=0.; seen=0
        for st in range(0,len(order),a.batch_size):
            ids=order[st:st+a.batch_size]
            if len(ids)<2: continue
            aug=np.stack([augment_2d(x2[i],rng) for i in ids]); z2,z3=model(torch.from_numpy(aug),torch.from_numpy(x3[ids])); loss=ntxent(z2,z3); opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss.detach())*len(ids); seen+=len(ids)
        print(json.dumps({"epoch":epoch+1,"loss":total/seen}))
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); torch.save({"state":model.state_dict(),"d2":x2.shape[-1],"d3":x3.shape[-1]},output)

def mine(a):
    import torch
    from sentence_transformers import SentenceTransformer
    from .model import GestureCLR
    from .pipeline import bisect
    w=np.load(a.wild); u=np.load(a.units); ck=torch.load(a.checkpoint,map_location="cpu",weights_only=True); model=GestureCLR(ck["d2"],ck["d3"]); model.load_state_dict(ck["state"]); model.eval()
    with torch.no_grad(): wz=model.pose2d(torch.from_numpy(w["pose2d"].astype("float32"))).numpy(); uz=model.motion3d(torch.from_numpy(u["motion3d"].astype("float32"))).numpy()
    labels,centers=bisect(uz,a.clusters); ids=np.asarray([str(x) for x in u["ids"]]); texts=[str(x) for x in w["texts"]]; te=SentenceTransformer(a.sbert).encode(texts,normalize_embeddings=True); nearest=(wz@uz.T).argmax(1); rules=[{"english_text":text,"text_embedding":emb.tolist(),"cluster_id":int(labels[j]),"source_gesture_id":ids[j],"pose_similarity":float(wz[i]@uz[j])} for i,(text,emb,j) in enumerate(zip(texts,te,nearest))]
    prefix=Path(a.output_prefix); prefix.parent.mkdir(parents=True,exist_ok=True); Path(str(prefix)+".rules.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rules),encoding="utf-8"); np.savez(str(prefix)+".clusters.npz",ids=ids,labels=labels,centroids=centers)

def get(a):
    from sentence_transformers import SentenceTransformer
    from .pipeline import require_english,retrieve
    rules=[json.loads(x) for x in Path(a.rules).read_text(encoding="utf-8").splitlines() if x]; d=np.load(a.clusters); clusters={int(k):[str(x) for x in d["ids"][d["labels"]==k]] for k in np.unique(d["labels"])}; translations=json.loads(Path(a.translations).read_text(encoding="utf-8")) if a.translations else {}; english=require_english(a.text,a.source_language,translations); model=SentenceTransformer(a.sbert); out={"source_text":a.text,"source_language":a.source_language,"english_text":english,"gestures":retrieve(english,rules,lambda x:model.encode(x,normalize_embeddings=True),clusters,a.seed)}; output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")

if __name__=="__main__": main()
