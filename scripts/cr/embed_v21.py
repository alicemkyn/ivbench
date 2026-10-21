import pandas as pd, numpy as np, time, glob, os, sys
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
p = snapshot_download('pritamdeka/S-PubMedBert-MS-MARCO', allow_patterns=['*.json','*.txt','pytorch_model.bin','1_Pooling/*'])
C = pd.read_csv(sys.argv[1], low_memory=False); CB = C[C.family=='binary'].reset_index(drop=True)
ENDP = ['outcome_title','outcome_desc','outcome_timeframe','conditions']
IVT = ['treat_intervention','treat_intervention_desc','treat_arm_desc','treat_group','treat_group_desc','ctrl_intervention','ctrl_arm_desc','ctrl_group','ctrl_group_desc']
def tc(df, cols): return df[cols].fillna('').astype(str).agg(' | '.join, axis=1).str.slice(0, 2000).tolist()
t0=time.time(); model = SentenceTransformer(p, device='cpu'); model.max_seq_length = 384
E = model.encode(tc(CB, ENDP), batch_size=32, normalize_embeddings=True); print('E', E.shape, round(time.time()-t0), flush=True)
I = model.encode(tc(CB, IVT), batch_size=32, normalize_embeddings=True); print('I', I.shape, round(time.time()-t0))
key = (CB.nct_id.astype(str)+'|'+CB.outcome_title.fillna('').astype(str)+'|'+CB.treat_group.fillna('').astype(str)+'|'+CB.ctrl_group.fillna('').astype(str)+'|'+CB.split_time.astype(str)).to_numpy()
np.savez('embeddings_v21_binary.npz', key=key, E=E, I=I)
