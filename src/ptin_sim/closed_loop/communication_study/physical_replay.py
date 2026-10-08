"""Lossless variable-candidate replay shards with explicit next-state actions."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import MappingProxyType
import numpy as np
import torch


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_shard(path,*,states,catalogues,chosen,rewards,terminal):
    path=Path(path)
    if path.exists():raise ValueError('Replay shards cannot be overwritten')
    count=len(chosen)
    if len(states)!=count+1 or len(catalogues)!=count+1 or len(rewards)!=count or len(terminal)!=count:
        raise ValueError('Each transition needs a real next state and next catalogue')
    offsets=np.cumsum([0]+[len(c) for c in catalogues],dtype=np.int64)
    arrays=dict(states=np.asarray(states,dtype=np.float32),candidates=np.concatenate(catalogues).astype(np.float32),
        offsets=offsets,chosen=np.asarray(chosen,dtype=np.int64),rewards=np.asarray(rewards,dtype=np.float32),
        terminated=np.asarray(terminal,dtype=np.bool_))
    validate_shard(arrays)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    return dict(path=str(path),sha256=file_sha256(path),transitions=count)


def validate_shard(a):
    required={'states','candidates','offsets','chosen','rewards','terminated'}
    if set(a)!=required:raise ValueError('Unknown or missing replay fields')
    count=len(a['chosen']);offsets=a['offsets'];sizes=np.diff(offsets)
    if (count==0 or a['states'].ndim!=2 or len(a['states'])!=count+1
            or a['candidates'].ndim!=2 or offsets.shape!=(count+2,) or offsets.dtype.kind not in 'iu'
            or offsets[0]!=0 or offsets[-1]!=len(a['candidates']) or np.any(sizes<=0)
            or a['chosen'].shape!=(count,) or a['chosen'].dtype.kind not in 'iu'
            or np.any(a['chosen']<0) or np.any(a['chosen']>=sizes[:-1])
            or a['rewards'].shape!=(count,) or a['terminated'].shape!=(count,)
            or a['terminated'].dtype!=np.bool_ or not a['terminated'][-1] or a['terminated'][:-1].any()):
        raise ValueError('Malformed complete-episode replay')
    for key in ('states','candidates','rewards'):
        if not np.isfinite(a[key]).all():raise ValueError('Nonfinite replay data')


class PhysicalReplay:
    def __init__(self,manifest,*,cache_bytes=0,cache_shards=None):
        # No verified process allocation means no persistent array cache. The
        # optional legacy shard count can only further restrict the byte limit.
        if type(cache_bytes) is not int or not 0<=cache_bytes<=sys.maxsize:
            raise ValueError('Nonnegative bounded cache bytes required')
        if cache_shards is not None and (type(cache_shards) is not int or cache_shards<1):
            raise ValueError('Positive cache limit required')
        self.path=Path(manifest).absolute()
        self._manifest_identity=self._identity(self.path)
        self.metrics=dict(hash_calls=0,hash_bytes=0,decompressions=0,
                          hash_s=0.,decompression_s=0.,cache_hits=0,cache_misses=0,
                          peak_cache_bytes=0,batch_reuses=0,peak_batch_array_bytes=0,
                          reward_reads=0,reward_read_s=0.)
        self._manifest_sha=self._hash(self.path)
        self.metadata=json.loads(self.path.read_text(encoding='utf-8'))
        self._check(self.path,self._manifest_identity)
        self._metadata_json=json.dumps(self.metadata,sort_keys=True)
        m=self.metadata
        if m.get('status')!='complete_training_corpus' or m.get('contains_test_or_validation_data') is not False:
            raise ValueError('Complete training-only corpus required')
        self._cache_limit=cache_bytes;self._cache_shards=cache_shards
        self._cache=OrderedDict();self._cache_view=MappingProxyType(self._cache)
        self._cache_bytes=0;self.index=[];self._shards=[]
        allowed={f'training_{i:02d}' for i in range(60)}
        for i,shard in enumerate(m['shards']):
            if shard['case_id'] not in allowed:raise ValueError('Evaluation case cannot enter training')
            if shard['encoder_sha256']!=m['encoder_sha256']:raise ValueError('Mixed encoder identities')
            p=(self.path.parent/shard['path']).resolve()
            if not p.is_relative_to(self.path.parent.resolve()):raise ValueError('Replay path escapes corpus')
            identity=self._identity(p)
            if self._hash(p)!=shard['sha256']:raise ValueError('Replay digest mismatch')
            a=self._load(p)
            self._check(p,identity)
            validate_shard(a)
            if (len(a['chosen'])!=shard['transitions'] or a['states'].shape[1]!=m['state_dim']
                    or a['candidates'].shape[1]!=m['action_dim']):raise ValueError('Replay schema mismatch')
            self.index.extend((i,j) for j in range(shard['transitions']))
            self._shards.append((p,identity,shard['sha256'],self._array_bytes(a)))
            del a
        if not 0<len(self.index)<=m['total_collection_transitions_including_teacher_expansions']<=20000:
            raise ValueError('Registered collection budget violated')
        if len(self.index)!=m['transition_count']:raise ValueError('Transition count mismatch')
        self._frozen_index=tuple(self.index)

    @staticmethod
    def _identity(path):
        stat=path.stat()
        # Corpus paths are resolved/contained once at entry. stat follows the
        # endpoint and preserves file identity checks without walking the full
        # directory chain on every sampled row.
        return (str(path),stat.st_dev,stat.st_ino,stat.st_size,
                stat.st_mtime_ns,stat.st_ctime_ns)

    @property
    def cache_limit(self):return self._cache_limit

    @property
    def cache_shards(self):return self._cache_shards

    @property
    def cache(self):return self._cache_view

    @property
    def cache_bytes(self):return self._cache_bytes

    def _check(self,path,identity):
        if self._identity(path)!=identity:
            raise ValueError('Replay changed after validation: '+str(path))

    @staticmethod
    def _array_bytes(arrays):
        return sum(a.nbytes for a in arrays.values())

    def _hash(self,path):
        started=time.monotonic();value=file_sha256(path)
        self.metrics['hash_calls']+=1;self.metrics['hash_bytes']+=path.stat().st_size
        self.metrics['hash_s']+=time.monotonic()-started
        return value

    def _load(self,path):
        started=time.monotonic()
        # An owning NumPy array can re-enable WRITEABLE even after setflags(False).
        # Immutable bytes back every cached array/view, so neither its public
        # view nor the base chain can enable writes. Freeze one NPZ member at a
        # time to avoid simultaneously copying the entire decoded shard.
        with np.load(path,allow_pickle=False) as f:
            arrays={k:self._immutable_array(f[k]) for k in f.files}
        self.metrics['decompressions']+=1
        self.metrics['decompression_s']+=time.monotonic()-started
        return arrays

    @staticmethod
    def _immutable_array(array):
        return np.frombuffer(array.tobytes(order='C'),dtype=array.dtype).reshape(array.shape)

    def verify_integrity(self):
        """Strong completion gate, including edits hidden by restored timestamps."""
        self._check(self.path,self._manifest_identity)
        if (self._hash(self.path)!=self._manifest_sha
                or json.dumps(self.metadata,sort_keys=True)!=self._metadata_json
                or tuple(self.index)!=self._frozen_index):
            raise ValueError('Replay manifest or sampling index changed after validation')
        for path,identity,sha,_ in self._shards:
            self._check(path,identity)
            if self._hash(path)!=sha:
                raise ValueError('Replay digest changed after validation')
            self._check(path,identity)
        self._check(self.path,self._manifest_identity)
        return dict(status='pass',manifest_sha256=self._manifest_sha,shards=len(self._shards))

    def __len__(self):return len(self.index)

    def episode_rewards(self,shard_id):
        """Read the already-validated reward vector once, with scalar semantics.

        NPZ members are independent: no state/candidate decode is needed here.
        No vector or shard is retained beyond the caller's return computation.
        """
        self._check(self.path,self._manifest_identity)
        path,identity,_,_=self._shards[shard_id];self._check(path,identity)
        started=time.monotonic()
        with np.load(path,allow_pickle=False) as f:rewards=f['rewards']
        self._check(path,identity)
        self.metrics['reward_reads']+=1;self.metrics['reward_read_s']+=time.monotonic()-started
        return [float(value) for value in rewards]

    def row(self,index):
        temporary=getattr(self,'_batch_arrays',None)
        if temporary is None:self._check(self.path,self._manifest_identity)
        shard_id,j=self._frozen_index[index]
        path,identity,_,size=self._shards[shard_id]
        # Direct hits are checked each time. Within a batch, immutable arrays
        # permit one check at first access plus a fence before returning tensors.
        if temporary is None or shard_id not in temporary:self._check(path,identity)
        if shard_id not in self._cache:
            self.metrics['cache_misses']+=1
            retain=0<size<=self.cache_limit
            if retain:
                while self._cache and (self.cache_bytes+size>self.cache_limit
                        or (self.cache_shards is not None and len(self._cache)>=self.cache_shards)):
                    _,old=self._cache.popitem(last=False);self._cache_bytes-=self._array_bytes(old)
                    del old
            if temporary is not None and shard_id in temporary:
                a=temporary[shard_id];self.metrics['batch_reuses']+=1
            else:
                a=self._load(path);self._check(path,identity)
            if self._array_bytes(a)!=size:raise ValueError('Replay array footprint changed')
            if retain:
                self._cache[shard_id]=MappingProxyType(a);self._cache_bytes+=self._array_bytes(a)
                self.metrics['peak_cache_bytes']=max(self.metrics['peak_cache_bytes'],self.cache_bytes)
        else:
            self.metrics['cache_hits']+=1
            self._cache.move_to_end(shard_id);a=self._cache[shard_id]
        if temporary is not None and shard_id not in temporary:
            temporary[shard_id]=a
            self.metrics['peak_batch_array_bytes']=max(self.metrics['peak_batch_array_bytes'],
                sum(self._array_bytes(arrays) for arrays in temporary.values()))
        o=a['offsets']
        return dict(state=a['states'][j],next_state=a['states'][j+1],
            candidates=a['candidates'][o[j]:o[j+1]],next_candidates=a['candidates'][o[j+1]:o[j+2]],
            chosen=int(a['chosen'][j]),reward=float(a['rewards'][j]),terminated=bool(a['terminated'][j]))

    def batch(self,indices,*,packed=False):
        # Transient references replace the repeated full-shard allocations that
        # row views already kept alive. They do not enter the persistent LRU;
        # even a zero/oversize cache decodes each selected shard only once.
        if getattr(self,'_batch_arrays',None) is not None:
            raise RuntimeError('Concurrent/nested replay batch requires separate ownership')
        self._check(self.path,self._manifest_identity)
        self._batch_arrays={}
        try:
            rows=[self.row(i) for i in indices]
            self._check(self.path,self._manifest_identity)
            for shard_id in self._batch_arrays:
                path,identity,_,_=self._shards[shard_id];self._check(path,identity)
        finally:self._batch_arrays=None
        if not rows:raise ValueError('Empty training batch')
        out=dict(states=torch.tensor(np.stack([r['state'] for r in rows])),
            next_states=torch.tensor(np.stack([r['next_state'] for r in rows])),
            chosen=torch.tensor([r['chosen'] for r in rows],dtype=torch.long),
            rewards=torch.tensor([r['reward'] for r in rows],dtype=torch.float32),
            terminated=torch.tensor([r['terminated'] for r in rows],dtype=torch.float32))
        for field,mask in (('candidates','valid'),('next_candidates','next_valid')):
            width=max(len(r[field]) for r in rows);features=rows[0][field].shape[1]
            out[field]=(torch.from_numpy(np.concatenate([r[field] for r in rows])) if packed
                        else torch.zeros(len(rows),width,features))
            out[mask]=torch.zeros(len(rows),width,dtype=torch.bool)
            for i,r in enumerate(rows):
                n=len(r[field]);out[mask][i,:n]=True
                if not packed:out[field][i,:n]=torch.from_numpy(r[field].copy())
        return out
