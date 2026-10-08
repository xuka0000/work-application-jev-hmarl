"""Lossless shared-field encoding for bounded Jev choice requests.

The byte budget is a conservative local guard, not a claim to know Jev's
tokenizer. Actual server token usage remains part of the credentialed pilot.
"""
from copy import deepcopy
from .decision_teacher import canonical,value_hash

REFERENCE_INSTRUCTIONS=('Each option lists action_refs into state.action_atoms. Resolve route_ref through state.routes '
    'and evidence_set_ref through state.evidence_sets. Device and report tables give column names and rows, '
    'with record_id in the first column; table constants apply to every row. Public-node and edge tables also use named columns. '
    'These references and tables only deduplicate identical data. ')


def _pack_mapping(records,*,identical=False):
    groups={}
    for key,row in records.items():
        columns=tuple(sorted(row));group=(columns,canonical(row) if identical else canonical(row.get('kind')))
        groups.setdefault(group,[]).append([key]+[row[c] for c in columns])
    tables=[]
    for (columns,_),rows in sorted(groups.items()):
        constants={name:rows[0][i+1] for i,name in enumerate(columns)
                   if all(row[i+1]==rows[0][i+1] for row in rows)}
        varying=[(i,name) for i,name in enumerate(columns) if name not in constants]
        tables.append(dict(columns=['record_id']+[name for _,name in varying],constants=constants,
            rows=[[row[0]]+[row[i+1] for i,_ in varying] for row in rows]))
    return dict(tables=tables)


def _unpack_mapping(packed):
    return {row[0]:dict(table['constants'],**dict(zip(table['columns'][1:],row[1:])))
            for table in packed['tables'] for row in table['rows']}


def _encode(request,ids):
    result=deepcopy(request);body=result['body'];state=body['state'];original=state.pop('candidates')
    for name in ('received_devices','received_reports'):
        state[name]=_pack_mapping(state[name],identical=name=='received_devices')
    for name in ('public_nodes','public_edges'):
        state[name]=_pack_mapping({str(i):row for i,row in enumerate(state[name])})
    banks={k:{} for k in ('action_atoms','routes','evidence_sets')};indices={k:{} for k in banks}
    def intern(kind,value):
        identity=canonical(value)
        if identity not in indices[kind]:
            key=f'{kind}_{len(banks[kind])}'
            indices[kind][identity]=key;banks[kind][key]=value
        return indices[kind][identity]
    options={}
    for cid in ids:
        refs=[]
        for item in original[cid]:
            action=deepcopy(item)
            if 'evidence' in action:action['evidence_set_ref']=intern('evidence_sets',action.pop('evidence'))
            if 'route' in action:action['route_ref']=intern('routes',action.pop('route'))
            refs.append(intern('action_atoms',action))
        options[cid]=dict(action_refs=refs)
    state.update(banks)
    question=body['questions']['action'];question['criteria']={**options,'abstain':question['criteria']['abstain']}
    question['instructions']=REFERENCE_INSTRUCTIONS+question['instructions']
    pairs=list(zip(request['candidate_ids'],request['candidate_indices']))
    result.update(contract_version='received_critical_ens_preferences_compact_v3',
        candidate_ids=ids,candidate_indices=[i for cid,i in pairs if cid in ids],
        request_sha256=value_hash(body),uncompacted_request_sha256=request['request_sha256'],
        compact_serialized_bytes=len(canonical(body).encode('utf-8')))
    return result


def compact_request(request,*,byte_budget=28000):
    if request['contract_version']!='received_critical_ens_preferences_v2':raise ValueError('Critical-ENS v2 input required')
    if not 1000<=byte_budget<=28000:raise ValueError('Unregistered request byte budget')
    ids=list(request['candidate_ids']);original=request['body']['state']['candidates']
    priority=sorted(ids,key=lambda cid:(original[cid]!=[{'kind':'wait'}],cid))
    for count in range(len(ids),1,-1):
        chosen=set(priority[:count]);selected=[cid for cid in ids if cid in chosen]
        encoded=_encode(request,selected)
        if encoded['compact_serialized_bytes']<=byte_budget:
            encoded['byte_budget']=byte_budget;return encoded
    raise ValueError('Received evidence cannot fit the bounded teacher request with two alternatives')


def expand_body(request):
    if request['contract_version']!='received_critical_ens_preferences_compact_v3':raise ValueError('Compact v3 required')
    body=deepcopy(request['body']);state=body['state']
    for name in ('received_devices','received_reports'):
        state[name]=_unpack_mapping(state[name])
    for name in ('public_nodes','public_edges'):
        records=_unpack_mapping(state[name]);state[name]=[records[str(i)] for i in range(len(records))]
    atoms=state.pop('action_atoms');routes=state.pop('routes');evidence=state.pop('evidence_sets')
    question=body['questions']['action'];options={}
    for cid in request['candidate_ids']:
        actions=[]
        for ref in question['criteria'][cid]['action_refs']:
            a=deepcopy(atoms[ref])
            if 'route_ref' in a:a['route']=deepcopy(routes[a.pop('route_ref')])
            if 'evidence_set_ref' in a:a['evidence']=deepcopy(evidence[a.pop('evidence_set_ref')])
            actions.append(a)
        options[cid]=actions
    state['candidates']=options;question['criteria']={**options,'abstain':question['criteria']['abstain']}
    question['instructions']=question['instructions'].removeprefix(REFERENCE_INSTRUCTIONS)
    return body
