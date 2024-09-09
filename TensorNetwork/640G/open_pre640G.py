# %%
import torch
import string
import re
import numpy as np

# %%
scheme = torch.load('TensorNetwork/640G/reproduce_scheme_n53_m20_ABCDCDAB_3000000_einsum_10_open.pt')

# %%
idx, ein, [bi, bj] = scheme[1][356][:3]
restletters = "".join(list((set(string.ascii_letters) - set(ein))))
ein_old = re.split('->|,', ein)
ein_new = ein_old[0].replace(ein_old[0][0], restletters[0]) + ',' +\
            ein_old[1].replace(ein_old[1][0], restletters[1]) + '->' +\
            ein_old[2].replace(ein_old[2][0], restletters[:2])
bnew = [torch.cat(bi)*16 + torch.cat(bj)]
scheme[1][356] = (idx, ein_new, [bnew,[]], None, None)

# %%
nsch = []
flat = [1] * 455
laten = -1
lasplit = -1
split = 512
for num,step in enumerate(scheme[1]):
    nstep = {}
    
    i, j = step[0]
    restletters = "".join(list((set(string.ascii_letters) - set(step[1]))))
    s1 = restletters[:flat[i]]
    s2 = restletters[flat[i]:flat[i] + flat[j]]
    ein_old = re.split('->|,', step[1])
    nstep1 = step[1]
    nstep1 = nstep1.replace(ein_old[0][0], s1)
    nstep1 = nstep1.replace(ein_old[1][0], s2)
    ein_new = re.split('->|,', nstep1)
    
    batch_i, batch_j = step[2]
    
    nstep['index'] = step[0]
    nstep['ein'] = nstep1
    nstep['split'] = 1
    
    if len(batch_i) > 1 or (len(step) > 3 and len(batch_i) == len(batch_j) == 1):
        bi = torch.cat(batch_i)
        bj = torch.cat(batch_j)
        blen = len(bi)
        if i == laten:
            laten = -1
            # print('1 st')
            # for index in range(blen):
            #     bi[index] = labatch[bi[index]]
            bi = labatch[bi]
        
        nstep['type'] = 1
        nstep['flat'] = (flat[i], flat[j])
        # print(num,' ',i,' ',j,' ',nstep['flat'])
        nstep['ein'] = step[1]
        if lasplit == -1:
            lasplit = i
            chubi_begin = list(range(0, bi[-1],  int(np.ceil(bi[-1]/split))))
            assert len(chubi_begin) == split, len(chubi_begin)
        assert lasplit == i
        # print(chubi_begin)
        chubi = []
        chubj = []
        begin = 0
        chubi_begin_new = []
        for index in range(split - 1):
            chubi_begin_new.append(begin)
            valuemin = chubi_begin[index]
            valuemax = chubi_begin[index+1]
            end = torch.searchsorted(bi, valuemax)
            # print(begin,end, begin-end,bi[begin:end]-valuemin)
            chubi.append(bi[begin:end]-valuemin)
            chubj.append(bj[begin:end])
            begin = end
        chubi_begin_new.append(begin)
        chubi_begin = chubi_begin_new
        chubi.append(bi[begin:] - valuemax)
        chubj.append(bj[begin:])
        nstep['chunk_batch'] = (chubi, chubj)
        flat[i] = 1
    elif len(step)>3:
        nstep['flat'] = (flat[i], flat[j])
        # print(num,' ',i,' ',j,' ',nstep['flat'])
        flat[i] = flat[i] + flat[j]
        nstep['type'] = 2
        if len(batch_i) == 1:
            laten = i
            labatch = batch_i[0]
    else:
        nstep['flat'] = (flat[i], flat[j])
        # print(num,' ',i,' ',j,' ',nstep['flat'])
        assert flat[i] == 1 or flat[j] == 1
        if flat[i] > 1:
            assert ein_old[2][0] == ein_old[0][0]
            flat[i] = flat[i]
        elif flat[j] > 1:
            assert ein_old[2][0] == ein_old[1][0]
            flat[i] = flat[j]
        nstep['type'] = 3
    nsch.append(nstep)

# %%
stemindex = 27
permute = {stemindex:np.arange(len(nsch[-1]['ein'].split('->')[1])).tolist()}
for nstep in range(len(nsch)-1,-1,-1):
    i, j = nsch[nstep]['index']
    ein = re.split('->|,', nsch[nstep]['ein'])
    # print(nstep, ein,permute[i])
    ein[2] = "".join([ein[2][t] for t in permute[i]])
    permute[i] = list(range(len(ein[0])))
    permute[j] = list(range(len(ein[1])))
    if len(ein[0])>=5:
        permute[i] = np.argsort([ein[2].find(x) if x in ein[2] else 100+ein[0].find(x) for x in ein[0]])
        
    if len(ein[1])>=5:
        permute[j] = np.argsort([ein[2].find(x) if x in ein[2] else 100+ein[0].find(x) for x in ein[1]])
        
    ein[0] = "".join([ein[0][t] for t in permute[i]])
    ein[1] = "".join([ein[1][t] for t in permute[j]])
    
    flati, flatj = nsch[nstep]['flat']
    if nsch[nstep]['type'] == 1:
        if flati > 1:
            assert permute[i][0] == 0
            permute[i] = list(range(flati)) + [x + flati - 1 for x in permute[i][1:]]
        if flatj > 1:
            assert permute[j][0] == 0
            permute[j] = list(range(flatj)) + [x + flatj - 1 for x in permute[j][1:]]
    
    nsch[nstep]['ein_2'] = f'{ein[0]},{ein[1]}->{ein[2]}'
    # nsch[nstep]['ein_2'] = nsch[nstep]['ein']
    
    # print(nsch[nstep]['ein_2'])

# %%
from math import  log
import os
nodes_per_task = int(os.environ["nodes_per_task"]) # 做一个子任务需要多少个node
gpus_per_task = int(os.environ["ntasks_per_node"])
mnmodes = int(log(nodes_per_task, 2)) # modes for multi nodes for all-to-all single
mgmodes = mnmodes+int(log(gpus_per_task, 2))# modes for multi gpus, 3 是每个节点有2^3个gpu

logmodelife = {}
prestep = {}
lastmodelife = []
tmp = None
for i in range(54,357,1):
    if stemindex not in nsch[i]['index']:
        continue
    ein = re.split('->|,', nsch[i]['ein_2'])
    # print(i, ein)
    if lastmodelife == []:
        lastmodelife = [0] * len(ein[2])
    modelife = []
    for c in ein[2]:
        if c in ein[0]:
            pos = ein[0].find(c)
            modelife.append(lastmodelife[pos] + 1)
        else:
            modelife.append(0)
    lastmodelife = modelife
    logmodelife[i] = modelife
    prestep[i] = tmp
    tmp = i
    # print(np.sort(modelife)[-3:], modelife)

# %%

out_selmode = None
lastorder = None
for nstep in range(356,53,-1):
    if nstep not in logmodelife.keys():
        continue
    ein_old = re.split('->|,', nsch[nstep]['ein_2'])
    ein_new = re.split('->|,', nsch[nstep]['ein_2'])
    stepmodelife = torch.tensor(logmodelife[nstep])
    if prestep[nstep]!=None:
        prestepmodelife = torch.tensor(logmodelife[prestep[nstep]]) 
    
    internode = 0
    try:
        if stepmodelife[out_selmode[:mnmodes]].min() <= 1:
            internode = 1
    except:
        pass

    if out_selmode == None:      # 起始
        outmgchar = [ein_old[2][i] for i in range(mgmodes)]
        in_selmode = [ein_old[0].find(c) for c in outmgchar]
        # # print(in_selmode)
        in_selmode_out = sorted(in_selmode)
        
        new_order = in_selmode + [i for i in range(len(ein_old[0])) if i not in in_selmode]
        ein_new[0] = np.array(list(ein_new[0]))[new_order]
        ein_new[0] = "".join(ein_new[0].tolist())
    elif prestep[nstep]==None: # 结束
        in_selmode = None
        
        ein_new[2] = np.array(list(ein_new[2]))[lastorder]        
        ein_new[2] = "".join(ein_new[2].tolist())
        
    elif stepmodelife[out_selmode].min() > 1: #不需要重排
        mgchar = [ein_old[2][i] for i in out_selmode]
        in_selmode = [ein_old[0].find(c) for c in mgchar]

        ein_new[2] = np.array(list(ein_new[2]))[lastorder]        
        ein_new[2] = "".join(ein_new[2].tolist())
        
        new_order = in_selmode + [i for i in range(len(prestepmodelife)) if i not in in_selmode]
        ein_new[0] = np.array(list(ein_new[0]))[new_order]     
        ein_new[0] = "".join(ein_new[0].tolist())
        
    elif internode: # 节点间all2all
        outmgchar = [ein_old[2][i] for i in out_selmode]
        in_selmode_out = [ein_old[0].find(c) for c in outmgchar]
        for x in in_selmode_out:
            prestepmodelife[x] = 0 
        in_selmode = prestepmodelife.argsort(descending=True,stable=True)[:mgmodes]
        in_selmode = in_selmode.tolist()
        # # print(f"out_selmode {out_selmode}, in_selmode_out {in_selmode_out}, in_selmode {in_selmode}")
        assert len(set(in_selmode_out) &set(in_selmode)) == 0
        
        ein_new[2] = np.array(list(ein_new[2]))[lastorder]        
        ein_new[2] = "".join(ein_new[2].tolist())
        
        new_order = in_selmode + in_selmode_out + \
                    [i for i in range(len(prestepmodelife)) if i not in (in_selmode + in_selmode_out)]
        ein_new[0] = np.array(list(ein_new[0]))[new_order]     
        ein_new[0] = "".join(ein_new[0].tolist())
        # # print(f"节点间all2all, input size {4*2**(len(ein_new[0])-35)}G, output size {4*2**(len(ein_new[2])-35)}G")

    else: # 节点内all2all
        outmgchar_mn = [ein_old[2][i] for i in out_selmode[:mnmodes]]
        in_selmode_out_mn = [ein_old[0].find(c) for c in outmgchar_mn]

        outmgchar_mg = [ein_old[2][i] for i in out_selmode[mnmodes:mgmodes]]
        in_selmode_out_mg = [ein_old[0].find(c) for c in outmgchar_mg]
        for x in in_selmode_out_mg:
            prestepmodelife[x] = 0 
        # TODO: modify here
        in_selmode = prestepmodelife.argsort(descending=True,stable=True).tolist()
        # # print(f"before: in_selmode {in_selmode}")
        for mode in in_selmode_out_mn:
            in_selmode.remove(mode)
        
        in_selmode = in_selmode_out_mn + in_selmode[:(mgmodes-mnmodes)]
        # # print(f"after: in_selmode_out_mn {in_selmode_out_mn}, in_selmode {in_selmode}")
        # # # print((in_selmode_out), (in_selmode))
        assert len(set(in_selmode_out_mg) &set(in_selmode)) == 0
        
        ein_new[2] = np.array(list(ein_new[2]))[lastorder]        
        ein_new[2] = "".join(ein_new[2].tolist())
        
        new_order = in_selmode + in_selmode_out_mg + \
                    [i for i in range(len(prestepmodelife)) if i not in (in_selmode + in_selmode_out_mg)]
        ein_new[0] = np.array(list(ein_new[0]))[new_order]     
        ein_new[0] = "".join(ein_new[0].tolist())
        
        
    # print(nstep,stepmodelife[out_selmode], nsch[nstep]['flat'], nsch[nstep]['type'])
    # print(f"{ein_new[0]},{ein_new[1]}->{ein_new[2]}")
    # print(f"{ein_old[0]},{ein_old[1]}->{ein_old[2]}")
    nsch[nstep]['reorder_ein'] = f"{ein_new[0]},{ein_new[1]}->{ein_new[2]}"
    out_selmode = in_selmode
    lastorder = new_order
        
        

# %%
import torch
torch.save(nsch,f'TensorNetwork/640G/640G_rep_nsch640_split{split}_mg{mgmodes}_splitmn.pt')


