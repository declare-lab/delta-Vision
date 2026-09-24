"""Check whether native paged FA2 can share video prefix keys across text runs."""
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_forward
from src.attention import prefix_plan_from_positions, attention_heads


def main():
    torch.set_num_threads(4)
    torch.manual_seed(7921)
    rows=[]
    with torch.inference_mode():
        cases=[([0,1,9,10],list(range(2,9))),([0,5,6,15,16],list(range(1,5))+list(range(7,15)))]
        for visual_per_frame in [110,125,126,132]:
            text=list(range(12));visual=[];start=12
            for frame in range(8):
                visual.extend(range(start,start+visual_per_frame));start+=visual_per_frame
                text.extend(range(start,start+5));start+=5
            text.extend(range(start,start+80))
            cases.append((text,visual))
        for text,visual in cases:
            plan=prefix_plan_from_positions(text,visual,'cuda')
            q=torch.randn(1,32,len(text),128,device='cuda',dtype=torch.bfloat16)
            k,v=[torch.randn(1,8,len(visual)+len(text),128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
            positions=visual+text
            order=torch.tensor(sorted(range(len(positions)),key=positions.__getitem__),device='cuda')
            pages=(len(positions)+255)//256
            keys,values=[torch.empty(pages,256,8,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
            keys.flatten(0,1)[:len(positions)].copy_(k[0].transpose(0,1).index_select(0,order))
            values.flatten(0,1)[:len(positions)].copy_(v[0].transpose(0,1).index_select(0,order))
            block_table=torch.arange(pages,device='cuda',dtype=torch.int32).expand(plan['cu_q'].numel()-1,-1).contiguous()
            def original():return attention_heads(q,k,v,scaling=128**-.5,plan=plan)
            def paged():
                result=flash_attn_varlen_func(q.transpose(1,2).reshape(-1,32,128),keys,values,
                    plan['cu_q'],plan['cu_k'],plan['max_q'],plan['max_k'],causal=True,
                    softmax_scale=128**-.5,block_table=block_table)
                return result.unsqueeze(0)
            a,b=original(),paged()
            shared_cu=torch.zeros_like(plan['cu_k'])
            shared_cu[-1]=len(positions)
            used=plan['cu_k'].diff()
            shared=_wrapped_flash_attn_varlen_forward(
                q.transpose(1,2).reshape(-1,32,128),
                keys.flatten(0,1)[:len(positions)],values.flatten(0,1)[:len(positions)],
                plan['cu_q'],shared_cu,plan['max_q'],plan['max_k'],0.,128**-.5,True,
                seqused_k=used)[0].unsqueeze(0)
            row=dict(visual=len(visual),text=len(text),equal=torch.equal(a,b),
                unequal=int(a.ne(b).sum()),max_diff=float((a-b).abs().max()),
                previous_packed_keys=plan['key_indices'].numel(),paged_capacity=pages*256)
            row.update(shared_equal=torch.equal(a,shared),shared_unequal=int(a.ne(shared).sum()),
                       shared_max_diff=float((a-shared).abs().max()))
            rows.append(row);print(json.dumps(row),flush=True)
    path=ROOT/'test/results/adapter_max_20260915/paged_fa2_probe.json'
    path.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
