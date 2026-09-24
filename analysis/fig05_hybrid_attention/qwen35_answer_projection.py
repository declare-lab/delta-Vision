"""Answer-direction projections of write-position source readouts.

The 4096-dimensional core is mapped to 2560 residual channels with the SAME
native total-readout RMS denominator and native gate for both sources.
This is a local unembedding-direction diagnostic, not a downstream causal effect.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

from analysis.fig05_hybrid_attention.qwen35_state_sources import StateSourceTracker, split_sources, relative_error


def map_sources(parts, native_core, gate, norm_weight, out_weight, eps):
    factor=(native_core.float().square().mean(-1,keepdim=True)+eps).rsqrt()
    factor=factor*norm_weight.float()*F.silu(gate.float())
    return F.linear((parts.float()*factor[None]).flatten(-2),out_weight.float())


class AnswerProjectionTracker(StateSourceTracker):
    def __init__(self,model,mask):
        super().__init__(model,mask)
        self.sources={};self.native_cores={};self.gates={};self.mixed={};self.source_checks={}
        self.text_idx=(~mask[0]).nonzero().flatten()

    def record(self,layer,q,k,v,kw,native_core,native_final):
        assert q.shape[1]==self.prompt_length and kw.get('initial_state') is None
        parts,stats,state,boundary=split_sources(q,k,v,kw['g'],kw['beta'],self.start,self.end)
        native=native_core[0].index_select(0,self.text_idx).detach()
        assert relative_error(parts[1]+parts[2],parts[0])<2e-5
        error=relative_error(parts[0],native);assert error<.03
        self.sources[layer]=parts
        self.native_cores[layer]=native
        self.source_checks[layer]=dict(core_additivity=relative_error(parts[1]+parts[2],parts[0]),
                                       core_replay_vs_native=error)

    @contextmanager
    def activate(self):
        handles=[]
        try:
            for i,layer in enumerate(self.model.model.language_model.layers):
                if layer.block_type!='linear_attention':continue
                def capture_gate(module,args,_i=i):
                    gate=args[1].reshape(self.prompt_length,32,128)
                    self.gates[_i]=gate.index_select(0,self.text_idx).detach()
                def capture_output(module,args,output,_i=i):
                    self.mixed[_i]=output[0].index_select(0,self.text_idx).detach()
                handles.append(layer.linear_attn.norm.register_forward_pre_hook(capture_gate))
                handles.append(layer.linear_attn.out_proj.register_forward_hook(capture_output))
            with super().activate():yield self
        finally:
            for h in handles:h.remove()

    def project(self,direction):
        assert direction.shape==(2560,)
        records=[]
        d=direction.float();dn=d.norm()
        for i in sorted(self.sources):
            module=self.model.model.language_model.layers[i].linear_attn
            parts=self.sources[i];native=self.mixed[i].float()
            mapped=map_sources(parts,self.native_cores[i],self.gates[i],module.norm.weight,
                               module.out_proj.weight,module.norm.variance_epsilon)
            error=relative_error(mapped[0],native);assert error<.03,(i,error)
            additivity=relative_error(mapped[1]+mapped[2],mapped[0]);assert additivity<2e-5
            a=mapped[1]@d;b=native@d;b_replay=mapped[0]@d;t=mapped[2]@d
            denominator_scale=native.norm(dim=-1)*dn
            stable=b.abs()>1e-6*denominator_scale
            numerical_stable=b.abs()>10*(b-b_replay).abs()
            c=a.abs()/b.abs().clamp_min(1e-30)
            c_replay=a.abs()/b_replay.abs().clamp_min(1e-30)
            values=torch.stack((a,b,b_replay,t,c,c_replay,
                parts[1].flatten(1).norm(dim=-1)/parts[0].flatten(1).norm(dim=-1).clamp_min(1e-30),
                mapped[1].norm(dim=-1)/native.norm(dim=-1).clamp_min(1e-30),
                b/denominator_scale.clamp_min(1e-30),stable.float(),numerical_stable.float()),-1).cpu().tolist()
            positions=[]
            for position,x in zip(self.text_idx.cpu().tolist(),values):
                if position<self.end:continue
                positions.append(dict(position=position,distance_after_visual=position-self.end+1,
                    answer_position=position==self.prompt_length-1,visual_signed_projection=x[0],
                    total_signed_projection=x[1],replay_total_projection=x[2],text_signed_projection=x[3],
                    C_visual=x[4] if x[9] else None,C_visual_replay_denominator=x[5],
                    core_norm_ratio=x[6],projected_norm_ratio=x[7],total_direction_cosine=x[8],
                    denominator_stable=bool(x[9]),denominator_above_rounding=bool(x[10])))
            records.append(dict(layer=i,positions=positions,checks=dict(self.source_checks[i],
                mapped_additivity=additivity,mapped_replay_vs_native=error)))
        assert len(records)==24
        return records
