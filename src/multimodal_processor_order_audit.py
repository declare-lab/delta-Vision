"""Check grouped preprocessing preserves image order and patch coordinates."""
import json
from pathlib import Path
import torch
from transformers import AutoProcessor
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def main():
    import src
    src.__path__.insert(0,str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    from src.data import QwenBenchmarkDataset
    from src.embedding_adapter_corrected_eval import MODEL
    torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(MODEL)
    directory=ROOT/'artifacts/diagnostics/multimodal_position_inference_20260914'
    output=ROOT/'artifacts/diagnostics/multimodal_processor_order_20260914'
    output.mkdir(parents=True,exist_ok=False)
    samples=[json.loads(line) for p in directory.glob('rows_*.jsonl') for line in p.open()]
    with torch.inference_mode(),(output/'rows.jsonl').open('w',buffering=1) as result:
        for benchmark in ('muirbench','mmiu'):
            ds=QwenBenchmarkDataset(str(ROOT/f'data/benchmarks/{benchmark}/test.jsonl'),processor,benchmark,max_samples=1000)
            for sample in samples:
                if sample['benchmark']!=benchmark:continue
                index=sample['index']
                paths=ds._image_paths(ds.rows[index])
                images=[]
                for path in paths:
                    with Image.open(path) as image:images.append(image.convert('RGB'))
                combined=processor.image_processor(images=images,return_tensors='pt')
                offset=0
                individual=[]
                for n,image in enumerate(images):
                    one=processor.image_processor(images=[image],return_tensors='pt')
                    count=len(one['pixel_values'])
                    assert torch.equal(one['image_grid_thw'][0],combined['image_grid_thw'][n]),(benchmark,index,n,'grid')
                    chunk=combined['pixel_values'][offset:offset+count]
                    assert torch.equal(one['pixel_values'],chunk),(benchmark,index,n,'patch order')
                    individual.append(dict(index=n,path=str(paths[n]),size=image.size,patches=count,
                        grid=one['image_grid_thw'][0].tolist()))
                    offset+=count
                assert offset==len(combined['pixel_values'])
                result.write(json.dumps(dict(benchmark=benchmark,index=index,images=individual,exact=True))+'\n')
                print('CHECKED',benchmark,index,'images',len(images),flush=True)
                for image in images:image.close()
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
