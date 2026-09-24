"""Unified Video-MME timing and resource entry. Pass profile arguments after --."""
import argparse,runpy,sys
PROFILES={
 'llm':'src.benchmarking.videomme.timing',
 'resources':'src.benchmarking.videomme.resources',
 'vision-removed':'src.benchmarking.videomme.vision_removed',
 'flops-removed':'src.benchmarking.videomme.flops_removed',
 'resource-report':'src.benchmarking.videomme.resource_report',
 'layer-report':'src.benchmarking.videomme.layer_report',
 'base-diagnostic':'src.benchmarking.engines.base',
 'adapter-diagnostic':'src.benchmarking.engines.adapter',
 'pruning-diagnostic':'src.benchmarking.engines.pruning',
 'prefill':'src.benchmarking.common.prefill',
}

def main(argv=None):
 raw=list(sys.argv[1:] if argv is None else argv);split=raw.index('--') if '--' in raw else len(raw)
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('dataset',choices=['videomme']);parser.add_argument('profile',nargs='?',choices=PROFILES);parser.add_argument('--list',action='store_true')
 a=parser.parse_args(raw[:split])
 if a.list or a.profile is None:
  print('\n'.join(PROFILES));return
 module=PROFILES[a.profile];previous=sys.argv
 try:
  sys.argv=[module,*raw[split+1:]]
  runpy.run_module(module,run_name='__main__')
 finally:sys.argv=previous

if __name__=='__main__':main()
