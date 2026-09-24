"""One dispatcher for paper analysis; experiment arguments pass through intact."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main(argv=None):
    root = Path(__file__).resolve().parent
    catalog = json.loads((root / 'catalog.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('experiment', nargs='?', choices=catalog)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('runner', nargs='?')
    args, remainder = parser.parse_known_args(argv)
    if args.list or args.experiment is None:
        for name, item in catalog.items():
            print(f"{name}: {item['paper']} | {', '.join(item['runners'])} | {item['source_status']}")
        return
    if args.describe:
        documentation = (root.parent / 'README.md').read_text()
        start = f'<!-- analysis:{args.experiment} -->'
        end = f'<!-- /analysis:{args.experiment} -->'
        if start not in documentation or end not in documentation:
            parser.error(f'Missing experiment documentation in README.md: {args.experiment}')
        print(documentation.split(start, 1)[1].split(end, 1)[0].strip())
        return
    runners = catalog[args.experiment]['runners']
    if args.runner not in runners:
        parser.error(f"Select a runner: {', '.join(runners)}")
    if remainder[:1] == ['--']:
        remainder = remainder[1:]
    module = runners[args.runner]
    previous = sys.argv
    try:
        sys.argv = [module, *remainder]
        runpy.run_module(module, run_name='__main__')
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
