"""Read original Table 5 results; computation lives in visual_rank_statistics.py."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2] /
                        'artifacts/diagnostics/native_visual_rank_20260912/overall.csv')
    args = parser.parse_args()
    with args.source.open(newline='') as handle:
        rows = list(csv.reader(handle))
    for row in rows:
        print('\t'.join(row))


if __name__ == '__main__':
    main()
