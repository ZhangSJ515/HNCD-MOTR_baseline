#!/usr/bin/env python3
"""Quick AirMOT environment sanity check for HNCD-MOTR."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--det-db', default=None)
    parser.add_argument('--dataset-name', default='AirMot')
    args = parser.parse_args()

    split_root = os.path.join(args.data_root, args.dataset_name)
    print('[AirMOT] dataset root:', split_root)

    for split in ['train', 'val', 'test']:
        p = os.path.join(split_root, split)
        print(f'[{split}]', 'OK' if os.path.isdir(p) else 'MISSING', p)

    if args.det_db:
        if not os.path.isfile(args.det_db):
            raise FileNotFoundError(args.det_db)
        with open(args.det_db, 'r', encoding='utf-8') as f:
            db = json.load(f)
        print('[DET_DB] entries:', len(db))

    print('AirMOT setup check finished.')


if __name__ == '__main__':
    main()
