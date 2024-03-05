import os
import argparse
import subprocess
import json
import pandas as pd
import shutil


def clean(data_unit_file, out_dir):
    metadata = ','.join(pd.read_csv(os.path.join(out_dir, data_unit_file), nrows=0).columns)
    metadata = json.loads(metadata)

    sequences = pd.read_csv(os.path.join(out_dir, data_unit_file), header=1)
    data = sequences.loc[:, ["ANARCI_numbering_heavy", "ANARCI_numbering_light"]]
    data.dropna(inplace=True)
    newname = metadata['BType'].split('/')[-1] + '_' + data_unit_file
    print(os.path.join(out_dir, newname))

    data.to_csv(os.path.join(out_dir, newname))
    return metadata['BType'] + '_' + data_unit_file


def download(args):
    if args.reset:
        shutil.rmtree(args.out_dir)
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    dfile = open(args.bulk_script, 'r')
    file_names = ['_'.join(f.split('_')[1:]) for f in os.listdir(args.out_dir)]
    Lines = dfile.readlines()

    for line in Lines:
        L = line.strip()
        name = L.split('/')[-1]
        if name[:-3] in file_names:
            print('skip ', L)
            continue
        subprocess.run(L + " -P " + args.out_dir, shell=True)
        line = "gzip -d {}".format(os.path.join(args.out_dir, name))
        subprocess.run(line, shell=True)
        clean(name[:-3], args.out_dir)
        line = "rm {}".format(os.path.join(args.out_dir, name[:-3]))
        subprocess.run(line, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reset",
        type=bool,
        default=False,
        help="Download all files",
    )
    parser.add_argument(
        "--bulk-script",
        type=str,
        default='data/bulk_download.sh',
        help="bash script file for downloading OAS files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default='Data',
        help="destination for raw files",
    )

    download(parser.parse_args())
