import os
import pandas as pd
import json
import argparse
import re


def read_data(data_dir):
    all_seqs = []
    all_pos = []
    print(len(os.listdir(data_dir)))
    for i, f in enumerate(os.listdir(data_dir)):
        if f[-3:] != "csv":
            continue
        t = pd.read_csv(os.path.join(data_dir, f))
        h = t["ANARCI_numbering_heavy"].map(lambda x: json.loads(x.replace("'", '"')))
        l = t["ANARCI_numbering_light"].map(
            lambda x: json.loads(x.replace("k", "l").replace("'", '"'))
        )
        h_seqs = h.map(
            lambda input_dict: "".join(
                [
                    value
                    for sub_dict in input_dict.values()
                    for value in sub_dict.values()
                ]
            )
        )
        l_seqs = l.map(
            lambda input_dict: "".join(
                [
                    value
                    for sub_dict in input_dict.values()
                    for value in sub_dict.values()
                ]
            )
        )
        h_pos = h.map(
            lambda input_dict: [
                key.strip()
                for sub_dict in input_dict.values()
                for key in sub_dict.keys()
            ]
        )
        l_pos = l.map(
            lambda input_dict: [
                key.strip()
                for sub_dict in input_dict.values()
                for key in sub_dict.keys()
            ]
        )
        seqs = [[h_seqs[i], l_seqs[i]] for i in range(len(h_seqs))]
        pos = [[h_pos[i], l_pos[i]] for i in range(len(h_seqs))]
        all_seqs.extend(seqs)
        all_pos.extend(pos)
        if i % 50 == 0:
            print("loaded", i, "files")
    return all_seqs, all_pos


def exec_mmseq(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text


def clean(args):
    all_seqs, all_pos = read_data(args.data_dir)
    tmp_dir = "tmp"
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    with open(os.path.join(tmp_dir, "DB.fasta"), "w") as f:
        for i in range(len(all_seqs)):
            f.write(f">{str(i)}\n{all_seqs[i][0]}\n")

    fasta = "tmp/DB.fasta"
    db = os.path.join(tmp_dir, "DB")
    cmd = f"mmseqs createdb {fasta} {db}"
    print(exec_mmseq(cmd))

    db_clustered = os.path.join(tmp_dir, "DB_clu")
    cmd = f"mmseqs linclust {db} {db_clustered} {tmp_dir} --min-seq-id {str(args.similarity_thresh)}"
    res = exec_mmseq(cmd)
    num_clusters = re.findall(r"Number of clusters: (\d+)", res)
    if len(num_clusters):
        print(f"Number of clusters: {num_clusters[0]}")
    else:
        print(res)
        raise ValueError("cluster failed!")

    tsv = os.path.join(tmp_dir, "DB_clu.tsv")
    cmd = f"mmseqs createtsv {db} {db} {db_clustered} {tsv}"
    print(exec_mmseq(cmd))

    with open(tsv, "r") as fin:
        entries = fin.read().strip().split("\n")
    clust = list(set([int(e.split("\t")[0]) for e in entries]))
    all_seqs_clus = [all_seqs[i] for i in clust]
    all_pos_clus = [all_pos[i] for i in clust]
    import pickle

    with open(args.out_path, "wb") as f:
        pickle.dump({"seqs": all_seqs_clus, "pos": all_pos_clus}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_path",
        type=str,
        default="data.pkl",
        help="output pkl file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Data",
        help="destination for raw files",
    )
    parser.add_argument(
        "--similarity-thresh",
        type=float,
        default=0.4,
        help="Similarity thereshold for clusring sequences",
    )
    clean(parser.parse_args())
