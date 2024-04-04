# shard NMR dataset (optimal for cluster computer use)

import argparse
import os
import zipfile

import webdataset as wds
import yaml


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dir", type=str, default="./data/nmr")
    parser.add_argument("-d", "--dest_dir", type=str, default="./data/nmr")
    parser.add_argument("-pc", "--percent", type=int, default=100)
    parser.add_argument("-sc", "--shard_count", type=int, default=4)
    return parser


def get_dataset_size(src_dir):
    src_zip = zipfile.ZipFile(os.path.join(src_dir, "NMR_Dataset.zip"))
    size_dict = dict()
    for split in ["train", "val", "test"]:
        metadata = yaml.safe_load(src_zip.read("NMR_Dataset/metadata.yaml"))
        split_dict = dict()

        for key in metadata.keys():
            base_path = f"NMR_Dataset/{key}"
            metadata[key]["list"] = [
                f"{base_path}/{dir_name.decode('utf-8')}"
                for dir_name in src_zip.read(f"{base_path}/{split}.lst").split()
            ]

            class_cnt = len(metadata[key]["list"])
            print(f"{metadata[key]['name']}: {class_cnt}")
            split_dict[key] = class_cnt

        size_dict[split] = split_dict
    return size_dict


def shard_dataset(src_dir, size_dict, dest_dir, split="test", percent=100, shard_cnt=4):
    src_zip = zipfile.ZipFile(os.path.join(src_dir, "NMR_Dataset.zip"))
    metadata = yaml.safe_load(src_zip.read("NMR_Dataset/metadata.yaml"))
    dest_dir = os.path.join(
        dest_dir, "_".join(("NMR_sharded", str(percent), str(shard_cnt)))
    )
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    split_dict = size_dict[split]

    for key in metadata.keys():
        base_path = f"NMR_Dataset/{key}"
        metadata[key]["list"] = [
            f"{base_path}/{dir_name.decode('utf-8')}"
            for dir_name in src_zip.read(f"{base_path}/{split}.lst").split()
        ]

    shard_idx = 0
    sample_no = 0
    limit = round(percent / 100 * sum(split_dict.values())) // shard_cnt

    tar_sink = wds.TarWriter(
        os.path.join(dest_dir, f"NMR-{split}-{shard_idx:02}.tar"), encoder=False
    )
    for key in metadata.keys():
        for dir_name in metadata[key]["list"]:

            sample = {"__key__": dir_name.split("/")[-1]}
            for i in range(24):
                fname = f"{i:04}.png"
                sample[fname] = src_zip.read(f"{dir_name}/image/{fname}")

            sample["cameras"] = src_zip.read(f"{dir_name}/cameras.npz")
            tar_sink.write(sample)
            sample_no += 1
            if (sample_no == limit) and (shard_idx < (shard_cnt - 1)):
                sample_no = 0
                shard_idx += 1
                tar_sink = wds.TarWriter(
                    os.path.join(dest_dir, f"NMR-{split}-{shard_idx:02}.tar"),
                    encoder=False,
                )


def main(args):
    size_dict = get_dataset_size(args.src_dir)
    for split in ["train", "val", "test"]:
        shard_dataset(
            args.src_dir,
            size_dict,
            args.dest_dir,
            split,
            args.percent,
            args.shard_count,
        )


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
