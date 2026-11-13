# saves the SlimPajama-6B dataset to a binary file for training, same format as data/openwebtext/prepare.py
# https://huggingface.co/datasets/DKYoon/SlimPajama-6B is a ~1% random sample of SlimPajama-627B
# (7 sources: CommonCrawl, C4, GitHub, Books, ArXiv, Wikipedia, StackExchange)

import os
import argparse
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset, DatasetDict  # huggingface datasets

# number of training documents to use. None = the full dataset (~5.5M docs, 14GB download).
# a subset is streamed, so only those documents are downloaded. the data is already shuffled,
# so the first n documents are a random sample with the natural mix of sources.
# ~1,100 tokens per doc on average, e.g. 100_000 docs ~= 110M tokens (~220MB train.bin).
# can also be set with --num_docs
num_docs = 100_000

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2. under SLURM, use the cpus allocated to the job
num_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

enc = tiktoken.get_encoding("gpt2")


def stream_docs(split, n):
    # yields the first n documents of a split (all of them if n is None) without downloading the whole dataset
    ds = load_dataset("DKYoon/SlimPajama-6B", split=split, streaming=True)
    if n is not None:
        ds = ds.take(n)
    for example in ds:
        yield {"text": example["text"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_docs",
        type=lambda s: None if s.lower() in ("none", "all", "full") else int(s),
        default=num_docs,
        help="number of training documents to use, or 'all' for the full dataset",
    )
    num_docs = parser.parse_args().num_docs

    print(f"streaming {'all' if num_docs is None else f'the first {num_docs:,}'} training documents of SlimPajama-6B")
    # slimpajama comes with its own validation split (9,347 docs, ~40MB), which is always used in full
    # so that val loss is comparable across runs with different num_docs
    dataset = DatasetDict(
        {
            "train": Dataset.from_generator(stream_docs, gen_kwargs={"split": "train", "n": num_docs}),
            "val": Dataset.from_generator(stream_docs, gen_kwargs={"split": "validation", "n": None}),
        }
    )

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example["text"])  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = min(1024, len(dset))  # avoid empty shards on small subsets

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"{split}: {len(dset):,} docs, {int(arr_len):,} tokens -> {filename}")

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
