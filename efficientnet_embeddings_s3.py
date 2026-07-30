#!/usr/bin/env python3
"""
efficientnet_embeddings_s3.py

Extract EfficientNetV2 feature embeddings for a folder of single-channel cell-crop
TIFFs in S3, writing ONE embedding per TIFF in the SAME layout the DINOv3 model uses, so
the two are a drop-in comparison.

OUTPUT FORMAT (matches the DINOv3 SageMaker Batch Transform results)
-------------------------------------------------------------------
The reference model at s3://saguaro-nmd-poc-dev-results/infer/embeddings/base-dinov3-1/
writes one object per input image, named "<input>.tiff.out", each a NumPy .npy file
holding a single float32 embedding vector (768-d for DINOv3 ViT-B). This script mirrors
that exactly:

    <s3-output>/<input-basename>.tiff.out   # e.g. P07_Run01_6h_B03_1_cell63_DNA.tiff.out
    contents: np.save of a float32 vector (1280-d for tf_efficientnetv2_l)

So downstream code that loads the DINOv3 .out files loads these identically -- only the
vector length differs (1280 vs 768), which is expected across models. The filename carries
identity (plate/run/tp/well/site/cell/channel); no metadata is stored inside the file.
(Use --out-encoding raw to write headerless little-endian float32 instead of .npy.)

FAITHFUL REPRODUCTION OF THE ORIGINAL EMBEDDING STEP (Cellpose_GPU_s3fs.py)
--------------------------------------------------------------------------
    load uint8 crop -> Image.fromarray(...).convert("RGB")
    -> AutoImageProcessor -> AutoModel(timm/tf_efficientnetv2_l.in21k).pooler_output
    (fp16 autocast on CUDA -> float32), 1280-d.
The finetune_subset crops are already 8-bit min-max scaled (uint8) from export time, so no
re-scaling is applied -> byte-for-byte what the original pipeline fed the model. If pointed
at RAW (non-uint8) crops, scale_to_8bit is applied automatically so behaviour still matches.

PARALLELISM (N replicas across GPUs, like the Cellpose producer/consumer)
------------------------------------------------------------------------
Each worker process loads its own model replica, pinned to a GPU via CUDA_VISIBLE_DEVICES
(set before CUDA init -> sees only its GPU as cuda:0; two workers can share one physical
GPU). Work is handed out as chunks of keys through a shared queue (dynamic load balancing);
inside a worker a thread pool overlaps S3 download/upload with GPU compute.

RESUME
------
--resume lists the objects already present under --s3-output and skips those inputs, so a
long run can be restarted safely.

DEPENDENCIES (run on a CUDA GPU box; CPU works but is slow)
    pip install "torch" "transformers>=4.44" "timm>=0.9" tifffile pillow numpy boto3 tqdm

EXAMPLE (your 4 replicas: 2 GPUs, 2 models each)
    python efficientnet_embeddings_s3.py \
      --s3-input  s3://newmath-poc-trainningdata/Phenotypic_screen_HY-L022-custom_U2OS/finetune_subset/ \
      --s3-output s3://<bucket-you-can-write>/infer/embeddings/efficientnetv2l-1/ \
      --replicas-per-gpu 2 --batch-size 128 --io-workers 8 --resume

    # if the target bucket enforces SSE-KMS (like the DINOv3 one):
    #   --sse aws:kms --kms-key-id <key-arn>
    # quick checks:  --limit 64      |     --dry-run
"""

import io
import os
import gc
import math
import time
import logging
import argparse
import functools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

import boto3

import torch
import torch.multiprocessing as mp
from transformers import AutoImageProcessor, AutoModel

# --- Config (defaults match Cellpose_GPU_s3fs.py) ---
MODEL_NAME = "timm/tf_efficientnetv2_l.in21k"
FEATURE_LENGTH = 1280

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# --- Helpers ---
def scale_to_8bit(image):
    """Per-crop min-max stretch to uint8, IDENTICAL to the inference pipeline.

    Applied only as a fallback for non-uint8 (raw) crops. For the already-8-bit
    finetune_subset TIFFs this is never called, so the model sees the exact same bytes
    as the original run.
    """
    min_val, max_val = np.min(image), np.max(image)
    if max_val == min_val:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = 255.0 * (image.astype(np.float32) - min_val) / (max_val - min_val)
    return scaled.astype(np.uint8)


def parse_s3_uri(uri):
    """'s3://bucket/some/prefix/' -> ('bucket', 'some/prefix/')."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {uri}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def parse_meta(bucket, key):
    """Parse {plate}_{run}_{tp}_{well}_{site}_cell{k}_{channel}.tiff -> metadata (dry-run only)."""
    filename = key.rsplit("/", 1)[-1]
    stem = filename[:-5] if filename.lower().endswith(".tiff") else os.path.splitext(filename)[0]
    t = stem.split("_")
    cell_tok = t[-2] if len(t) >= 2 else ""
    return {
        "filename": filename,
        "channel": t[-1],
        "cell_index": int(cell_tok[4:]) if cell_tok.startswith("cell") and cell_tok[4:].isdigit() else None,
        "site": t[-3] if len(t) >= 3 else None,
        "well": t[-4] if len(t) >= 4 else None,
        "name_prefix": "_".join(t[:-4]) if len(t) >= 5 else "",
        "cell_id": "_".join(t[:-1]),   # groups the channels of one cell
    }


def list_objects(s3, bucket, prefix, suffix=None):
    """Return all keys under prefix (optionally filtered by suffix), paginated."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if suffix is None or k.lower().endswith(suffix):
                keys.append(k)
    return keys


def download_decode(s3, bucket, key):
    """Fetch one TIFF from S3 and return (key, uint8 2D array) or (key, None) on failure."""
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        arr = tifffile.imread(io.BytesIO(body))
        if arr.ndim == 3 and arr.shape[-1] == 1:   # (H, W, 1) -> (H, W)
            arr = arr[..., 0]
        if arr.dtype != np.uint8:                  # fallback for raw (unscaled) crops
            arr = scale_to_8bit(arr)
        return key, arr
    except Exception as e:
        logging.warning(f"Failed to read/decode {key}: {e}")
        return key, None


def embed_batch(pil_images, processor, model, device):
    """processor + model on a list of PIL RGB images -> (N, FEATURE_LENGTH) float32.

    Mirrors the original: fp16 autocast on CUDA, pooler_output, cast back to float32.
    """
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")
    ):
        outputs = model(**inputs)
    return outputs.pooler_output.float().cpu().numpy()


def embed_with_backoff(pil_images, processor, model, device):
    """embed_batch but halves the batch and retries on CUDA OOM (replicas share GPU vRAM)."""
    try:
        return embed_batch(pil_images, processor, model, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        if len(pil_images) == 1:
            raise
        mid = len(pil_images) // 2
        first = embed_with_backoff(pil_images[:mid], processor, model, device)
        second = embed_with_backoff(pil_images[mid:], processor, model, device)
        return np.vstack([first, second])


def serialize_embedding(vec, encoding):
    """Serialize one embedding vector to bytes.

    'npy'  -> np.save format (matches the DINOv3 .out files: .npy header + float32 data).
    'raw'  -> headerless little-endian float32 bytes.
    """
    vec = np.ascontiguousarray(vec, dtype="<f4")
    if encoding == "npy":
        buf = io.BytesIO()
        np.save(buf, vec)
        return buf.getvalue()
    return vec.tobytes()


def process_chunk(s3, in_bucket, out_bucket, out_prefix, chunk_keys, processor, model,
                  device, batch_size, encoding, extra_args, pool, report):
    """Embed a chunk of input keys and write one '<input>.out' object per image.

    Downloads and uploads run on the shared thread pool to overlap I/O with GPU compute.
    Returns the list of keys that failed to decode. `report(n)` advances the progress bar.
    """
    failures = []
    download = functools.partial(download_decode, s3, in_bucket)

    def put_one(item):
        key, data = item
        s3.put_object(Bucket=out_bucket, Key=key, Body=data, **extra_args)

    for start in range(0, len(chunk_keys), batch_size):
        batch_keys = chunk_keys[start:start + batch_size]
        good_keys, arrays = [], []
        for key, arr in pool.map(download, batch_keys):
            if arr is None:
                failures.append(key)
            else:
                good_keys.append(key)
                arrays.append(arr)

        if good_keys:
            pil_images = [Image.fromarray(a).convert("RGB") for a in arrays]
            emb = embed_with_backoff(pil_images, processor, model, device)
            payloads = []
            for key, vec in zip(good_keys, emb):
                base = key.rsplit("/", 1)[-1]              # e.g. ..._DNA.tiff
                out_key = f"{out_prefix}/{base}.out"       # -> ..._DNA.tiff.out
                payloads.append((out_key, serialize_embedding(vec, encoding)))
            list(pool.map(put_one, payloads))

        report(len(batch_keys))

    return failures


def worker_main(worker_id, gpu_id, in_bucket, out_bucket, out_prefix, model_name,
                batch_size, io_workers, encoding, extra_args, cpu_threads,
                chunk_queue, progress, failures):
    """One replica: pin to a GPU, load a model, drain the chunk queue."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # before any CUDA call
    torch.set_num_threads(max(1, cpu_threads))            # avoid CPU oversubscription

    use_cuda = (gpu_id is not None) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    logging.info(f"Worker {worker_id} on gpu={gpu_id} device={device} loading {model_name} ...")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    s3 = boto3.client("s3")
    pool = ThreadPoolExecutor(max_workers=io_workers)

    def report(n):
        with progress.get_lock():
            progress.value += n

    try:
        while True:
            item = chunk_queue.get()
            if item is None:  # sentinel -> no more work
                break
            chunk_idx, chunk_keys = item
            t0 = time.time()
            fails = process_chunk(s3, in_bucket, out_bucket, out_prefix, chunk_keys,
                                  processor, model, device, batch_size, encoding,
                                  extra_args, pool, report)
            if fails:
                failures.extend(fails)
            logging.info(f"Worker {worker_id} chunk {chunk_idx}: {len(chunk_keys)} imgs "
                         f"({len(fails)} failed) in {time.time() - t0:.1f}s")
    finally:
        pool.shutdown(wait=True)


def resolve_gpus(args):
    """Return (list of GPU ids; [None] means CPU) and the per-GPU replica count."""
    device_count = torch.cuda.device_count()
    if args.gpus:
        return [int(x) for x in args.gpus.split(",") if x.strip() != ""], max(1, args.replicas_per_gpu)
    if device_count > 0:
        return list(range(device_count)), max(1, args.replicas_per_gpu)
    logging.warning("No CUDA device found -> running a single CPU worker (slow).")
    return [None], 1


def main(args):
    s3 = boto3.client("s3")
    in_bucket, in_prefix = parse_s3_uri(args.s3_input)
    out_bucket, out_prefix = parse_s3_uri(args.s3_output)
    out_prefix = out_prefix.rstrip("/")

    logging.info(f"Listing TIFFs under s3://{in_bucket}/{in_prefix} ...")
    keys = sorted(list_objects(s3, in_bucket, in_prefix, ".tiff"))
    if args.limit:
        keys = keys[:args.limit]
    if not keys:
        logging.error("No .tiff files found. Check --s3-input.")
        return

    if args.resume:
        existing = {k.rsplit("/", 1)[-1] for k in list_objects(s3, out_bucket, out_prefix, ".out")}
        before = len(keys)
        keys = [k for k in keys if f"{k.rsplit('/', 1)[-1]}.out" not in existing]
        logging.info(f"Resume: {before - len(keys)} already done, {len(keys)} remaining.")
    if not keys:
        logging.info("Nothing to do (all outputs already exist).")
        return

    if args.dry_run:
        logging.info(f"{len(keys)} TIFFs to process. Output pattern: "
                     f"s3://{out_bucket}/{out_prefix}/<name>.tiff.out ({args.out_encoding})")
        for k in keys[:5]:
            logging.info(f"  sample: {parse_meta(in_bucket, k)}")
        logging.info("Dry run: not loading the model or writing output.")
        return

    extra_args = {}
    if args.sse:
        extra_args["ServerSideEncryption"] = args.sse
    if args.kms_key_id:
        extra_args["SSEKMSKeyId"] = args.kms_key_id

    gpus, replicas = resolve_gpus(args)
    total_workers = len(gpus) * replicas
    cpu_threads = max(1, (os.cpu_count() or 8) // total_workers)

    chunks = [(i, keys[i * args.chunk_size:(i + 1) * args.chunk_size])
              for i in range(math.ceil(len(keys) / args.chunk_size))]
    logging.info(f"{len(keys)} imgs -> {len(chunks)} chunk(s) of up to {args.chunk_size}. "
                 f"Launching {total_workers} replica(s): gpus={gpus} x {replicas}/gpu, "
                 f"batch_size={args.batch_size}, torch_threads/worker={cpu_threads}, "
                 f"encoding={args.out_encoding}.")

    manager = mp.Manager()
    failures = manager.list()
    progress = mp.Value("i", 0)
    chunk_queue = mp.Queue()
    for chunk in chunks:
        chunk_queue.put(chunk)
    for _ in range(total_workers):
        chunk_queue.put(None)  # one sentinel per worker

    workers = []
    for w in range(total_workers):
        gpu_id = gpus[w % len(gpus)]
        p = mp.Process(
            target=worker_main,
            args=(w, gpu_id, in_bucket, out_bucket, out_prefix, args.model_name,
                  args.batch_size, args.io_workers, args.out_encoding, extra_args,
                  cpu_threads, chunk_queue, progress, failures),
            name=f"Worker-{w}-gpu{gpu_id}",
        )
        p.start()
        workers.append(p)

    t0 = time.time()
    with tqdm(total=len(keys), desc="Embedding", unit="img") as pbar:
        while any(p.is_alive() for p in workers):
            time.sleep(1)
            with progress.get_lock():
                cur = progress.value
            pbar.update(cur - pbar.n)
        with progress.get_lock():
            cur = progress.value
        pbar.update(cur - pbar.n)

    for p in workers:
        p.join()

    elapsed = time.time() - t0
    fail_list = list(failures)
    n_ok = len(keys) - len(fail_list)
    logging.info(f"Done. {n_ok}/{len(keys)} embedded in {elapsed:.1f}s "
                 f"({n_ok / max(elapsed, 1e-6):.1f} img/s). Output: s3://{out_bucket}/{out_prefix}/")
    if fail_list:
        logging.warning(f"{len(fail_list)} files failed to decode. First few: {fail_list[:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract EfficientNetV2 embeddings (one .npy '.out' per TIFF, matching the "
                    "DINOv3 batch-transform layout), parallelised across GPUs/replicas.")
    parser.add_argument("--s3-input", required=True,
                        help="S3 prefix containing the cell-crop .tiff files.")
    parser.add_argument("--s3-output", required=True,
                        help="S3 prefix for the '<input>.tiff.out' embedding files "
                             "(mirror the reference: .../infer/embeddings/<model-name>/).")
    parser.add_argument("--model-name", default=MODEL_NAME, help="timm/HF model id.")
    parser.add_argument("--out-encoding", choices=["npy", "raw"], default="npy",
                        help="npy = np.save format (matches DINOv3 .out). raw = headerless float32.")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU ids, e.g. '0,1'. Default: all detected.")
    parser.add_argument("--replicas-per-gpu", type=int, default=1,
                        help="Model replicas per GPU (you can fit 2). Total workers = gpus x this.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Per-replica GPU batch size (auto-halves on OOM).")
    parser.add_argument("--io-workers", type=int, default=8,
                        help="S3 download/upload threads PER replica.")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Input keys per queue item (load-balancing granularity). Keep it "
                             ">> total replicas so every replica stays busy.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip inputs whose '<input>.tiff.out' already exists under --s3-output.")
    parser.add_argument("--sse", default=None,
                        help="ServerSideEncryption for outputs, e.g. 'aws:kms' or 'AES256'.")
    parser.add_argument("--kms-key-id", default=None,
                        help="KMS key id/arn (with --sse aws:kms) if the output bucket requires it.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N files (for a quick test).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files + show parsed metadata, then exit (no model/output).")
    args = parser.parse_args()

    try:
        mp.set_start_method("spawn", force=True)  # required for CUDA in child processes
    except RuntimeError:
        pass
    main(args)
