# set the mirror if necessary
# export HF_ENDPOINT=https://hf-mirror.com

from huggingface_hub import snapshot_download, hf_hub_download

snapshot_download(repo_id='Michael4933/MGrounding-630k', # Michael4933/MIG-Bench
                  repo_type='dataset',
                  local_dir='Your_Own_Path',
                  force_download=False,
                  max_workers=32,
                )