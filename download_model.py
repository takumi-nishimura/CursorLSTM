import huggingface_hub

model_id = "cyberagent/calm2-7b"
local_dir = "model/"
huggingface_hub.snapshot_download(model_id, local_dir=local_dir+model_id, local_dir_use_symlinks=False)
