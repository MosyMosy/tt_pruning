CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method source_only
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method prototype_prune


CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scanobject_src_only.pth --method source_only
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scanobject_src_only.pth --method prototype_prune


CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method source_only
CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method prototype_prune