# Prototype Pruning
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method prototype_prune
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method prototype_prune
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method prototype_prune

CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method prototype_prune --BN_reset
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method prototype_prune --BN_reset
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method prototype_prune --BN_reset



# Cla_Token Pruning
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method cls_prune
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method cls_prune
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method cls_prune

CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method cls_prune --BN_reset
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method cls_prune --BN_reset
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method cls_prune --BN_reset

#-------------------------------------------------- Baselines------------------------------------------------

# source only
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta/tta_scanobj.yaml --ckpts checkpoints/scan_object_src_only.pth --method source_only
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta/tta_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method source_only
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta/tta_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method source_only

CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta/tta_scanobj.yaml --ckpts checkpoints/scan_object_src_only.pth --method source_only --BN_reset
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta/tta_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method source_only --BN_reset
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta/tta_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method source_only --BN_reset



# BFTT3D
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method bftt3d
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method bftt3d
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method bftt3d

CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method bftt3d --BN_reset
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method bftt3d --BN_reset
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method bftt3d --BN_reset



# tent
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/scanobject/tta_tent.yaml --ckpts checkpoints/scan_object_src_only.pth --method tent
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/modelnet/tta_tent.yaml --ckpts checkpoints/modelnet_src_only.pth --method tent
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/shapenet/tta_tent.yaml --ckpts checkpoints/shapenet_src_only.pth --method tent


# rotnet
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/scanobject/tta_rotnet.yaml --ckpts checkpoints/scan_object_src_only.pth --method rotnet
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/modelnet/tta_rotnet.yaml --ckpts cfgs/baselines/modelnet/tta_rotnet.yaml --method rotnet
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/shapenet/tta_rotnet.yaml --ckpts checkpoints/shapenet_src_only.pth --method rotnet


# t3a
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/scanobject/tta_t3a.yaml --ckpts checkpoints/scan_object_src_only.pth --method t3a
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/modelnet/tta_t3a.yaml --ckpts checkpoints/modelnet_src_only.pth --method t3a
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/shapenet/tta_t3a.yaml --ckpts checkpoints/shapenet_src_only.pth --method t3a


# shot
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/scanobject/tta_shot.yaml --ckpts checkpoints/scan_object_src_only.pth --method shot
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/modelnet/tta_shot.yaml --ckpts checkpoints/modelnet_src_only.pth --method shot
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/shapenet/tta_shot.yaml --ckpts checkpoints/shapenet_src_only.pth --method shot


# dua
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/scanobject/tta_dua.yaml --ckpts checkpoints/scan_object_src_only.pth --method dua
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/modelnet/tta_dua.yaml --ckpts checkpoints/modelnet_src_only.pth --method dua
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/baselines/shapenet/tta_dua.yaml --ckpts checkpoints/shapenet_src_only.pth --method dua