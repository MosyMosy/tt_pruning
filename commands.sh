export CUDA_VISIBLE_DEVICES=3

# Prototype Pruning
python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method prototype_prune
python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method prototype_prune
python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method prototype_prune

python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method prototype_prune --BN_reset
python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method prototype_prune --BN_reset
python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method prototype_prune --BN_reset

# Cla_Token Pruning
python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method cls_prune
python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method cls_prune
python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method cls_prune

python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method cls_prune --BN_reset
python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method cls_prune --BN_reset
python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method cls_prune --BN_reset

#-------------------------------------------------- Baselines------------------------------------------------

# source only
python main.py --config cfgs/tta/tta_scanobj.yaml --ckpts checkpoints/scan_object_src_only.pth --method source_only
python main.py --config cfgs/tta/tta_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method source_only
python main.py --config cfgs/tta/tta_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method source_only

python main.py --config cfgs/tta/tta_scanobj.yaml --ckpts checkpoints/scan_object_src_only.pth --method source_only --BN_reset
python main.py --config cfgs/tta/tta_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method source_only --BN_reset
python main.py --config cfgs/tta/tta_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method source_only --BN_reset

# -----------------------------------------------------------------------------------------------------------

# BFTT3D
python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method bftt3d
python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method bftt3d
python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method bftt3d

python main.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method bftt3d --BN_reset
python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method bftt3d --BN_reset
python main.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method bftt3d --BN_reset

# -----------------------------------------------------------------------------------------------------------

# tent
python main.py --config cfgs/baselines/scanobject/tta_tent.yaml --ckpts checkpoints/scan_object_src_only.pth --method tent
python main.py --config cfgs/baselines/modelnet/tta_tent.yaml --ckpts checkpoints/modelnet_src_only.pth --method tent
python main.py --config cfgs/baselines/shapenet/tta_tent.yaml --ckpts checkpoints/shapenet_src_only.pth --method tent

python main.py --config cfgs/baselines/scanobject/tta_tent.yaml --ckpts checkpoints/scan_object_src_only.pth --method tent --BN_reset
python main.py --config cfgs/baselines/modelnet/tta_tent.yaml --ckpts checkpoints/modelnet_src_only.pth --method tent --BN_reset
python main.py --config cfgs/baselines/shapenet/tta_tent.yaml --ckpts checkpoints/shapenet_src_only.pth --method tent --BN_reset

# -----------------------------------------------------------------------------------------------------------

# t3a
python main.py --config cfgs/baselines/scanobject/tta_t3a.yaml --ckpts checkpoints/scan_object_src_only.pth --method t3a
python main.py --config cfgs/baselines/modelnet/tta_t3a.yaml --ckpts checkpoints/modelnet_src_only.pth --method t3a
python main.py --config cfgs/baselines/shapenet/tta_t3a.yaml --ckpts checkpoints/shapenet_src_only.pth --method t3a

# -----------------------------------------------------------------------------------------------------------

# shot
python main.py --config cfgs/baselines/scanobject/tta_shot.yaml --ckpts checkpoints/scan_object_src_only.pth --method shot
python main.py --config cfgs/baselines/modelnet/tta_shot.yaml --ckpts checkpoints/modelnet_src_only.pth --method shot
python main.py --config cfgs/baselines/shapenet/tta_shot.yaml --ckpts checkpoints/shapenet_src_only.pth --method shot

# -----------------------------------------------------------------------------------------------------------

# dua
python main.py --config cfgs/baselines/scanobject/tta_dua.yaml --ckpts checkpoints/scan_object_src_only.pth --method dua
python main.py --config cfgs/baselines/modelnet/tta_dua.yaml --ckpts checkpoints/modelnet_src_only.pth --method dua
python main.py --config cfgs/baselines/shapenet/tta_dua.yaml --ckpts checkpoints/shapenet_src_only.pth --method dua

# -----------------------------------------------------------------------------------------------------------

# rotnet
python main.py --config cfgs/baselines/scanobject/tta_rotnet.yaml --ckpts checkpoints/scan_object_src_only.pth --method rotnet
python main.py --config cfgs/baselines/modelnet/tta_rotnet.yaml --ckpts cfgs/baselines/modelnet/tta_rotnet.yaml --method rotnet
python main.py --config cfgs/baselines/shapenet/tta_rotnet.yaml --ckpts checkpoints/shapenet_src_only.pth --method rotnet


export CUDA_VISIBLE_DEVICES=2

python test_prune-size.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method prototype_prune
python test_prune-size.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method prototype_prune
python test_prune-size.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method prototype_prune

python test_prune-size.py --config cfgs/tta_prune/tta_prune_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method prototype_prune  --BN_reset
python test_prune-size.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method prototype_prune  --BN_reset
python test_prune-size.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method prototype_prune  --BN_reset



python test_prune-size.py --config cfgs/tta_prune/tta_prune_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method prototype_prune  --BN_reset --exp_name batch_test
python test_prune-size.py --config cfgs/baselines/shapenet/tta_tent.yaml--ckpts checkpoints/shapenet_src_only.pth --method tent  --exp_name batch_test



python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method prototype_prune --BN_reset --exp_name time_eval
python main.py --config cfgs/tta_prune/tta_prune_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method bftt3d --exp_name time_eval

