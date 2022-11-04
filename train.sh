CUDA_VISIBLE_DEVICES=0 python region_dsl_am.py --batch_size 512  --epochs 150 --num_workers 4 --learning_rate 1e-1 --loss_type arcface --optim_type SGD --dataset neuro_face --model resnet18  
CUDA_VISIBLE_DEVICES=0 python region_dsl_am.py --batch_size 512  --epochs 150 --num_workers 4 --learning_rate 1e-1 --loss_type cosface --optim_type SGD --dataset neuro_face --model resnet18 
CUDA_VISIBLE_DEVICES=0 python region_dsl_am.py --batch_size 512  --epochs 150 --num_workers 4 --learning_rate 1e-1 --loss_type sphereface2 --optim_type SGD --dataset neuro_face --model resnet18 
