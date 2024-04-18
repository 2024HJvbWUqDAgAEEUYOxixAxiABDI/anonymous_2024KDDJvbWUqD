wandb_name=basic
data_dir=$hypformer_data_dir
python main-batch.py \
    --method hyperformer \
    --dataset amazon2m \
    --metric acc \
    --lr 0.005 \
    --hidden_channels 256 \
    --gnn_num_layers 3 \
    --gnn_dropout 0.0 \
    --weight_decay 0. \
    --gnn_use_residual \
    --gnn_use_weight \
    --gnn_use_bn \
    --gnn_use_init \
    --gnn_use_act \
    --trans_num_layers 1 \
    --trans_dropout 0. \
    --trans_use_residual \
    --trans_use_weight \
    --trans_use_bn \
    --use_graph \
    --graph_weight 0.5 \
    --batch_size 100000 \
    --seed 123 \
    --runs 1 \
    --epochs 200 \
    --eval_step 1 \
    --device 0 \
    --data_dir $data_dir
