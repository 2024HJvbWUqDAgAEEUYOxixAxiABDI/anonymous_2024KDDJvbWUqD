python main-batch.py \
--method hyperformer \
--dataset ogbn-proteins \
--metric rocauc \
--lr 0.001 \
--hidden_channels 256 \
--gnn_num_layers 2 \
--gnn_dropout 0. \
--gnn_use_residual \
--gnn_use_weight \
--gnn_use_bn \
--gnn_use_act \
--trans_num_layers 1 \
--trans_dropout 0. \
--weight_decay 0. \
--trans_use_residual \
--trans_use_weight \
--graph_weight 0.5 \
--batch_size 10000 \
--seed 123 \
--runs 1 \
--epochs 1000 \
--eval_step 5 \
--device 0 \
--power_k 1.5 \
--data_dir $data_dir