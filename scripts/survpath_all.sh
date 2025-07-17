#!/bin/bash
echo "Script has started"
DATA_ROOT_DIR='/data/TCGA_Embed'
BASE_DIR="/data/SurvPath-com"
TYPE_OF_PATH="combine"
MODEL="survpath"
DIM1=8
DIM2=16
STUDIES=("blca" "br")
LRS=(0.0005)
DECAYS=(0.001)

for decay in ${DECAYS[@]}; do
    for lr in ${LRS[@]}; do
        for STUDY in ${STUDIES[@]}; do
            for fold in {0..4}; do  # 添加fold循环（0到4）
                CUDA_VISIBLE_DEVICES=0 nohup python main_all.py \
	   --fold $fold \
                    --study tcga_${STUDY} \
                    --task survival \
                    --split_dir splits \
                    --which_splits traincv \
                    --type_of_path $TYPE_OF_PATH \
                    --modality $MODEL \
                    --data_root_dir $DATA_ROOT_DIR/TCGA-${STUDY^^}/clam_gen_1024/conch_v1_5/ \
                    --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
                    --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} \
                    --results_dir results_titan_${STUDY} \
                    --batch_size 1 \
                    --lr $lr \
                    --opt radam \
                    --reg $decay \
                    --alpha_surv 0.5 \
                    --weighted_sample \
                    --max_epochs 100 \
                    --encoding_dim 768 \
                    --label_col survival_months_dss \
                    --k 5 \
                    --bag_loss nll_surv \
                    --n_classes 4 \
                    --num_patches 4096 \
                    --wsi_projection_dim 256 \
                    --encoding_layer_1_dim ${DIM1} \
                    --encoding_layer_2_dim ${DIM2} \
                    --encoder_dropout 0.25 \
                    > "${MODEL}_chief_${STUDY}_fold${fold}.out" 2>&1 &
            done
        done
    done
done

## CUDA_VISIBLE_DEVICES=0 nohup python main_all.py \
##> "${MODEL}_chief_${STUDY}_fold${fold}.out" 2>&1 &
# # chief ctranspath
# for decay in ${DECAYS[@]};
# do
#     for lr in ${LRS[@]};
#     do 
#         for STUDY in ${STUDIES[@]};
#         do
#             CUDA_VISIBLE_DEVICES=3 nohup python main.py \
#                 --study tcga_${STUDY} --task survival --split_dir splits \
#                 --which_splits traincv \
#                 --type_of_path $TYPE_OF_PATH --modality $MODEL \
#                 --data_root_dir $DATA_ROOT_DIR/TCGA-${STUDY^^}/clam_gen_1024/chief/ \
#                 --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
#                 --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} \
#                 --results_dir results_chief_${STUDY} \
#                 --batch_size 1 --lr $lr --opt radam --reg $decay \
#                 --alpha_surv 0.5 --weighted_sample --max_epochs 100 --encoding_dim 768 \
#                 --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 \
#                 --encoding_layer_1_dim ${DIM1} --encoding_layer_2_dim ${DIM2} --encoder_dropout 0.25 \
#                 > ${MODEL}_chief_${STUDY}.out 2>&1 & 
#         done 
#     done
# done 

# ctranspath
#for decay in ${DECAYS[@]};
#do
#    echo "Starting loop for decay: $decay"
#    for lr in ${LRS[@]};
#    do
#        echo "Starting loop for learning rate: $lr"
#        for STUDY in ${STUDIES[@]};
#        do
#            echo "Starting loop for study: tcga_${STUDY}"
#            echo "CUDA_VISIBLE_DEVICES=0 nohup python main.py \
#                --study tcga_${STUDY} --task survival --split_dir splits \
#                --which_splits traincv \
#                --type_of_path $TYPE_OF_PATH --modality $MODEL \
#                --data_root_dir $DATA_ROOT_DIR/TCGA-${STUDY^^}/clam_gen_1024/ctrans/ \
#                --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
#                --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} \
#                --results_dir results_ctrans_${STUDY} \
#                --batch_size 1 --lr $lr --opt radam --reg $decay \
#                --alpha_surv 0.5 --weighted_sample --max_epochs 100 --encoding_dim 768 \
#                --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 \
#                --encoding_layer_1_dim ${DIM1} --encoding_layer_2_dim ${DIM2} --encoder_dropout 0.25 \
#                > ${MODEL}_ctrans_${STUDY}.out 2>&1 &"
#
#            CUDA_VISIBLE_DEVICES=0 nohup python main.py \
#                --study tcga_${STUDY} --task survival --split_dir splits \
#                --which_splits traincv \
#                --type_of_path $TYPE_OF_PATH --modality $MODEL \
#                --data_root_dir $DATA_ROOT_DIR/TCGA-${STUDY^^}/clam_gen_1024/ctrans/ \
#                --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
#                --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} \
#                --results_dir results_ctrans_${STUDY} \
#                --batch_size 1 --lr $lr --opt radam --reg $decay \
#                --alpha_surv 0.5 --weighted_sample --max_epochs 100 --encoding_dim 768 \
#                --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 \
#                --encoding_layer_1_dim ${DIM1} --encoding_layer_2_dim ${DIM2} --encoder_dropout 0.25 \
#                > ${MODEL}_ctrans_${STUDY}.out 2>&1 &
#
#            echo "Completed execution for study: tcga_${STUDY}"
#        done
#        echo "Completed loop for learning rate: $lr"
#    done
#    echo "Completed loop for decay: $decay"
#done