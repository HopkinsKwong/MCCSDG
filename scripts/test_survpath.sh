#!/bin/bash
echo "Script has started"
DATA_ROOT_DIR='/data/TCGA_Embed' # where are the TCGA features stored?
BASE_DIR="/data/SurvPath-com" # where is the repo cloned?
TYPE_OF_PATH="combine" # what type of pathways?
MODEL="survpath" # what type of model do you want to train?
DIM1=8
DIM2=16
STUDIES=("blca" "br" "hnsc" "stad")
# STUDIES=("blca")
#TEST_STUDIES=("blca" "br" "hnsc" "stad")
TEST_FOLDER=("testcv")
# Learning rates and weight decays
LRS=(0.0005)
DECAYS=(0.001)


for STUDY in "${STUDIES[@]}"; do
    for lr in "${LRS[@]}"; do
        for decay in "${DECAYS[@]}"; do
            if [ "$STUDY" == "blca" ]; then
                CUDA_VISIBLE_DEVICES=0 python test.py \
                    --study tcga_blca --task survival --split_dir splits \
                    --which_splits "$TEST_FOLDER" \
                    --type_of_path $TYPE_OF_PATH --modality $MODEL \
                    --data_root_dir "$DATA_ROOT_DIR/TCGA-BLCA/clam_gen_1024/conch_v1_5/" \
                    --label_file datasets_csv/metadata/tcga_blca.csv \
                    --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/blca \
                    --results_dir results_titan_blca \
                    --batch_size 1 --lr "$lr" --opt radam --reg "$decay" \
                    --alpha_surv 0.5 --weighted_sample --max_epochs 100 --encoding_dim 768 \
                    --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 \
                    --encoding_layer_1_dim "$DIM1" --encoding_layer_2_dim "$DIM2" --encoder_dropout 0.25
            elif [ "$STUDY" == "br" ]; then
                CUDA_VISIBLE_DEVICES=0 python test.py \
                    --study tcga_br --task survival --split_dir splits \
                    --which_splits "$TEST_FOLDER" \
                    --type_of_path $TYPE_OF_PATH --modality $MODEL \
                    --data_root_dir "$DATA_ROOT_DIR/TCGA-BR/clam_gen_1024/conch_v1_5/" \
                    --label_file datasets_csv/metadata/tcga_br.csv \
                    --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/br \
                    --results_dir results_titan_br \
                    --batch_size 1 --lr "$lr" --opt radam --reg "$decay" \
                    --alpha_surv 0.5 --weighted_sample --max_epochs 100 --encoding_dim 768 \
                    --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 \
                    --encoding_layer_1_dim "$DIM1" --encoding_layer_2_dim "$DIM2" --encoder_dropout 0.25
            elif [ "$STUDY" == "hnsc" ]; then
                CUDA_VISIBLE_DEVICES=0 python test.py \
                    --study tcga_hnsc --task survival --split_dir splits \
                    --which_splits "$TEST_FOLDER" \
                    --type_of_path $TYPE_OF_PATH --modality $MODEL \
                    --data_root_dir "$DATA_ROOT_DIR/TCGA-HNSC/clam_gen_1024/conch_v1_5/" \
                    --label_file datasets_csv/metadata/tcga_hnsc.csv \
                    --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/hnsc \
                    --results_dir results_titan_hnsc \
                    --batch_size 1 --lr "$lr" --opt radam --reg "$decay" \
                    --alpha_surv 0.5 --weighted_sample --max_epochs 100 --encoding_dim 768 \
                    --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 \
                    --encoding_layer_1_dim "$DIM1" --encoding_layer_2_dim "$DIM2" --encoder_dropout 0.25
            elif [ "$STUDY" == "stad" ]; then
                CUDA_VISIBLE_DEVICES=0 python test.py \
                    --study tcga_stad --task survival --split_dir splits \
                    --which_splits "$TEST_FOLDER" \
                    --type_of_path $TYPE_OF_PATH --modality $MODEL \
                    --data_root_dir "$DATA_ROOT_DIR/TCGA-STAD/clam_gen_1024/conch_v1_5/" \
                    --label_file datasets_csv/metadata/tcga_stad.csv \
                    --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/stad \
                    --results_dir results_titan_stad \
                    --batch_size 1 --lr "$lr" --opt radam --reg "$decay" \
                    --alpha_surv 0.5 --weighted_sample --max_epochs 100 --encoding_dim 768 \
                    --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 \
                    --encoding_layer_1_dim "$DIM1" --encoding_layer_2_dim "$DIM2" --encoder_dropout 0.25
            fi
        done
    done
done