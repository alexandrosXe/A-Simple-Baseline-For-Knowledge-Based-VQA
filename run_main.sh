#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=1:0:0     # XX hours runtime
#$ -l h_vmem=11G      # 11G RAM per core
#$ -pe smp 8          # 8 cores per GPU
#$ -l gpu=1           # request 1 GPU
#$ -l gpu_type=ampere  
##$ -m se


source /data/home/eey569/a_simple_baseline_for_kb_vqa/env/bin/activate


python /data/home/eey569/a_simple_baseline_for_kb_vqa/main.py \
    --dataset ok_vqa \
    --evaluation_set val \
    --train_annotations_path annotations/ok_vqa/train_annots_fixed.csv.zip \
    --val_annotations_path annotations/ok_vqa/val_annots_fixed.csv.zip \
    --test_annotations_path None \
    --train_images_dir /path_to_the_train_images/ \
    --val_images_dir /path_to_the_val_images/ \
    --test_images_dir None \
    --n_shots 10 \
    --k_ensemble 5 \
    --no_of_captions 9 \
    --use_mcan_examples False \
    --mcan_examples_path mcan_examples/ok_vqa/examples.json \
    --llama_path meta-llama/Llama-2-13b-hf \
    --train_captions_path question_related_captions/ok_vqa/train_data_qr_captions_csv \
    --val_captions_path question_related_captions/ok_vqa/val_data_qr_captions_csv \
    --test_captions_path None \
    --blip_train_question_embedds_path blip_embedds/ok_vqa/blip_normalized_q_embedds/blip_train_question_embedds.csv.zip \
    --blip_train_image_embedds_path blip_embedds/ok_vqa/blip_normalized_i_embedds/blip_train_image_embedds.csv.zip \
    --blip_val_question_embedds_path blip_embedds/ok_vqa/blip_normalized_q_embedds/blip_val_question_embedds.csv.zip \
    --blip_val_image_embedds_path blip_embedds/ok_vqa/blip_normalized_i_embedds/blip_val_image_embedds.csv.zip \
    --path_to_save_preds results/ok_vqa_val_with_mcan_llama2.csv


# python /data/home/eey569/a_simple_baseline_for_kb_vqa/main.py \
#     --dataset a_ok_vqa \
#     --evaluation_set val \
#     --train_annotations_path annotations/a_ok_vqa/a_ok_vqa_train_fixed_annots.csv.zip \
#     --val_annotations_path  annotations/a_ok_vqa/a_ok_vqa_val_fixed_annots.csv.zip \
#     --test_annotations_path annotations/a_ok_vqa/a_ok_vqa_test_fixed_annots.csv.zip \
#     --train_images_dir /path_to_the_train_images/ \
#     --val_images_dir /path_to_the_val_images/ \
#     --test_images_dir /path_to_the_test_images/ \
#     --n_shots 10 \
#     --k_ensemble 5 \
#     --no_of_captions 9 \
#     --use_mcan_examples True \
#     --mcan_examples_path mcan_examples/a_ok_vqa/examples_aokvqa_val.json \
#     --llama_path meta-llama/Llama-2-13b-hf \
#     --train_captions_path question_related_captions/a_ok_vqa/a_ok_vqa_train_qr_captions.csv.zip \
#     --val_captions_path question_related_captions/a_ok_vqa/a_ok_vqa_val_qr_captions.csv.zip \
#     --test_captions_path question_related_captions/a_ok_vqa/a_ok_vqa_test_qr_captions.csv.zip \
#     --blip_train_question_embedds_path blip_embedds/a_ok_vqa/blip_normalized_q_embedds/blip_train_question_embedds.csv.zip \
#     --blip_train_image_embedds_path blip_embedds/a_ok_vqa/blip_normalized_i_embedds/blip_train_image_embedds.csv.zip \
#     --blip_val_question_embedds_path blip_embedds/a_ok_vqa/blip_normalized_q_embedds/blip_val_question_embedds.csv.zip \
#     --blip_val_image_embedds_path blip_embedds/a_ok_vqa/blip_normalized_i_embedds/blip_val_image_embedds.csv.zip \
#     --blip_test_question_embedds_path blip_embedds/a_ok_vqa/blip_normalized_q_embedds/blip_test_question_embedds.csv.zip \
#     --blip_test_image_embedds_path blip_embedds/a_ok_vqa/blip_normalized_i_embedds/blip_test_image_embedds.csv.zip \
#     --path_to_save_preds results/a_ok_vqa_val_with_mcan_llama2.csv

