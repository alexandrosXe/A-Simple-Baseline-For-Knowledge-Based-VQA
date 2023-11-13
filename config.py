import os
import argparse


def get_config(sysv):
    parser = argparse.ArgumentParser(description='In-context learning variables.')
    parser.add_argument('--dataset', type=str, choices = ['ok_vqa', 'a_ok_vqa'], default='ok_vqa', help='dataset to use')
    parser.add_argument('--evaluation_set', type=str, choices = ['val', 'test'], default='val', help='Set to perform the experiments')
    parser.add_argument('--train_annotations_path', type=str, default=None, help='The path to the train annotations csv file')
    parser.add_argument('--val_annotations_path', type=str, default=None, help='The path to the train annotations csv file')
    parser.add_argument('--test_annotations_path', type=str, default=None, help='The path to the train annotations csv file')

    parser.add_argument('--train_images_dir', type=str, default=None, help='Path for the training images dir')
    parser.add_argument('--val_images_dir', type=str, default=None, help='Path for the val images dir')
    parser.add_argument('--test_images_dir', type=str, default=None, help='Path for the test images dir (only for A-OK-VQA)')

    parser.add_argument('--n_shots', type=int, default=10, help='Number of shots for in-context-learning')
    parser.add_argument('--k_ensemble', type=int, default=5, help='Number of ensmembles for in-context-learning')
    parser.add_argument('--no_of_captions', type=int, default=9, help='Number of question informative captions for in-context-learning')
    parser.add_argument('--use_mcan_examples', type=str, default="False", choices=["True", "False"], help='If true uses the mcan based shot selection strategy. If false uses the avg question and image similarity')
    parser.add_argument('--mcan_examples_path', type=str, default=None, help='The path to the json file containing the mcan examples')
    parser.add_argument('--llama_path', type=str, default=None, help='The path to the llama (1 or 2) weights')
    parser.add_argument('--blip_train_question_embedds_path', type=str, default=None, help='The path to the normalized blip train question embeddings')
    parser.add_argument('--blip_train_image_embedds_path', type=str, default=None, help='The path to the normalized blip train image embeddings')
    parser.add_argument('--blip_val_question_embedds_path', type=str, default=None, help='The path to the normalized blip val question embeddings')
    parser.add_argument('--blip_val_image_embedds_path', type=str, default=None, help='The path to the normalized blip val image embeddings')
    parser.add_argument('--blip_test_question_embedds_path', type=str, default=None, help='The path to the normalized blip test question embeddings (only for A-OK-VQA)')
    parser.add_argument('--blip_test_image_embedds_path', type=str, default=None, help='The path to the normalized blip test image embeddings (only for A-OK-VQA)')

    parser.add_argument('--train_captions_path', type=str, default=None, help='The path to the train question informative captions')
    parser.add_argument('--val_captions_path', type=str, default=None, help='The path to the val question informative captions')
    parser.add_argument('--test_captions_path', type=str, default=None, help='The path to the train question informative captions (only for A-OK-VQA)')

    parser.add_argument('--path_to_save_preds', type=str, default=None, help='Path to save the final predictions (needs to have .csv extension)')

    args, _ = parser.parse_known_args(sysv)

    return args 

