from transformers import AutoTokenizer
import numpy as np
from datasets import load_from_disk

from src.dataset.paligemma_processing import PaliGemmaProcessor, PaliGemmaVLAProcessor
from src.model.action.fast_tokenizer import UniversalActionProcessor

def test_Gemma_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "/share_data/checkpoints/paligemma-3b-pt-224", padding_side="right"
    )
    print(tokenizer)
    print(tokenizer.all_special_ids)
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    print(f"bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")

    input_string = "\n"

    output = tokenizer(
        input_string,
        max_length=16,
        padding="max_length",
        truncation=True,
    )

    for key, value in output.items():
        print(f"{key}: {value}")

    input_string = " "

    output = tokenizer(
        input_string,
        max_length=16,
        padding="max_length",
        truncation=True,
    )

    for key, value in output.items():
        print(f"{key}: {value}")

def test_PaliGemma_processor():
    tokenizer = AutoTokenizer.from_pretrained(
        "/share_data/checkpoints/paligemma-3b-pt-224", padding_side="right"
    )
    
    processor = PaliGemmaProcessor(
        tokenizer,
        num_image_tokens=256,
        max_seq_len=1024,
        ignore_index=-100,
        image_size=224,
    )

    text = "What should the robot do to move the object to the target location?"
    target = "Grasp the object first"
    images = np.random.rand(2, 224, 224, 3)
    output = processor(text, images, target)
    np.set_printoptions(threshold=np.inf)
    print(output)
    
def test_PaliGemmaVLA_processor():
    tokenizer = AutoTokenizer.from_pretrained(
        "/share_data/checkpoints/paligemma-3b-pt-224", padding_side="right"
    )
    fast_tokenizer = UniversalActionProcessor.from_pretrained(
        "/home/chenzhang/project/EgoVLA/outputs/2025.09.11/22.44_fast_tokenizer_train_fast_tokenizer/tokenizer"
    )

    processor = PaliGemmaVLAProcessor(
        tokenizer,
        fast_tokenizer,
        num_image_tokens=256,
        max_seq_len=2048,
        ignore_index=-100,
        image_size=224,
        state_vocab_size=256,
    )

    text = "Move the object to the target location"
    images = np.random.rand(2, 224, 224, 3)
    state = np.random.rand(6, 48) * 2 - 1
    human_action = np.zeros((30, 48))
    output = processor(text, images, state, human_action)
    np.set_printoptions(threshold=np.inf)
    print(output)

def test_COCO_Caption2017():
    dataset = load_from_disk("/share_data/datasets/VLM/COCO-Caption2017")
    print(dataset)

# test_PaliGemma_processor()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_type", type=str, default="PaliGemmaProcessor")
args = parser.parse_args()

if args.test_type == "gemma_tokenizer":
    test_Gemma_tokenizer()
elif args.test_type == "PaliGemmaProcessor":
    test_PaliGemma_processor()
elif args.test_type == "PaliGemmaVLAProcessor":
    test_PaliGemmaVLA_processor()
else: 
    raise ValueError(f"Invalid test type: {args.test_type}")

