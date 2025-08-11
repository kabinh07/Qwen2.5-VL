import os
import csv
import gc
from difflib import SequenceMatcher
import multiprocessing as mp

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm

# --- Configuration ---
# Model and processor paths
FINETUNED_CKPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
BASE_CKPT_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

# Data paths
INFERENCE_DIR = os.path.join(os.path.dirname(__file__), "../data/inference")
IMAGES_DIR = os.path.join(INFERENCE_DIR, "images")
ANNOTATION_FILE = os.path.join(INFERENCE_DIR, "annotation.csv")


def load_annotations(annotation_path):
    """Loads annotations from the CSV file."""
    annotations = []
    with open(annotation_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            annotations.append(row)
    return annotations


def run_model_inference(model, processor, image_path, prompt):
    """
    Run inference on a single image.
    Assumes the processor is already fully configured by the worker.
    """
    try:
        # This utility function should be available in your environment
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]


        # Robustly set chat template on both tokenizer and backend_tokenizer
        chat_template = getattr(processor.tokenizer, "chat_template", None)
        if chat_template:
            processor.tokenizer.chat_template = chat_template
            if hasattr(processor.tokenizer, "backend_tokenizer") and hasattr(processor.tokenizer.backend_tokenizer, "chat_template"):
                processor.tokenizer.backend_tokenizer.chat_template = chat_template

        # Try to pass chat_template directly if supported
        try:
            text = processor.apply_chat_template(
                [messages], tokenize=False, add_generation_prompt=True, chat_template=chat_template
            )
        except TypeError:
            text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])

        inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None  # Suppress warning, not used in greedy decoding
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    except Exception as e:
        # This error will be caught by the worker and reported
        print(f"Error during model inference: {str(e)}")
        return f"Error: {str(e)}"


def inference_worker(device_id, image_chunk, model_path, prompt, quantization_config):
    """
    A worker function that runs on a single GPU.
    It initializes a model and processes its assigned chunk of images.
    """
    device = f'cuda:{device_id}'
    worker_results = []
    model = None
    processor = None

    try:
        print(f"Worker {device_id}: Starting and loading model '{os.path.basename(model_path)}' on {device}...")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            quantization_config=quantization_config,
            torch_dtype=torch.float16
        )
        processor = AutoProcessor.from_pretrained(model_path)

        # **THE FIX IS HERE**: For the fine-tuned model, we must ensure the chat template
        # is correctly set on both the tokenizer and its backend.
        if "output" in model_path:
            print(f"Worker {device_id}: Applying base model chat template to fine-tuned processor.")
            base_processor = AutoProcessor.from_pretrained(BASE_CKPT_PATH)
            chat_template = getattr(base_processor.tokenizer, "chat_template", None)
            if chat_template:
                # Set it on the main tokenizer...
                processor.tokenizer.chat_template = chat_template
                # ...AND the backend_tokenizer if it exists. This is the crucial part.
                if hasattr(processor.tokenizer, "backend_tokenizer") and hasattr(processor.tokenizer.backend_tokenizer, "chat_template"):
                    processor.tokenizer.backend_tokenizer.chat_template = chat_template
            del base_processor

        for img_data in tqdm(image_chunk, desc=f"Worker {device_id}", unit="image", position=device_id):
            image_name = img_data['image_name']
            image_path = img_data['image_path']
            prediction = run_model_inference(model, processor, image_path, prompt)
            
            img_data_with_pred = img_data.copy()
            img_data_with_pred['prediction'] = prediction
            worker_results.append(img_data_with_pred)

    except Exception as e:
        print(f"FATAL ERROR in Worker {device_id}: {e}")
    finally:
        # Clean up to free GPU memory before the process exits
        del model
        del processor
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Worker {device_id}: Finished and cleaned up.")

    return worker_results


def main():
    """Main orchestration function."""
    # --- 1. Initial Setup ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    if num_gpus == 0:
        print("No GPUs found. Exiting.")
        return
        
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    annotations = load_annotations(ANNOTATION_FILE)
    prompt = "Please extract and return the complete address from this image. Only return the address text, no additional commentary."

    # --- 2. Prepare Image List ---
    valid_images = []
    for ann in annotations:
        image_name = ann.get('upload_id')
        if not image_name:
            continue
            
        image_path = os.path.join(IMAGES_DIR, image_name)
        if not os.path.exists(image_path):
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                test_path = image_path + ext if not image_name.lower().endswith(ext) else image_path
                if os.path.exists(test_path):
                    image_path = test_path
                    found = True
                    break
            if not found:
                continue
        
        valid_images.append({
            'annotation': ann,
            'image_name': image_name,
            'image_path': image_path
        })
    

    # Limit to 3 images for testing
    # valid_images = valid_images[:3]
    # if not valid_images:
    #     print("No valid images found to process.")
    #     return
    # print(f"\nFound {len(valid_images)} valid images to process across {num_gpus} GPUs (limited to 3 for testing).")

    # --- 3. Split Data for Parallel Processing ---
    image_chunks = [[] for _ in range(num_gpus)]
    for i, img_data in enumerate(valid_images):
        image_chunks[i % num_gpus].append(img_data)

    # --- 4. Run Base Model Inference in Parallel ---
    print("\n=== PHASE 1: Base Model Inference (Parallel) ===")
    base_model_args = [
        (i, image_chunks[i], BASE_CKPT_PATH, prompt, quantization_config) for i in range(num_gpus)
    ]
    with mp.Pool(processes=num_gpus) as pool:
        base_results_list = pool.starmap(inference_worker, base_model_args)
    
    base_predictions = {
        item['image_name']: item['prediction']
        for chunk in base_results_list for item in chunk
    }
    print("Base model inference completed.")

    # --- 5. Run Fine-tuned Model Inference in Parallel ---
    print("\n=== PHASE 2: Fine-tuned Model Inference (Parallel) ===")
    finetuned_model_args = [
        (i, image_chunks[i], FINETUNED_CKPT_PATH, prompt, quantization_config) for i in range(num_gpus)
    ]
    with mp.Pool(processes=num_gpus) as pool:
        finetuned_results_list = pool.starmap(inference_worker, finetuned_model_args)

    all_finetuned_results = [item for sublist in finetuned_results_list for item in sublist]
    print("Fine-tuned model inference completed.")

    # --- 6. Combine Results and Calculate Scores ---
    print("\n=== Combining results and calculating scores... ===")
    results = []
    finetuned_scores = []
    base_scores = []

    def string_similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    for item in all_finetuned_results:
        image_name = item['image_name']
        ann = item['annotation']
        gt_address = ann.get('address', '')
        pred_address_finetuned = item['prediction']
        pred_address_base = base_predictions.get(image_name, "Error: Base prediction not found")

        score_finetuned = string_similarity(gt_address, pred_address_finetuned)
        score_base = string_similarity(gt_address, pred_address_base)
        finetuned_scores.append(score_finetuned)
        base_scores.append(score_base)

        # Surya prediction and accuracy from annotation if present
        surya_prediction = ann.get('address_ocr_v6', '')
        surya_accuracy = ann.get('address_ocr_v6_accuracy', '')

        results.append({
            "image": image_name,
            "ground_truth": gt_address,
            "finetuned_prediction": pred_address_finetuned,
            "finetuned_score": f"{score_finetuned:.3f}",
            "base_prediction": pred_address_base,
            "base_score": f"{score_base:.3f}",
            "surya_prediction": surya_prediction,
            "surya_accuracy": surya_accuracy
        })

    avg_score_finetuned = sum(finetuned_scores) / len(finetuned_scores) if finetuned_scores else 0.0
    avg_score_base = sum(base_scores) / len(base_scores) if base_scores else 0.0
    print(f"\nAverage address similarity score (base): {avg_score_base:.3f}")
    print(f"Average address similarity score (finetuned): {avg_score_finetuned:.3f}")

    # --- 7. Save Results ---
    results_csv_path = os.path.join(INFERENCE_DIR, "inference_results.csv")
    with open(results_csv_path, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "image", "ground_truth", "finetuned_prediction", "finetuned_score", "base_prediction", "base_score",
            "surya_prediction", "surya_accuracy"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Inference results saved to {results_csv_path}")

    score_txt_path = os.path.join(INFERENCE_DIR, "inference_score.txt")
    with open(score_txt_path, "w", encoding='utf-8') as f:
        f.write(f"Average address similarity score (base): {avg_score_base:.3f}\n")
        f.write(f"Average address similarity score (finetuned): {avg_score_finetuned:.3f}\n")
    print(f"Average scores saved to {score_txt_path}")


if __name__ == "__main__":
    # It's crucial to set the start method for CUDA multiprocessing to 'spawn'
    # This ensures that child processes start with a clean slate.
    try:
        mp.set_start_method("spawn", force=True)
        main()
    except RuntimeError as e:
        print(f"RuntimeError: {e}. This can happen if the start method is already set. Ensure this script is run directly.")