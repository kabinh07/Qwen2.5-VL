import os
import csv

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
import gc
from tqdm import tqdm

from difflib import SequenceMatcher

# Model and processor paths
FINETUNED_CKPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
BASE_CKPT_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

INFERENCE_DIR = os.path.join(os.path.dirname(__file__), "../data/inference")
IMAGES_DIR = os.path.join(INFERENCE_DIR, "images")
ANNOTATION_FILE = os.path.join(INFERENCE_DIR, "annotation.csv")


def load_annotations(annotation_path):
    annotations = []
    with open(annotation_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            annotations.append(row)
    return annotations



def run_model_inference(model, processor, image_path, prompt):
    """Run inference on a single image using the evaluation framework pattern"""
    try:
        from qwen_vl_utils import process_vision_info


        # Always ensure chat template is set on processor's tokenizer before apply_chat_template
        base_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        chat_template = getattr(base_processor.tokenizer, "chat_template", None)
        if chat_template:
            processor.tokenizer.chat_template = chat_template
            # Also set on backend_tokenizer if it exists
            if hasattr(processor.tokenizer, "backend_tokenizer") and hasattr(processor.tokenizer.backend_tokenizer, "chat_template"):
                processor.tokenizer.backend_tokenizer.chat_template = chat_template
        print(f"[DEBUG] processor type: {type(processor)}")
        print(f"[DEBUG] processor.tokenizer type: {type(processor.tokenizer)}")
        print(f"[DEBUG] chat_template present: {getattr(processor.tokenizer, 'chat_template', None) is not None}")
        if hasattr(processor.tokenizer, "backend_tokenizer"):
            print(f"[DEBUG] backend_tokenizer type: {type(processor.tokenizer.backend_tokenizer)}")
            print(f"[DEBUG] backend_tokenizer chat_template present: {getattr(processor.tokenizer.backend_tokenizer, 'chat_template', None) is not None}")

        # Create message structure following evaluation code pattern
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template and process vision info like in evaluation code
        # Pass chat_template directly if supported
        try:
            text = processor.apply_chat_template(
                [messages], tokenize=False, add_generation_prompt=True, chat_template=chat_template
            )
        except TypeError:
            # Fallback for older versions that don't support chat_template arg
            text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])

        # DEBUG: Show what is being sent to the model
        # print("\n[DEBUG] Example input to model:")
        # print(f"[DEBUG] Image path: {image_path}")
        # print(f"[DEBUG] Prompt text (after chat template):\n{text}\n")

        # Process inputs
        inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to(model.device)

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None  # Suppress warning, not used in greedy decoding
            )

        # Decode response following evaluation pattern
        generated_ids_trimmed = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    except Exception as e:
        print(f"Error in inference: {str(e)}")
        return f"Error: {str(e)}"

def main():
    # Use quantization to reduce memory usage
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    annotations = load_annotations(ANNOTATION_FILE)
    results = []
    
    def string_similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # Prepare the prompt
    prompt = "Please extract and return the complete address from this image. Only return the address text, no additional commentary."
    
    # Process images and collect valid image paths
    valid_images = []
    for ann in annotations:
        # Try different possible column names for the image filename
        image_name = None
        for possible_key in ['upload_id']:
            if possible_key in ann and ann[possible_key]:
                image_name = ann[possible_key]
                break
        
        if image_name is None:
            print(f"No image filename found in row: {ann}")
            print(f"Available columns: {list(ann.keys())}")
            continue
            
        # Try to find the image file with or without extension
        image_path = os.path.join(IMAGES_DIR, image_name)
        if not os.path.exists(image_path):
            # Try common extensions
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                test_path = image_path + ext if not image_name.lower().endswith(ext) else image_path
                if os.path.exists(test_path):
                    image_path = test_path
                    found = True
                    break
            if not found:
                print(f"Image not found: {image_path} (tried common extensions)")
                continue
        
        valid_images.append({
            'annotation': ann,
            'image_name': image_name,
            'image_path': image_path
        })
    
    # For testing, limit to first 5 images
    # valid_images = valid_images[:2]
    
    # print(f"Processing {len(valid_images)} valid images (limited to first {len(valid_images)} for testing)...")
    
    # PHASE 1: Run base model inference across all GPUs
    print("\n=== PHASE 1: Base Model Inference ===")
    print("Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_CKPT_PATH, 
        device_map='auto',
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )
    base_processor = AutoProcessor.from_pretrained(BASE_CKPT_PATH)
    
    # Check GPU memory usage after loading base model
    for i in range(num_gpus):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
        print(f"GPU {i} - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    base_predictions = {}
    for img_data in tqdm(valid_images, desc="Base model inference", unit="image"):
        image_name = img_data['image_name']
        image_path = img_data['image_path']
        
        # tqdm.write(f"Processing {image_name} - Running base model...")
        pred_address_base = run_model_inference(base_model, base_processor, image_path, prompt)
        base_predictions[image_name] = pred_address_base
        
        # Clear cache to free up memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Clean up base model
    del base_model
    del base_processor
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Base model inference completed. Cleaning up...")
    
    # PHASE 2: Run fine-tuned model inference across all GPUs
    print("\n=== PHASE 2: Fine-tuned Model Inference ===")
    print("Loading fine-tuned model...")
    finetuned_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        FINETUNED_CKPT_PATH, 
        device_map='auto',
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )
    finetuned_processor = AutoProcessor.from_pretrained(FINETUNED_CKPT_PATH)

    # Always set the chat template from the base processor after loading
    print("Checking chat template...")
    temp_base_processor = AutoProcessor.from_pretrained(BASE_CKPT_PATH)
    print(f"Base model chat template exists: {temp_base_processor.tokenizer.chat_template is not None}")
    print(f"Fine-tuned model chat template exists: {finetuned_processor.tokenizer.chat_template is not None}")
    print("Copying chat template from base model to fine-tuned model...")
    finetuned_processor.tokenizer.chat_template = temp_base_processor.tokenizer.chat_template
    print("Chat template copied successfully!")
    print(f"Fine-tuned model chat template after copy: {finetuned_processor.tokenizer.chat_template is not None}")
    del temp_base_processor
    
    # Check GPU memory usage after loading fine-tuned model
    for i in range(num_gpus):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
        print(f"GPU {i} - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    finetuned_scores = []
    base_scores = []
    
    for img_data in tqdm(valid_images, desc="Fine-tuned model inference", unit="image"):
        image_name = img_data['image_name']
        image_path = img_data['image_path']
        ann = img_data['annotation']
        
        # tqdm.write(f"Processing {image_name} - Running fine-tuned model...")
        pred_address_finetuned = run_model_inference(finetuned_model, finetuned_processor, image_path, prompt)
        
        # Clear cache to free up memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get base model prediction for this image
        pred_address_base = base_predictions[image_name]
        
        gt_address = ann.get('address', '')
        score_finetuned = string_similarity(gt_address, pred_address_finetuned)
        score_base = string_similarity(gt_address, pred_address_base)
        finetuned_scores.append(score_finetuned)
        base_scores.append(score_base)

        results.append({
            "image": image_name,
            "ground_truth": gt_address,
            "finetuned_prediction": pred_address_finetuned,
            "finetuned_score": score_finetuned,
            "base_prediction": pred_address_base,
            "base_score": score_base
        })
        tqdm.write(f"[Finetuned] {image_name}: pred='{pred_address_finetuned}', gt='{gt_address}', score={score_finetuned:.3f}")
        tqdm.write(f"[Base]      {image_name}: pred='{pred_address_base}', gt='{gt_address}', score={score_base:.3f}")

    avg_score_finetuned = sum(finetuned_scores) / len(finetuned_scores) if finetuned_scores else 0.0
    avg_score_base = sum(base_scores) / len(base_scores) if base_scores else 0.0
    print(f"\nAverage address similarity score (base): {avg_score_base:.3f}")
    print(f"Average address similarity score (finetuned): {avg_score_finetuned:.3f}")

    # Save results to a CSV
    with open(os.path.join(INFERENCE_DIR, "inference_results.csv"), "w", newline='') as csvfile:
        fieldnames = ["image", "ground_truth", "finetuned_prediction", "finetuned_score", "base_prediction", "base_score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    # Save average scores
    with open(os.path.join(INFERENCE_DIR, "inference_score.txt"), "w") as f:
        f.write(f"Average address similarity score (base): {avg_score_base:.3f}\n")
        f.write(f"Average address similarity score (finetuned): {avg_score_finetuned:.3f}\n")


if __name__ == "__main__":
    main()
