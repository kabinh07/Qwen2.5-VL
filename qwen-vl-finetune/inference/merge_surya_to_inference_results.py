import csv
import os

# Paths (edit as needed)
INFERENCE_RESULTS_PATH = "data/inference/inference_results.csv"
NID_INFOS_PATH = "data/inference/nid_infos_compared_31-07-25_12-14.csv"
OUTPUT_PATH = "data/inference/inference_results_with_surya.csv"

# Load Surya info from nid_infos_compared_31-07-25_12-14.csv into a dict by image name (or other key)
def load_surya_info(path):
    surya_dict = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use upload_id or image or another unique key as needed
            key = row.get('upload_id') or row.get('image')
            if key:
                surya_dict[key] = {
                    'surya_prediction': row.get('address_ocr_v6', ''),
                    'surya_accuracy': row.get('address_ocr_v6_accuracy', '')
                }
    return surya_dict

# Read inference_results.csv, add surya columns, and write new file
def merge_results(inf_path, surya_path, out_path):
    surya_info = load_surya_info(surya_path)
    with open(inf_path, newline='', encoding='utf-8') as f_in, \
         open(out_path, 'w', newline='', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ['surya_prediction', 'surya_accuracy']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            key = row.get('image') or row.get('upload_id')
            surya = surya_info.get(key, {'surya_prediction': '', 'surya_accuracy': ''})
            row['surya_prediction'] = surya['surya_prediction']
            row['surya_accuracy'] = surya['surya_accuracy']
            writer.writerow(row)
    print(f"Merged results written to {out_path}")

if __name__ == "__main__":
    merge_results(INFERENCE_RESULTS_PATH, NID_INFOS_PATH, OUTPUT_PATH)
