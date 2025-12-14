# quantize_experiment.py - Run this in Kaggle after connecting your GitHub repo

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

# === Add your repo code to Python path ===
CODE_DIR = "/kaggle/input/hateprune"  # Kaggle mounts your repo here
import sys
sys.path.append(CODE_DIR)

# Now import your custom modules
from data import load_and_preprocess_data, HateSpeechDataset
from train import evaluate
from utils import get_model_metrics

# === CONFIGURATION ===
HF_MODEL_NAME = "your-username/your-best-model"  # CHANGE THIS! e.g., "Nabil619/best-gradual-magnitude-pruned-bangla-hate"
DATASET_PATH = "/kaggle/input/hateprune/data/HateSpeech.csv"  # Your dataset in repo

VAL_FRACTION = 0.2  # Use 20% as validation

# Local save paths (will appear in Kaggle Output)
SAVE_FP16 = "/kaggle/working/quantized_fp16"
SAVE_INT8 = "/kaggle/working/quantized_int8"
SAVE_INT4 = "/kaggle/working/quantized_int4"

os.makedirs(SAVE_FP16, exist_ok=True)
os.makedirs(SAVE_INT8, exist_ok=True)
os.makedirs(SAVE_INT4, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === LOAD DATA ===
print("Loading dataset...")
comments, labels = load_and_preprocess_data(DATASET_PATH)
num_val = int(len(comments) * VAL_FRACTION)
val_comments = comments[-num_val:]
val_labels = labels[-num_val:]
train_comments = comments[:-num_val:]  # Optional, for train metrics if needed

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

val_dataset = HateSpeechDataset(val_comments, val_labels, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# === HELPERS ===
def run_evaluation(model, name):
    model.eval()
    with torch.no_grad():
        metrics = evaluate(model, val_loader, device)
    
    size_metrics = get_model_metrics(model)
    
    print("\n" + "="*70)
    print(f"{name} PERFORMANCE & SIZE")
    print("="*70)
    print(f"Val Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Val Precision (Hate) : {metrics['precision']:.4f}")
    print(f"Val Recall (Hate)    : {metrics['recall']:.4f}")
    print(f"Val F1 (Hate)        : {metrics['f1']:.4f}")
    print(f"Val Precision (Non-Hate): {metrics['precision_negative']:.4f}")
    print(f"Val Recall (Non-Hate)   : {metrics['recall_negative']:.4f}")
    print(f"Val F1 (Non-Hate)       : {metrics['f1_negative']:.4f}")
    print(f"Val Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"Val ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"Val Loss             : {metrics['loss']:.4f}")
    print(f"Best Threshold       : {metrics['best_threshold']:.3f}")
    
    print("\nModel Size:")
    print(f"  Total params     : {size_metrics['total_parameters']:,}")
    print(f"  Trainable params : {size_metrics['trainable_parameters']:,}")
    print(f"  Model size (MB)  : {size_metrics['model_size_mb']:.2f} MB")
    if 'sparsity_percent' in size_metrics:
        print(f"  Sparsity         : {size_metrics['sparsity_percent']:.2f}%")
    print("="*70)
    
    return metrics, size_metrics

# === 1. LOAD ORIGINAL PRUNED MODEL ===
print(f"Loading original model from Hugging Face: {HF_MODEL_NAME}")
original_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
original_model.to(device)

orig_metrics, orig_size = run_evaluation(original_model, "ORIGINAL PRUNED (FP32)")

# === 2. FP16 QUANTIZATION ===
print("\nQuantizing to FP16...")
fp16_model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
fp16_metrics, fp16_size = run_evaluation(fp16_model, "QUANTIZED FP16")

fp16_model.save_pretrained(SAVE_FP16)
tokenizer.save_pretrained(SAVE_FP16)
print(f"FP16 model saved to: {SAVE_FP16}")

# === 3. INT8 QUANTIZATION ===
print("\nQuantizing to INT8...")
int8_config = BitsAndBytesConfig(load_in_8bit=True)
int8_model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL_NAME,
    quantization_config=int8_config,
    device_map="auto"
)
int8_metrics, int8_size = run_evaluation(int8_model, "QUANTIZED INT8")

int8_model.save_pretrained(SAVE_INT8)
tokenizer.save_pretrained(SAVE_INT8)
print(f"INT8 model saved to: {SAVE_INT8}")

# === 4. INT4 QUANTIZATION (NF4 - Recommended) ===
print("\nQuantizing to INT4 (NF4)...")
int4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
int4_model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL_NAME,
    quantization_config=int4_config,
    device_map="auto"
)
int4_metrics, int4_size = run_evaluation(int4_model, "QUANTIZED INT4 (NF4)")

int4_model.save_pretrained(SAVE_INT4)
tokenizer.save_pretrained(SAVE_INT4)
print(f"INT4 model saved to: {SAVE_INT4}")

# === FINAL SUMMARY ===
print("\n" + "="*70)
print("QUANTIZATION RESULTS SUMMARY")
print("="*70)
print(f"Original → Macro F1: {orig_metrics['macro_f1']:.4f} | Size: {orig_size['model_size_mb']:.1f} MB")
print(f"FP16     → Macro F1: {fp16_metrics['macro_f1']:.4f} (Δ {orig_metrics['macro_f1'] - fp16_metrics['macro_f1']:.4f}) | Size: {fp16_size['model_size_mb']:.1f} MB")
print(f"INT8     → Macro F1: {int8_metrics['macro_f1']:.4f} (Δ {orig_metrics['macro_f1'] - int8_metrics['macro_f1']:.4f}) | Size: {int8_size['model_size_mb']:.1f} MB")
print(f"INT4     → Macro F1: {int4_metrics['macro_f1']:.4f} (Δ {orig_metrics['macro_f1'] - int4_metrics['macro_f1']:.4f}) | Size: {int4_size['model_size_mb']:.1f} MB")
print("="*70)
print("All quantized models saved in /kaggle/working/")
print("Go to Output tab → Download the folders!")
