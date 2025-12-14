# quantize_locally.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader
from data import HateSpeechDataset, load_and_preprocess_data
from train import evaluate
from utils import get_model_metrics
import os

# ========================= CONFIGURATION =========================
HF_MODEL_NAME = "your-username/your-best-pruned-model"  # e.g., "Nabil619/best-gradual-magnitude-pruned-bangla-hate"
DATASET_PATH = "/kaggle/working/HatePrune/data/HateSpeech.csv"  # Your CSV path
VAL_FRACTION = 0.2  # Use 20% of data as validation

# Local save directories
SAVE_FP16 = "./quantized_fp16"
SAVE_INT8 = "./quantized_int8"
SAVE_INT4 = "./quantized_int4"

os.makedirs(SAVE_FP16, exist_ok=True)
os.makedirs(SAVE_INT8, exist_ok=True)
os.makedirs(SAVE_INT4, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================= LOAD DATA =========================
print("Loading dataset...")
comments, labels = load_and_preprocess_data(DATASET_PATH)
num_val = int(len(comments) * VAL_FRACTION)
val_comments, val_labels = comments[-num_val:], labels[-num_val:]
train_comments, train_labels = comments[:-num_val:], labels[:-num_val:]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

val_dataset = HateSpeechDataset(val_comments, val_labels, tokenizer)
train_dataset = HateSpeechDataset(train_comments, train_labels, tokenizer)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# ========================= EVALUATION HELPERS =========================
def evaluate_model(model, name_prefix=""):
    model.eval()
    with torch.no_grad():
        val_metrics = evaluate(model, val_loader, device)
        train_metrics = evaluate(model, train_loader, device)
    
    combined = val_metrics.copy()
    for k, v in train_metrics.items():
        if k != 'best_threshold':
            combined[f'train_{k}'] = v
    return combined

def print_summary(metrics, size_metrics, title="MODEL"):
    print("\n" + "="*70)
    print(f"{title} PERFORMANCE & SIZE")
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

    print(f"Train Accuracy       : {metrics['train_accuracy']:.4f}")
    print(f"Train F1 (Hate)      : {metrics['train_f1']:.4f}")
    print(f"Train Macro F1       : {metrics['train_macro_f1']:.4f}")
    print(f"Train Loss           : {metrics['train_loss']:.4f}")

    print("\nModel Size & Params:")
    print(f"  Total params     : {size_metrics['total_parameters']:,}")
    print(f"  Trainable params : {size_metrics['trainable_parameters']:,}")
    print(f"  Model size (MB)  : {size_metrics['model_size_mb']:.2f} MB")
    if 'sparsity_percent' in size_metrics:
        print(f"  Sparsity         : {size_metrics['sparsity_percent']:.2f}%")
    print("="*70)

# ========================= 1. LOAD ORIGINAL (FP32) =========================
print(f"Loading original model from Hugging Face: {HF_MODEL_NAME}")
original_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
original_model.to(device)

original_metrics = evaluate_model(original_model, "Original")
original_size = get_model_metrics(original_model)
print_summary(original_metrics, original_size, "ORIGINAL PRUNED (FP32)")

# ========================= 2. FP16 QUANTIZATION =========================
print("\nQuantizing to FP16 (float16)...")
fp16_model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

fp16_metrics = evaluate_model(fp16_model, "FP16")
fp16_size = get_model_metrics(fp16_model)
print_summary(fp16_metrics, fp16_size, "QUANTIZED FP16")

# Save locally
fp16_model.save_pretrained(SAVE_FP16)
tokenizer.save_pretrained(SAVE_FP16)
print(f"FP16 model saved locally to: {SAVE_FP16}")

# ========================= 3. INT8 QUANTIZATION =========================
print("\nQuantizing to INT8...")
int8_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

int8_model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL_NAME,
    quantization_config=int8_config,
    device_map="auto"
)

int8_metrics = evaluate_model(int8_model, "INT8")
int8_size = get_model_metrics(int8_model)
print_summary(int8_metrics, int8_size, "QUANTIZED INT8")

# Save locally
int8_model.save_pretrained(SAVE_INT8)
tokenizer.save_pretrained(SAVE_INT8)
print(f"INT8 model saved locally to: {SAVE_INT8}")

# ========================= 4. INT4 QUANTIZATION =========================
print("\nQuantizing to INT4...")
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

int4_metrics = evaluate_model(int4_model, "INT4")
int4_size = get_model_metrics(int4_model)
print_summary(int4_metrics, int4_size, "QUANTIZED INT4")

# Save locally
int4_model.save_pretrained(SAVE_INT4)
tokenizer.save_pretrained(SAVE_INT4)
print(f"INT4 model saved locally to: {SAVE_INT4}")

# ========================= FINAL COMPARISON =========================
print("\n" + "="*70)
print("FINAL QUANTIZATION COMPARISON")
print("="*70)
print(f"Original → Macro F1: {original_metrics['macro_f1']:.4f} | Size: {original_size['model_size_mb']:.1f} MB")
print(f"FP16     → Macro F1: {fp16_metrics['macro_f1']:.4f} (Δ {original_metrics['macro_f1'] - fp16_metrics['macro_f1']:.4f}) | Size: {fp16_size['model_size_mb']:.1f} MB")
print(f"INT8     → Macro F1: {int8_metrics['macro_f1']:.4f} (Δ {original_metrics['macro_f1'] - int8_metrics['macro_f1']:.4f}) | Size: {int8_size['model_size_mb']:.1f} MB")
print(f"INT4     → Macro F1: {int4_metrics['macro_f1']:.4f} (Δ {original_metrics['macro_f1'] - int4_metrics['macro_f1']:.4f}) | Size: {int4_size['model_size_mb']:.1f} MB")
print("="*70)
print("All quantized models saved locally in ./quantized_fp16, ./quantized_int8, ./quantized_int4")
