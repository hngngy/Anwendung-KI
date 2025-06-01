!pip install transformers==4.38.2 --user
!pip install scikit-learn --user

# Restart kernel after installation
print("After installation completes, please restart your kernel and run the next cell!")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification, AutoImageProcessor
import requests
import zipfile
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists, skipping download.")
        return
    print(f"⬇️ Downloading {dest_path}...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"Downloaded {dest_path}")

# Download ISIC Test Images (this will take a while - ~3.6GB)
print("Downloading ISIC 2019 Test Images...")
url_zip = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Input.zip"
zip_path = "ISIC_2019_Test_Input.zip"
extract_dir = "ISIC_test_images"

download_file(url_zip, zip_path)

# Extract images
if not os.path.exists(extract_dir):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Test images extracted to '{extract_dir}'")
else:
    print(f"{extract_dir} already exists")

# Download Ground Truth CSV
csv_path = "ISIC_2019_Test_GroundTruth.csv"
if not os.path.exists(csv_path):
    print("Downloading ground truth CSV...")
    url_csv = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_GroundTruth.csv"
    download_file(url_csv, csv_path)

# Extract model ZIP files
zip_files = [
    'efficientnet-b2-isic.zip',
    'efficientnet-b3-isic.zip',
    'efficientnet-b4-isic.zip'
]

MODEL_DIRS = []
for zip_file in zip_files:
    if os.path.exists(zip_file):
        extract_dir_model = zip_file.replace('.zip', '')
        MODEL_DIRS.append(extract_dir_model)

        if not os.path.exists(extract_dir_model):
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir_model)
            print(f"Extracted to {extract_dir_model}")
        else:
            print(f"{extract_dir_model} already exists")

# Set up paths
IMG_DIR = "ISIC_test_images/ISIC_2019_Test_Input"
CSV_PATH = csv_path
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Images: {IMG_DIR}")
print(f"  Ground truth: {CSV_PATH}")
print(f"  Model directories: {MODEL_DIRS}")
print(f"  Results: {RESULTS_DIR}")

# Load and clean ground truth data
ground_truth = pd.read_csv(CSV_PATH)
metadata_columns = ['score_weight', 'validation_weight']
class_columns = [col for col in ground_truth.columns if col not in ['image'] + metadata_columns]

print("Class columns:", class_columns)
print("Number of classes:", len(class_columns))

# Create clean ground truth with only class columns
ground_truth_clean = ground_truth[['image'] + class_columns].copy()

# Convert one-hot encoded labels to single label indices
def get_label_from_row(row):
    label_values = row.drop('image').astype(int).values
    return label_values.argmax()

ground_truth_clean['label'] = ground_truth_clean.apply(get_label_from_row, axis=1)

# Create final dataframe with image and label columns
test_df = ground_truth_clean[['image', 'label']].copy()
test_df['image'] = test_df['image'].apply(lambda x: f"{x}.jpg" if not x.endswith('.jpg') else x)

print("Test dataframe shape:", test_df.shape)
print("Label distribution:")
print(test_df['label'].value_counts().sort_index())

# Show class mapping
print("\nClass mapping:")
for i, class_name in enumerate(class_columns):
    count = (test_df['label'] == i).sum()
    print(f"  {i}: {class_name} ({count} samples)")

# Load models and processors
models = {}
processors = {}
base_model_names = ["google/efficientnet-b2", "google/efficientnet-b3", "google/efficientnet-b4"]
model_names = ['B2', 'B3', 'B4']

for model_dir, base_model_name, model_name in zip(MODEL_DIRS, base_model_names, model_names):
    print(f"Loading {model_name}...")

    # Load processor from base model
    processor = AutoImageProcessor.from_pretrained(base_model_name)
    processors[model_name] = processor

    # Load fine-tuned model from local directory
    model = AutoModelForImageClassification.from_pretrained(
        model_dir,
        local_files_only=True
    ).to(device)
    model.eval()
    models[model_name] = model

    print(f" Loaded {model_name} from {model_dir}")
    print(f" Number of classes: {model.config.num_labels}")

print(f"\n Loaded {len(models)} models: {list(models.keys())}")

# IMPORTANT: Set test subset size (change to test_df for full dataset)
test_subset = test_df  # Change this to test_df for full dataset
print(f"Processing {len(test_subset)} samples...")

# Collect predictions with proper alignment
all_ensemble_probs = []
all_labels = []

print("Collecting predictions...")

with torch.no_grad():
    for i, row in enumerate(test_subset.itertuples()):
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(test_subset)}")

        # Get the image
        img_path = os.path.join(IMG_DIR, row.image)
        if not os.path.exists(img_path):
            img_path_no_ext = os.path.join(IMG_DIR, row.image.replace('.jpg', ''))
            if os.path.exists(img_path_no_ext):
                img_path = img_path_no_ext
            else:
                continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {row.image}: {e}")
            continue

        # Collect predictions from all models
        model_probs = []
        for model_name, model in models.items():
            inputs = processors[model_name](images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            model_probs.append(probs)

        # Average the probabilities (NO renormalization!)
        ensemble_prob = np.mean(model_probs, axis=0)

        # Add to collections
        all_ensemble_probs.append(ensemble_prob)
        all_labels.append(row.label)

# Convert to numpy arrays
all_ensemble_probs = np.array(all_ensemble_probs)
all_labels = np.array(all_labels)

print(f"\n Final shapes - Predictions: {all_ensemble_probs.shape}, Labels: {all_labels.shape}")
print(f" Processed {len(all_labels)} samples successfully")

print("=== FIXING CLASS MAPPING ===")

# CRITICAL: Correct mapping from training code
correct_mapping = {
    0: 'AK',   # LABEL_0 -> AK (Actinic keratosis)
    1: 'BCC',  # LABEL_1 -> BCC (Basal cell carcinoma)
    2: 'BKL',  # LABEL_2 -> BKL (Benign keratosis-like lesions)
    3: 'DF',   # LABEL_3 -> DF (Dermatofibroma)
    4: 'MEL',  # LABEL_4 -> MEL (Melanoma)
    5: 'NV',   # LABEL_5 -> NV (Melanocytic nevus)
    6: 'SCC',  # LABEL_6 -> SCC (Squamous cell carcinoma)
    7: 'UNK',  # LABEL_7 -> UNK (Unknown)
    8: 'VASC'  # LABEL_8 -> VASC (Vascular lesions)
}

# Test set class order (from CSV columns)
test_class_order = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']

print("Model was trained with:", [correct_mapping[i] for i in range(9)])
print("Test set order:", test_class_order)

# Create mapping from model output indices to test set indices
model_to_test_mapping = {}
for model_idx, class_name in correct_mapping.items():
    test_idx = test_class_order.index(class_name)
    model_to_test_mapping[model_idx] = test_idx

print("Model output -> Test set mapping:", model_to_test_mapping)

# Re-map the ensemble predictions to correct class order
corrected_ensemble_probs = np.zeros_like(all_ensemble_probs)
for model_idx, test_idx in model_to_test_mapping.items():
    corrected_ensemble_probs[:, test_idx] = all_ensemble_probs[:, model_idx]

print(f"Corrected ensemble shape: {corrected_ensemble_probs.shape}")

# Check which classes are present in our test subset
present_classes = sorted(np.unique(all_labels))
print(f"Present classes: {[test_class_order[i] for i in present_classes]}")

# Filter for present classes and normalize
corrected_ensemble_filtered = corrected_ensemble_probs[:, present_classes]
corrected_ensemble_normalized = corrected_ensemble_filtered / corrected_ensemble_filtered.sum(axis=1, keepdims=True)

# Create mapped labels for sklearn
label_mapping = {original: new for new, original in enumerate(present_classes)}
mapped_labels = np.array([label_mapping[label] for label in all_labels])

print(f"Final ensemble shape: {corrected_ensemble_normalized.shape}")
print(f"Mapped labels shape: {mapped_labels.shape}")

print("=== CALCULATING CORRECTED AUCs ===")

# Calculate corrected ensemble AUC
corrected_ensemble_auc = roc_auc_score(mapped_labels, corrected_ensemble_normalized,
                                       average='macro', multi_class='ovr')

print(f"CORRECTED Ensemble Macro-AUC: {corrected_ensemble_auc:.4f}")

# Calculate individual model AUCs with correct mapping
corrected_individual_aucs = {}

with torch.no_grad():
    for model_name, model in models.items():
        print(f"\nProcessing {model_name} with correct mapping...")
        model_probs = []

        for i, row in enumerate(test_subset.itertuples()):
            if i >= len(all_labels):
                break

            img_path = os.path.join(IMG_DIR, row.image)
            if not os.path.exists(img_path):
                img_path = os.path.join(IMG_DIR, row.image.replace('.jpg', ''))
                if not os.path.exists(img_path):
                    continue

            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processors[model_name](images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

                # Reorder probabilities according to correct mapping
                corrected_probs = np.zeros(9)
                for model_idx, test_idx in model_to_test_mapping.items():
                    corrected_probs[test_idx] = probs[model_idx]

                model_probs.append(corrected_probs)

            except Exception as e:
                continue

        if len(model_probs) == len(all_labels):
            model_probs = np.array(model_probs)

            # Filter for present classes and normalize
            model_probs_filtered = model_probs[:, present_classes]
            model_probs_normalized = model_probs_filtered / model_probs_filtered.sum(axis=1, keepdims=True)

            # Calculate AUC with mapped labels
            model_auc = roc_auc_score(mapped_labels, model_probs_normalized,
                                      average='macro', multi_class='ovr')
            corrected_individual_aucs[model_name] = model_auc
            print(f"{model_name} CORRECTED AUC: {model_auc:.4f}")
        else:
            corrected_individual_aucs[model_name] = 0.0

print(f"\nCORRECTED Individual Model AUCs:")
for name, auc_score in corrected_individual_aucs.items():
    print(f"  {name}: {auc_score:.4f}")

print("=== FINAL RESULTS ===")

# Create results dataframe
corrected_results_data = []
for name, auc_score in corrected_individual_aucs.items():
    corrected_results_data.append({'model': name, 'test_auc': auc_score})
corrected_results_data.append({'model': 'ensemble', 'test_auc': corrected_ensemble_auc})

corrected_results_df = pd.DataFrame(corrected_results_data)
print("CORRECTED Results:")
print(corrected_results_df)

# Show ensemble performance vs individual models
valid_aucs = [auc for auc in corrected_individual_aucs.values() if auc > 0]
if len(valid_aucs) > 0:
    best_individual = max(valid_aucs)
    improvement = corrected_ensemble_auc - best_individual
    print(f"\nEnsemble Performance:")
    print(f"  Best individual model: {best_individual:.4f}")
    print(f"  Ensemble AUC: {corrected_ensemble_auc:.4f}")
    print(f"  Improvement: {improvement:+.4f}")

# Save results
corrected_results_df.to_csv(os.path.join(RESULTS_DIR, "ensemble_results_final.csv"), index=False)

# Show prediction distribution
print(f"\nPrediction distribution:")
predicted_classes_mapped = np.argmax(corrected_ensemble_normalized, axis=1)
for i, class_idx in enumerate(present_classes):
    count = np.sum(predicted_classes_mapped == i)
    true_count = np.sum(mapped_labels == i)
    print(f"  {test_class_order[class_idx]}: Predicted={count}, True={true_count}")

print(f"\n Results saved to {RESULTS_DIR}/ensemble_results_final.csv")

# Plot ROC curves
if len(valid_aucs) > 0:
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(present_classes)))

    for i, class_idx in enumerate(present_classes):
        # Create binary labels for this class vs all others
        y_true_binary = (mapped_labels == i).astype(int)
        y_scores = corrected_ensemble_normalized[:, i]

        fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[i], lw=2.5,
                 label=f"{test_class_order[class_idx]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Ensemble ROC Curves - Macro AUC: {corrected_ensemble_auc:.4f}',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves_final.png"), dpi=300, bbox_inches='tight')
    plt.show()

    print(f" ROC curves saved to {RESULTS_DIR}/roc_curves_final.png")

print("\n ENSEMBLE EVALUATION COMPLETE!")
print(f"Final Ensemble AUC: {corrected_ensemble_auc:.4f}")

def create_training_style_auc_plot(individual_aucs, ensemble_auc, save_path=None):
    """Create AUC plot similar to training visualization"""

    # Prepare data
    model_names = list(individual_aucs.keys()) + ['Ensemble']
    auc_scores = list(individual_aucs.values()) + [ensemble_auc]

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Color scheme (similar to training plots)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    # Create bar plot
    bars = plt.bar(model_names, auc_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Customize the plot
    plt.title('Model Performance Comparison\nMacro-Average AUC on ISIC 2019 Test Set',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('AUC Score', fontsize=14, fontweight='bold')
    plt.xlabel('Models', fontsize=14, fontweight='bold')

    # Set y-axis limits for better visualization
    plt.ylim(0.75, 0.95)  # Adjust based on your AUC range

    # Add value labels on top of bars
    for bar, auc in zip(bars, auc_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{auc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Add horizontal grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add a horizontal line at 0.5 (random baseline)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline (0.5)')

    # Highlight the ensemble bar
    bars[-1].set_color('#d62728')  # Red for ensemble
    bars[-1].set_alpha(1.0)

    # Add legend
    plt.legend(loc='lower right', fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f" AUC comparison plot saved to {save_path}")

    plt.show()

    # Print summary statistics
    print(f"\n Performance Summary:")
    print(f"{'Model':<12} {'AUC':<8} {'Rank':<6}")
    print("-" * 26)

    # Sort models by performance
    all_results = list(zip(model_names, auc_scores))
    all_results.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, auc) in enumerate(all_results, 1):
        print(f"{name:<12} {auc:<8.4f} #{rank}")

# Create the plot
create_training_style_auc_plot(
    corrected_individual_aucs,
    corrected_ensemble_auc,
    save_path=os.path.join(RESULTS_DIR, "auc_comparison.png")
)

def create_ensemble_analysis_plot(corrected_individual_aucs, corrected_ensemble_auc):
    """Create detailed analysis of ensemble performance"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. AUC Comparison Bar Chart
    models = list(corrected_individual_aucs.keys()) + ['Ensemble']
    aucs = list(corrected_individual_aucs.values()) + [corrected_ensemble_auc]
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']

    bars = ax1.bar(models, aucs, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title('AUC Comparison', fontweight='bold', fontsize=14)
    ax1.set_ylabel('AUC Score')
    ax1.set_ylim(0.8, 0.92)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                 f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Ensemble vs Best Individual
    best_individual = max(corrected_individual_aucs.values())
    improvement = corrected_ensemble_auc - best_individual

    ax2.bar(['Best Individual', 'Ensemble'], [best_individual, corrected_ensemble_auc],
            color=['skyblue', 'orange'], alpha=0.8, edgecolor='black')
    ax2.set_title('Ensemble vs Best Individual Model', fontweight='bold', fontsize=14)
    ax2.set_ylabel('AUC Score')
    ax2.set_ylim(0.88, 0.91)

    # Add improvement annotation
    ax2.annotate(f'Δ = {improvement:+.4f}',
                 xy=(1, corrected_ensemble_auc), xytext=(0.5, corrected_ensemble_auc + 0.005),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=12, fontweight='bold', ha='center')

    # 3. Model Weights in Ensemble
    model_names = list(corrected_individual_aucs.keys())
    weights = [1/len(model_names)] * len(model_names)  # Equal weights

    wedges, texts, autotexts = ax3.pie(weights, labels=model_names, autopct='%1.1f%%',
                                       colors=['lightblue', 'lightgreen', 'lightyellow'])
    ax3.set_title('Ensemble Composition\n(Equal Weights)', fontweight='bold', fontsize=14)

    # 4. Performance vs Complexity
    model_complexities = {'B2': 2, 'B3': 3, 'B4': 4}  # Relative complexity
    individual_aucs_list = [corrected_individual_aucs[model] for model in model_names]
    complexities = [model_complexities[model] for model in model_names]

    ax4.scatter(complexities, individual_aucs_list, s=100, alpha=0.7, color='blue')
    ax4.scatter([3.0], [corrected_ensemble_auc], s=150, alpha=0.9, color='red',
                marker='*', label='Ensemble')

    for i, model in enumerate(model_names):
        ax4.annotate(model, (complexities[i], individual_aucs_list[i]),
                     xytext=(5, 5), textcoords='offset points')

    ax4.set_title('Performance vs Model Complexity', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Relative Model Complexity')
    ax4.set_ylabel('AUC Score')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ensemble_analysis.png"), dpi=300, bbox_inches='tight')
    plt.show()

    print(f" Ensemble analysis saved to {RESULTS_DIR}/ensemble_analysis.png")

# Create the detailed analysis
create_ensemble_analysis_plot(corrected_individual_aucs, corrected_ensemble_auc)

def create_per_class_analysis():
    """Analyze ensemble performance per class"""

    # Calculate per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    # Get predictions
    ensemble_predictions = np.argmax(corrected_ensemble_normalized, axis=1)

    # Map back to original class indices for interpretation
    original_predictions = [present_classes[pred] for pred in ensemble_predictions]
    original_labels = all_labels.copy()

    # Create classification report
    class_names = [test_class_order[i] for i in present_classes]
    report = classification_report(mapped_labels, ensemble_predictions,
                                   target_names=class_names, output_dict=True)

    # Extract per-class AUC scores
    per_class_aucs = {}
    for i, class_idx in enumerate(present_classes):
        y_true_binary = (mapped_labels == i).astype(int)
        y_scores = corrected_ensemble_normalized[:, i]

        if len(np.unique(y_true_binary)) > 1:  # Only if class is present
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            per_class_aucs[test_class_order[class_idx]] = auc(fpr, tpr)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Per-class AUC scores
    classes = list(per_class_aucs.keys())
    class_aucs = list(per_class_aucs.values())

    bars = ax1.bar(range(len(classes)), class_aucs, color='lightblue', alpha=0.8, edgecolor='black')
    ax1.set_title('Per-Class AUC Scores', fontweight='bold', fontsize=14)
    ax1.set_ylabel('AUC Score')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, auc_val in zip(bars, class_aucs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Confusion Matrix
    cm = confusion_matrix(mapped_labels, ensemble_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_class_analysis.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Print detailed report
    print(" Per-Class Performance Report:")
    print("=" * 50)
    for class_name in class_names:
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            auc_score = per_class_aucs.get(class_name, 0)

            print(f"{class_name:>6}: AUC={auc_score:.3f}, Precision={precision:.3f}, "
                  f"Recall={recall:.3f}, F1={f1:.3f}, Support={support}")

    print(f"\nOverall Accuracy: {report['accuracy']:.3f}")
    print(f"Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
    print(f" Per-class analysis saved to {RESULTS_DIR}/per_class_analysis.png")

# Create per-class analysis
create_per_class_analysis()

print("=== VERIFYING REAL ENSEMBLE vs AUC AVERAGING ===")

# Method 1: Real Ensemble (what we actually do)
print("Method 1: REAL ENSEMBLE")
print(f"  B2 individual AUC: {corrected_individual_aucs['B2']:.4f}")
print(f"  B3 individual AUC: {corrected_individual_aucs['B3']:.4f}")
print(f"  B4 individual AUC: {corrected_individual_aucs['B4']:.4f}")
print(f"  REAL ensemble AUC: {corrected_ensemble_auc:.4f}")

# Method 2: Simple AUC averaging (what your teammate is worried about)
fake_ensemble_auc = np.mean(list(corrected_individual_aucs.values()))
print(f"\nMethod 2: FAKE ENSEMBLE (just averaging AUCs)")
print(f"  Average of individual AUCs: {fake_ensemble_auc:.4f}")

# Show the difference
difference = corrected_ensemble_auc - fake_ensemble_auc
print(f"\nDifference: {difference:.4f}")

if abs(difference) > 0.001:
    print(" CONFIRMED: We have a REAL ensemble! The AUCs are different.")
    print("   This proves we're averaging predictions, not just AUCs.")
else:
    print("  The values are very close - let's investigate further...")

# Method 3: Show step-by-step ensemble process for first few samples
print(f"\n=== STEP-BY-STEP ENSEMBLE PROCESS (first 5 samples) ===")

# Let's manually recreate the ensemble for first 5 samples to show the process
sample_indices = range(min(5, len(test_subset)))

for i in sample_indices:
    row = test_subset.iloc[i]
    img_path = os.path.join(IMG_DIR, row['image'])

    if not os.path.exists(img_path):
        img_path = os.path.join(IMG_DIR, row['image'].replace('.jpg', ''))

    if os.path.exists(img_path):
        try:
            image = Image.open(img_path).convert("RGB")
            true_class = row['label']

            print(f"\nSample {i}: {row['image']}")
            print(f"  True class: {test_class_order[true_class]}")

            # Get individual model predictions
            individual_preds = {}
            for model_name, model in models.items():
                inputs = processors[model_name](images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

                # Apply correct class mapping
                corrected_probs = np.zeros(9)
                for model_idx, test_idx in model_to_test_mapping.items():
                    corrected_probs[test_idx] = probs[model_idx]

                individual_preds[model_name] = corrected_probs
                pred_class_idx = np.argmax(corrected_probs)
                pred_class_name = test_class_order[pred_class_idx]
                max_prob = corrected_probs[pred_class_idx]

                print(f"  {model_name} predicts: {pred_class_name} ({max_prob:.3f})")

            # Calculate ensemble prediction
            ensemble_pred = np.mean(list(individual_preds.values()), axis=0)
            ensemble_class_idx = np.argmax(ensemble_pred)
            ensemble_class_name = test_class_order[ensemble_class_idx]
            ensemble_max_prob = ensemble_pred[ensemble_class_idx]

            print(f"  ENSEMBLE predicts: {ensemble_class_name} ({ensemble_max_prob:.3f})")

            # Show if ensemble differs from individual models
            individual_predictions = [np.argmax(pred) for pred in individual_preds.values()]
            if ensemble_class_idx not in individual_predictions:
                print(f"  🎯 ENSEMBLE EFFECT: Ensemble chose different class than any individual model!")
            elif len(set(individual_predictions)) > 1:
                print(f"  🎯 ENSEMBLE EFFECT: Models disagreed, ensemble made consensus decision!")

        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            continue

# Method 4: Show mathematical proof
print(f"\n=== MATHEMATICAL PROOF ===")
print("Individual model predictions are combined as:")
print("ensemble_prob[class] = (prob_B2[class] + prob_B3[class] + prob_B4[class]) / 3")
print("\nThis creates NEW predictions that may differ from any individual model.")
print("The ensemble AUC is calculated on these NEW predictions, not averaged from individual AUCs.")

print(f"\nFinal verification:")
print(f"  Real ensemble AUC: {corrected_ensemble_auc:.6f}")
print(f"  Average of AUCs:   {fake_ensemble_auc:.6f}")
print(f"  Difference:        {abs(difference):.6f}")

if abs(difference) > 0.0001:
    print(" DEFINITIVE PROOF: This is a real ensemble, not just averaged AUCs!")
else:
    print("  Values are very close, which can happen when models perform similarly.")
    print("   But the process is still a real ensemble - we average predictions, not AUCs.")