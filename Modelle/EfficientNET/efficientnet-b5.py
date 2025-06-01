# JUPYTERLAB BEFEHL
!pip install tqdm --user

# Schritt 1: Pakete installieren
!pip install --user datasets transformers==4.37.2 torch==2.0.1 torchvision timm scikit-learn matplotlib

# JUPYTERLAB BEFEHL
import sys
import os
sys.path.append(f"{os.environ['HOME']}/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")

# Schritt 2: Imports
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoImageProcessor, EfficientNetForImageClassification
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

import os
import requests
from tqdm import tqdm
import zipfile

# URLs and paths Training
url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
zip_path = "ISIC_2019_Training_Input.zip"
extract_dir = "ISIC_2019_Training_Input"

# 1) Download the ZIP if needed
if not os.path.exists(zip_path):
    print(f"Downloading {zip_path}...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(zip_path, 'wb') as f, tqdm(
            desc=zip_path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))
else:
    print(f"{zip_path} already exists, skipping download.")


# URLs and paths Test
url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Input.zip"
zip_path = "ISIC_2019_Test_Input.zip"
extract_dir = "ISIC_2019_Test_Input"

# 2) Download the ZIP if needed
if not os.path.exists(zip_path):
    print(f"Downloading {zip_path}...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(zip_path, 'wb') as f, tqdm(
            desc=zip_path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))
else:
    print(f"{zip_path} already exists, skipping download.")

# URLs and paths CSV Test
url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_GroundTruth.csv"
zip_path = "ISIC_2019_Test_GroundTruth.csv"
extract_dir = "ISIC_2019_Test_GroundTruth"

# 3) Download the ZIP if needed
if not os.path.exists(zip_path):
    print(f"Downloading {zip_path}...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(zip_path, 'wb') as f, tqdm(
            desc=zip_path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))
else:
    print(f"{zip_path} already exists, skipping download.")


# URLs and paths CSV Test
url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
zip_path = "ISIC_2019_Training_GroundTruth.csv"
extract_dir = "ISIC_2019_Training_GroundTruth"

# 3) Download the ZIP if needed
if not os.path.exists(zip_path):
    print(f"Downloading {zip_path}...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(zip_path, 'wb') as f, tqdm(
            desc=zip_path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))
else:
    print(f"{zip_path} already exists, skipping download.")

# Testbilder entpacken
# NUR MACHEN WENN MAN DAS ERSTE MAL DIE ZIP HOCHGELADEN HAT + NOCH NICHT ENTPACKT
import zipfile

with zipfile.ZipFile("ISIC_2019_Test_Input.zip", 'r') as zip_ref:
    zip_ref.extractall("ISIC_2019_Test_Input")

print("Testbilder entpackt: ./ISIC_2019_Test_Input")

# NUR MACHEN WENN MAN DAS ERSTE MAL DIE ZIP HOCHGELADEN HAT + NOCH NICHT ENTPACKT
import zipfile

with zipfile.ZipFile("ISIC_2019_Training_Input.zip", 'r') as zip_ref:
    zip_ref.extractall("ISIC_2019_Training_Input")

print(" Trainingsbilder entpackt: ./ISIC_2019_Training_Input")

# Schritt 4: Labels mappen
df_train = pd.read_csv("ISIC_2019_Training_GroundTruth.csv")

class_columns = [col for col in df_train.columns if col not in ['image', 'score_weight', 'validation_weight']]
class_columns = sorted(class_columns)

# Mapping
label2id = {label: idx for idx, label in enumerate(class_columns)}
id2label = {idx: label for label, idx in label2id.items()}

print(label2id)

# Schritt 5: Processor laden

processor = AutoImageProcessor.from_pretrained(
    "google/efficientnet-b5",
    do_rescale=False
)

# Schritt 6: Dataset Klasse + collate_fn
def collate_fn(batch):
    inputs = {k: torch.stack([d[0][k] for d in batch]) for k in batch[0][0]}
    labels = torch.tensor([d[1] for d in batch])
    return inputs, labels

class ISICDataset(Dataset):
    def __init__(self, image_dir, label_csv_or_df, processor, label2id, transform=None):
        self.image_dir = image_dir
        self.processor = processor
        self.label2id = label2id
        self.transform = transform

        if isinstance(label_csv_or_df, str):
            df = pd.read_csv(label_csv_or_df)
        else:
            df = label_csv_or_df

        class_columns = sorted([col for col in df.columns if col not in ['image', 'score_weight', 'validation_weight']])
        self.image_labels = {
            row['image']: class_columns.index(class_name)
            for _, row in df.iterrows()
            for class_name in class_columns
            if row[class_name] == 1
        }

        self.images = list(self.image_labels.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label_idx = self.image_labels[img_name]
        image = Image.open(os.path.join(self.image_dir, img_name + ".jpg")).convert("RGB")

        if self.transform:
            image = self.transform(image)

        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs, label_idx



class ISICTestDatasetFromFolder(Dataset):
    def __init__(self, image_dir, label_csv, processor, label2id, transform=None):
        self.image_dir = image_dir
        self.processor = processor
        self.label2id = label2id
        self.transform = transform

        df = pd.read_csv(label_csv)

        class_columns = [col for col in df.columns if col not in ['image', 'score_weight', 'validation_weight']]
        class_columns = sorted(class_columns)

        expected = sorted(label2id.keys())
        assert class_columns == expected, f"Inkompatible Labels: {class_columns} ≠ {expected}"

        # Bild → Label-Index abbilden
        self.image_labels = {
            row['image']: class_columns.index(class_name)
            for _, row in df.iterrows()
            for class_name in class_columns
            if row[class_name] == 1
        }

        self.images = list(self.image_labels.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label = self.image_labels[img_name]
        image = Image.open(os.path.join(self.image_dir, img_name + ".jpg")).convert("RGB")

        if self.transform:
            image = self.transform(image)

        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs, label

# Schritt 6a: Augmentierung
train_transforms = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.3, contrast=0.3)
    ], p=0.8),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomAffine(
        degrees=15,
        scale=(0.9, 1.1),
        shear=10,
        translate=(0.05, 0.05),
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=0
    ),
    transforms.Resize((456, 456)),
    transforms.ToTensor()
])

# Für Val/Test (nur Resize + ToTensor)
val_test_transforms = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.ToTensor(),
])

# Schritt 7: Train/Val Split mit offizieller ISIC_2019_Training_Input

df_train = pd.read_csv("ISIC_2019_Training_GroundTruth.csv")

# Shuffle und Split in 90% Train, 10% Val
from sklearn.model_selection import train_test_split

df_train["main_label"] = df_train[class_columns].idxmax(axis=1)

df_train_split, df_val_split = train_test_split(
    df_train,
    test_size=0.1,
    stratify=df_train["main_label"],
    random_state=42
)

# Datasets initialisieren
train_dataset = ISICDataset(
    image_dir="./ISIC_2019_Training_Input/ISIC_2019_Training_Input",
    label_csv_or_df=df_train_split,
    processor=processor,
    label2id=label2id,
    transform=train_transforms
)

# Klassenverteilung im Trainingsdatensatz analysieren
train_labels = [
    train_dataset.image_labels[img]
    for img in train_dataset.images
]

unique_labels = np.unique(train_labels)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_labels,
    y=train_labels
)

class_weights_full = np.zeros(len(label2id), dtype=np.float32)
for i, cls in enumerate(unique_labels):
    class_weights_full[cls] = class_weights[i]

class_weights_tensor = torch.tensor(class_weights_full, dtype=torch.float).to('cuda')

val_dataset = ISICDataset(
    image_dir="./ISIC_2019_Training_Input/ISIC_2019_Training_Input",
    label_csv_or_df=df_val_split,
    processor=processor,
    label2id=label2id,
    transform=val_test_transforms
)

print(f"Val-Dataset Größe: {len(val_dataset)}")
val_labels_check = [val_dataset.image_labels[img] for img in val_dataset.images]
print(f"Eindeutige Labels im val_dataset: {np.unique(val_labels_check)}")
print(f"Val-Dataset Klassenverteilung: {np.bincount(val_labels_check)}")

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=1,
    persistent_workers=False,
    pin_memory=True,
    prefetch_factor=None
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=1,
    persistent_workers=False,
    pin_memory=True,
    prefetch_factor=None
)

# Schritt 7.1: Test Loader mit offiziellen ISIC-Testdaten
test_dataset = ISICTestDatasetFromFolder(
    image_dir="./ISIC_2019_Test_Input/ISIC_2019_Test_Input",
    label_csv="./ISIC_2019_Test_GroundTruth.csv",
    processor=processor,
    label2id=label2id,
    transform=val_test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=1,
    persistent_workers=False,
    pin_memory=True
)

# IMPORT ZIPFILE, NUR WENN MAN VON PC IN NEUE RUNTIME LÄDT
import zipfile

with zipfile.ZipFile("efficientnet-b5-isic.zip", 'r') as zip_ref:
    zip_ref.extractall("efficientnet-b5-isic")

print(" Modellordner entpackt: ./efficientnet-b5-isic")

# Schritt 7.2: Modell ggf. laden statt neu trainieren
model_path = "./efficientnet-b5-isic"

if os.path.exists(model_path):
    print("Modell gefunden")
    model = EfficientNetForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path, do_rescale=False)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    #skip_training = True
else:
    print("Kein gespeichertes Modell gefunden")
    #skip_training = False

# NUR WENN MAN NOCH KEIN PRETRAINED MODELL HAT (KEIN MODELL AUS ZIP IMPORTIERT SIEHE VOR SCHRITT 7.5)
# Schritt 8: Modell laden
model = EfficientNetForImageClassification.from_pretrained(
    "google/efficientnet-b5",
    num_labels=len(label2id),
    ignore_mismatched_sizes=True
)
model = model.to('cuda')

#Feature Extractor einfrieren
for param in model.efficientnet.parameters():
    param.requires_grad = False

# Schritt 9: Optimizer und Loss definieren
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Schritt 10–12: Training + Evaluation£
best_val_loss = float('inf')
early_stopping_counter = 0
EPOCHS = 100
EARLY_STOPPING = 10
val_aucs = []
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        inputs, labels = batch
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        labels = labels.to('cuda')

        optimizer.zero_grad()

        with autocast():
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.4f}")
    train_losses.append(avg_train_loss)

    # Validation mit ROC/AUC
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc=f"Validierung Epoch {epoch+1}")):
            inputs, labels = batch

            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            labels = labels.to('cuda')
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")
    val_losses.append(avg_val_loss)

    y_true = np.array(all_labels)
    y_score = np.array(all_probs)

    if len(np.unique(y_true)) < 2:
        print(f"AUC nicht berechnet – nur eine Klasse in y_true (Epoche {epoch+1})")
        val_aucs.append(None)
    else:
        try:
            present_classes = sorted(np.unique(y_true))
            y_score_filtered = y_score[:, present_classes]

            y_score_normalized = y_score_filtered / y_score_filtered.sum(axis=1, keepdims=True)

            auc_macro = roc_auc_score(y_true, y_score_normalized, average='macro', multi_class='ovr', labels=present_classes)
            val_aucs.append(auc_macro)
            print(f"Epoch {epoch+1}: Validation AUC = {auc_macro:.4f}")
        except Exception as e:
            print(f"AUC konnte nicht berechnet werden: {e}")
            val_aucs.append(None)

    # Early Stopping prüfen
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        model.save_pretrained("./efficientnet-b5-isic")
        processor.save_pretrained("./efficientnet-b5-isic")
        print("Modell gespeichert (lokal unter ./efficientnet-b5-isic)")

        # Modellordner zippen
        import shutil
        shutil.make_archive("efficientnet-b5-isic", 'zip', "./efficientnet-b5-isic")
        print("Modell als ZIP-Datei gespeichert: efficientnet-b5-isic.zip")

        print("Modell gespeichert (bester Val-Loss)")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= EARLY_STOPPING:
            print("Early Stopping aktiviert.")
            break

# Schritt 13: Nur ROC AUC + Mittelwert
try:
    y_true = np.array(all_labels)
    y_score = np.array(all_probs)

    present_classes = sorted(np.unique(y_true))
    y_score_filtered = y_score[:, present_classes]
    y_score_normalized = y_score_filtered / y_score_filtered.sum(axis=1, keepdims=True)

    # Macro-AUC berechnen
    auc_macro = roc_auc_score(y_true, y_score_normalized, average='macro', multi_class='ovr', labels=present_classes)
    print(f"\nValidation AUC (macro average): {auc_macro:.4f}")

    # One-vs-Rest ROC-Kurven plotten
    y_true_bin = np.zeros((len(y_true), len(present_classes)))  # ← FIX: present_classes statt train_label2id
    for i, label in enumerate(y_true):
        class_idx = present_classes.index(label)  # ← FIX: Index in present_classes finden
        y_true_bin[i, class_idx] = 1

    plt.figure(figsize=(10, 8))
    for i, class_id in enumerate(present_classes):  # ← FIX: über present_classes iterieren
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_normalized[:, i])  # ← FIX: normalized verwenden
            if len(fpr) < 2 or np.any(np.diff(fpr) < 0):
                print(f"⚠ROC für Klasse {id2label[class_id]} konnte nicht geplottet werden (zu wenig Daten).")
                continue
            plt.plot(fpr, tpr, label=f"{id2label[class_id]} (AUC = {auc(fpr, tpr):.2f})")
        except Exception as e:
            print(f"ROC für Klasse {id2label[class_id]} konnte nicht geplottet werden: {e}")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Kurven je Klasse')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
except Exception as e:
    print("AUC konnte nicht berechnet oder geplottet werden:", e)

# Schritt 14: Test-Auswertung mit ROC AUC + Kurven
model.eval()
all_preds_test = []
all_labels_test = []
all_probs_test = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testdaten"):
        inputs, labels = batch
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        labels = labels.to('cuda')
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_probs_test.extend(probs.cpu().numpy())
        all_preds_test.extend(preds.cpu().numpy())
        all_labels_test.extend(labels.cpu().numpy())

# ROC AUC berechnen mit verbesserter Auswertung
try:
    y_true_test = np.array(all_labels_test)
    y_score_test = np.array(all_probs_test)

    # Fix für UNK-Problem: Nur vorhandene Klassen verwenden
    present_classes = sorted(np.unique(y_true_test))
    y_score_filtered = y_score_test[:, present_classes]
    y_score_normalized = y_score_filtered / y_score_filtered.sum(axis=1, keepdims=True)

    # Macro-AUC berechnen
    auc_test_macro = roc_auc_score(y_true_test, y_score_normalized, average='macro', multi_class='ovr', labels=present_classes)
    print(f"\nTest Macro AUC: {auc_test_macro:.4f}")

    # AUC pro Klasse berechnen
    print(f"\nTest AUC pro Klasse:")
    y_true_bin = label_binarize(y_true_test, classes=present_classes)
    if len(present_classes) == 2:
        y_true_bin = y_true_bin.reshape(-1, 1)

    individual_aucs_test = []
    for i, class_id in enumerate(present_classes):
        if len(present_classes) > 2:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_normalized[:, i])
            class_auc = auc(fpr, tpr)
        else:
            fpr, tpr, _ = roc_curve(y_true_test, y_score_normalized[:, 1])
            class_auc = auc(fpr, tpr)

        individual_aucs_test.append(class_auc)
        print(f"   {id2label[class_id]:>6}: {class_auc:.4f}")

    # ROC-Kurven für Testdaten plotten
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(present_classes)))

    for i, class_id in enumerate(present_classes):
        try:
            if len(present_classes) > 2:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_normalized[:, i])
                class_auc = individual_aucs_test[i]
            else:
                fpr, tpr, _ = roc_curve(y_true_test, y_score_normalized[:, 1])
                class_auc = individual_aucs_test[i]

            if len(fpr) < 2:
                print(f"⚠ ROC für {id2label[class_id]} übersprungen (zu wenig Daten)")
                continue

            plt.plot(fpr, tpr, color=colors[i], lw=2.5,
                     label=f"{id2label[class_id]} (AUC = {class_auc:.3f})")
        except Exception as e:
            print(f"ROC für {id2label[class_id]} fehlgeschlagen: {e}")

    # Random line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='Random (AUC = 0.5)')

    # Plot formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Test ROC Curves - Macro AUC: {auc_test_macro:.4f}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nTest Macro AUC: {auc_test_macro:.4f} | Samples: {len(y_true_test)} | Klassen: {len(present_classes)}")
    print(f"ROC Kurven gespeichert als: test_roc_curves.png")

except Exception as e:
    print("Test AUC konnte nicht berechnet werden:", e)

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.plot(val_aucs, label='Validation AUC', marker='x')
plt.xticks(ticks=range(len(train_losses)), labels=[str(i+1) for i in range(len(train_losses))])
plt.xlabel('Epoche')
plt.ylabel('Wert')
plt.title('Training Loss / Validation Loss / AUC pro Epoche')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()