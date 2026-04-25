#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from PIL import Image
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# === CONFIG ===
BASE_DIR = Path("/data/brhanu/thesis_project")
MANIFEST_PATH = BASE_DIR / "final_dataset/metadata/manifest_v2.csv"
IMAGES_DIR = BASE_DIR / "final_dataset/images/restored"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
SCENE_CLASSES = ["teaching", "family", "playing", "landscape", "drawing"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(SCENE_CLASSES)}

class SceneClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SceneClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

class ArchivalSceneDataset(Dataset):
    def __init__(self, df, preprocess):
        self.df = df
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = IMAGES_DIR / f"{row['image_id']}.jpg"
        image = self.preprocess(Image.open(img_path))
        label = CLASS_TO_IDX.get(row['vqa_primary_scene'], -1)
        return image, label

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading CLIP features on {device}...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=device)
    
    # Load and filter manifest
    df = pd.read_csv(MANIFEST_PATH)
    df = df[df['vqa_primary_scene'].isin(SCENE_CLASSES)].reset_index(drop=True)
    print(f"📊 Training on {len(df)} archival images with labeled scenes.")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_loader = DataLoader(ArchivalSceneDataset(train_df, preprocess), batch_size=32, shuffle=True)
    val_loader = DataLoader(ArchivalSceneDataset(val_df, preprocess), batch_size=32)

    # Freeze CLIP, train head
    classifier = SceneClassifier(512, len(SCENE_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

    print("🧠 Training Scene Classifier head...")
    for epoch in range(10):
        classifier.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            with torch.no_grad():
                features = clip_model.encode_image(imgs)
                features = features / features.norm(dim=-1, keepdim=True)
            
            outputs = classifier(features.float())
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                features = clip_model.encode_image(imgs)
                features = features / features.norm(dim=-1, keepdim=True)
                outputs = classifier(features.float())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"📉 Loss: {total_loss/len(train_loader):.4f} | ✅ Val Acc: {100 * correct / total:.2f}%")

    model_path = BASE_DIR / "results/scene_classifier_head_v1.pth"
    torch.save(classifier.state_dict(), model_path)
    print(f"🎉 Scene Classifier saved to {model_path}")

if __name__ == "__main__":
    main()
