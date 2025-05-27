# 📸 Photo Collection Guide - Cutlery Classifier MVP

## 🎯 **Target: 240 images total (40 per class)**

### 📋 **Collection Checklist**

#### Fork

- [ ] **IKEA**: 0/40 images → `data/raw/fork/ikea/`
- [ ] **OBH**: 0/40 images → `data/raw/fork/obh/`

#### Knife

- [ ] **IKEA**: 0/40 images → `data/raw/knife/ikea/`
- [ ] **OBH**: 0/40 images → `data/raw/knife/obh/`

#### Spoon

- [ ] **IKEA**: 0/40 images → `data/raw/spoon/ikea/`
- [ ] **OBH**: 0/40 images → `data/raw/spoon/obh/`

---

## 📷 **Photo Guidelines**

### 🛠️ **Setup**

- 📱 **Camera**: Mobile phone camera
- 🟫 **Background**: Light, neutral surface (white plate/tray)
- 💡 **Lighting**: Natural daylight (near window)
- 📏 **Framing**: Fill frame but leave some margin

### 🔄 **Variation Strategy (per manufacturer)**

- **Angles**: Top-down, 45°, side view
- **Orientations**: Horizontal, vertical, diagonal
- **Distances**: Close-up, medium, slightly farther
- **Positions**: Center, slightly off-center

### ⚡ **Quick Workflow**

1. Take 5-10 photos of one piece
2. Move/rotate the cutlery slightly
3. Change your position/height
4. Repeat until you have 40 images
5. Keep background consistent within each session

### 📁 **File Naming**

- Use format: `img_001.jpg`, `img_002.jpg`, etc.
- Save directly to the correct folder

---

## 🧪 **Testing Strategy**

### Phase 1: Quick Test (20 images per class)

- Take 20 images per class first
- Test the training pipeline
- Verify everything works

### Phase 2: Full Dataset (40 images per class)

- Complete to 40 images per class
- Final training and evaluation

---

## ✅ **Success Criteria**

- **Minimum**: 20 images per class (120 total)
- **Target**: 40 images per class (240 total)
- **Quality**: Clear, well-lit, varied angles
- **Consistency**: Similar background within manufacturer

---

## 🚀 **After Collection**

Run these commands to validate and prepare data:

```bash
# Validate dataset
python scripts/prepare_dataset.py --validate-dataset

# Create train/val/test splits
python scripts/prepare_dataset.py --create-splits
```

**Good luck! 📸✨**
