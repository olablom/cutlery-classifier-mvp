# Photo Collection Guide – Cutlery Classifier Project

## Equipment Required

**Hardware:**

- iPhone 13 camera
- White kitchen island (smooth surface)
- Philips Hue Ensis lamp (overhead)
- Black electrical tape (conveyor belt simulation)
- Phone holder/tripod (fixed position)
- White A4 paper (photography base)

**Measurements:**

- Conveyor belt: 80mm width × 200–250mm length (taped on white A4)
- Mobile camera: 40–60 cm above surface, directly overhead

## Lighting Setup (Philips Hue Ensis)

**Configuration:**

- Color temperature: 4000–5000K (neutral white)
- Brightness: 80–100%
- Position: Directly overhead
- Quality control: No shadows or side reflections

## Camera Optimization – iPhone 13

**Settings:**

- Resolution: 4032×3024 (12MP)
- Format: JPEG
- HDR: OFF
- Focus: Tap on cutlery, hold to lock
- Exposure: Lock exposure manually
- Timer: 3 seconds to avoid camera shake

**Focus Lock & Exposure:**

1. Tap on the cutlery piece
2. Hold down to lock focus/exposure
3. Adjust brightness with slider if needed

## Conveyor Belt Simulation

**Setup Process:**

1. Place white A4 paper on kitchen island
2. Use black electrical tape to mark area:
   - Width: 80mm
   - Length: 200–250mm
3. Ensure tape lies completely flat
4. Frame the area with tape for visual contrast

## Camera Mount

**Requirements:**

- Use tripod, box, or build fixture to keep iPhone completely still and level
- Camera should be 45–60 cm above surface
- Position camera so entire 80mm belt is visible with margins around edges

## Image Quality Checklist (Before Each Photo Session)

**Verification Points:**

- [ ] Entire belt (8cm) visible in frame
- [ ] Cutlery positioned within black frame
- [ ] No shadows from cutlery, hands, or phone
- [ ] Even lighting without hot spots
- [ ] Sharp focus on cutlery
- [ ] No reflections from lamp
- [ ] Consistent camera angle (directly overhead)

## File Structure & Naming

**Directory Structure:**

```
data/raw/
├── fork/     # Forks
├── knife/    # Knives
└── spoon/    # Spoons
```

**Naming Convention:**

- `fork_01.jpg`, `fork_02.jpg`, etc.
- `knife_01.jpg`, `knife_02.jpg`, etc.
- `spoon_01.jpg`, `spoon_02.jpg`, etc.

## Image Variation per Class (40 images per class)

**Position Variation (10 images):**

- Centered placement
- Angled 15° left/right

**Rotation Variation (10 images):**

- 0°, 45°, 90°, 135°

**Placement Variation (20 images):**

- Different positions within belt area

**Important:** Maintain consistent background, lighting, and camera angle throughout all images for each class.

## Time Estimation

**Session Breakdown:**

- Setup rigging: 15 minutes
- Photo session (3 classes × 40 images): 60 minutes
- Dataset validation: 2 minutes
- Training + evaluation: 30–40 minutes
- Inference/demo: 10 minutes

**Total Time:** ~2 hours from setup to working model

## Backup & Validation

**Quick Check (after each class):**

```bash
python scripts/validate_dataset.py --quick-check
```

**Backup Strategy:**

- Upload each class to Google Drive/OneDrive immediately
- Alternative Git backup:

```bash
git add .
git commit -m "Added fork images"
```

## Troubleshooting

**Problem:** Blurry images  
**Solution:** Use timer, ensure phone is completely still

**Problem:** Uneven lighting  
**Solution:** Adjust lamp position and color temperature

**Problem:** Too many shadows  
**Solution:** Use diffused lighting, move hands away

**Problem:** Images too dark/bright  
**Solution:** Adjust exposure, lock correct point

## Post-Collection Workflow

**Step-by-Step Commands:**

```bash
# 1. Validate dataset
python scripts/validate_dataset.py

# 2. Create train/validation/test splits
python scripts/prepare_dataset.py --create-splits

# 3. Train the model
python scripts/train_type_detector.py --epochs 30

# 4. Evaluate performance
python scripts/evaluate_model.py --model models/checkpoints/type_detector_best.pth

# 5. Test inference
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --visualize
```

## Quality Assurance

**Expected Results:**

- Dataset size: 120 images total (40 per class)
- Training accuracy: >85%
- Validation accuracy: >80%
- Consistent image quality across all classes

## Success Criteria

**Technical Requirements:**

- [ ] All images meet quality checklist
- [ ] Consistent lighting and positioning
- [ ] No corrupted or duplicate files
- [ ] Proper file naming convention
- [ ] Successful model training completion

---

**READY!**

You are now prepared to capture perfect training images for the Cutlery Classifier! This systematic approach will ensure high-quality data and excellent model performance.

**Good luck – this will be an impressive project! 🚀**
