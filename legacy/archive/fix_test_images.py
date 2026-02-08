import os
from pathlib import Path
import shutil

exts = [".jpg", ".jpeg", ".png"]
base = Path("data/processed/test")
backup = Path("data/processed/test_unused_classes")
classes = ["fork", "knife", "spoon"]

# Skapa mappstruktur
for c in classes:
    (base / c).mkdir(parents=True, exist_ok=True)

found = {c: [] for c in classes}

# Leta upp bilder
for root, _, files in os.walk("."):
    for f in files:
        if any(f.lower().endswith(e) for e in exts):
            fl = f.lower()
            for c in classes:
                if c in fl and len(found[c]) < 20:
                    src = os.path.join(root, f)
                    dst = base / c / f
                    if not dst.exists():
                        shutil.copy2(src, dst)
                        found[c].append(f)

print("Kopierade bilder:")
for c in classes:
    print(f"{c}: {len(found[c])} st")

# --- NY DEL: Flytta tomma klassmappar till backup ---
backup.mkdir(parents=True, exist_ok=True)
move_count = 0
for subdir in base.iterdir():
    if subdir.is_dir():
        has_image = any(
            f.suffix.lower() in exts for f in subdir.iterdir() if f.is_file()
        )
        if not has_image:
            shutil.move(str(subdir), str(backup / subdir.name))
            move_count += 1
print(f"Flyttade {move_count} tomma klassmappar till {backup}")

# Kopiera finetunad modell till rätt namn om den inte finns
src = Path("models/checkpoints/type_detector_best_tuned.pth")
dst = Path("models/checkpoints/type_detector_finetuned.pth")
if src.exists() and not dst.exists():
    shutil.copy2(src, dst)
    print(f"Kopierade {src} till {dst}")
