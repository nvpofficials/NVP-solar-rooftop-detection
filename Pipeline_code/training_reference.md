## CODE USED TO TRAIN THE MODEL WHICH IS EXCECUTED IN GOOGLE COLAB ON (T4 GPU)


# Edit these variables as needed BEFORE running:
DRIVE_ZIP_FOLDER = "/content/drive/MyDrive/solar_dataset_zip"   # <--- Path in your Google Drive where you uploaded ZIPs
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/rooftop_solar_training"  # where checkpoints & outputs will be saved
MERGED_DIR = "/content/datasets/rooftop_merged"  # local merged dataset path
ROBOFLOW_API_KEY = ""  # leave blank; not required for Drive-ZIP approach

# Training hyperparameters (tuned for Colab Free with good accuracy)
WARMUP = {"model": "yolov8m.pt", "imgsz": 512, "epochs": 40, "batch": 12}
FINETUNE = {"model": None, "imgsz": 768, "epochs": 80, "batch": 6}
SAVE_PERIOD = 5
CLASS_NAMES = ["solar_panel"]
NUM_CLASSES = len(CLASS_NAMES)

# ------------------ Begin runtime (do not change below unless you know what you are doing) ------------------
import os, sys, json, glob, shutil, zipfile, random
from pathlib import Path
print("Starting Drive-ZIP -> merge -> train pipeline")

# 1) Install required packages
print("Installing packages (ultralytics, pycocotools)...")
!pip install -U ultralytics pycocotools roboflow --quiet

# 2) Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)
print("Drive mounted. Project dir:", DRIVE_PROJECT_DIR)
print("Merged dir:", MERGED_DIR)

# 3) Unzip all zip files in the DRIVE_ZIP_FOLDER into /content/datasets/unzipped/*
UNZIP_ROOT = "/content/datasets/unzipped"
os.makedirs(UNZIP_ROOT, exist_ok=True)
zip_files = glob.glob(os.path.join(DRIVE_ZIP_FOLDER, "*.zip"))
if not zip_files:
    print("WARNING: No .zip files found in", DRIVE_ZIP_FOLDER)
else:
    print("Found zip files:", zip_files)
for z in zip_files:
    try:
        target = os.path.join(UNZIP_ROOT, Path(z).stem)
        if os.path.exists(target) and os.listdir(target):
            print("Already extracted:", target)
            continue
        os.makedirs(target, exist_ok=True)
        print("Extracting", z, "->", target)
        with zipfile.ZipFile(z, 'r') as zip_ref:
            zip_ref.extractall(target)
    except Exception as e:
        print("Failed to extract", z, ":", e)

# 4) Utility: COCO -> YOLO converter
def coco_to_yolo(coco_json_path, images_dir, out_labels_dir, map_all_to_zero=True):
    # Converts COCO json to YOLO txt files; if map_all_to_zero -> single class 0
    import json, os
    os.makedirs(out_labels_dir, exist_ok=True)
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
    imgs = {img['id']: img for img in coco.get('images', [])}
    anns = {}
    for ann in coco.get('annotations', []):
        anns.setdefault(ann['image_id'], []).append(ann)
    for img_id, info in imgs.items():
        fname = info['file_name']
        w, h = info.get('width', None), info.get('height', None)
        if w is None or h is None:
            # try open with PIL to get size (slower)
            try:
                from PIL import Image
                im = Image.open(os.path.join(images_dir, fname))
                w, h = im.size
            except:
                continue
        lines = []
        for ann in anns.get(img_id, []):
            bbox = ann.get('bbox', None)
            if not bbox:
                continue
            x, y, bw, bh = bbox
            xc = (x + bw/2) / w
            yc = (y + bh/2) / h
            nw = bw / w
            nh = bh / h
            cls = 0 if map_all_to_zero else ann.get('category_id', 0) - 1
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        if lines:
            labpath = os.path.join(out_labels_dir, Path(fname).stem + ".txt")
            open(labpath, "w").write("\n".join(lines))
    print("Converted COCO->YOLO for", coco_json_path)

# 5) Discover image & label folders inside UNZIP_ROOT, collect them
print("Scanning extracted folders for images and labels...")
img_dirs = []
label_dirs = []
for root, dirs, files in os.walk(UNZIP_ROOT):
    for d in dirs:
        dp = os.path.join(root, d)
        # common patterns
        low = d.lower()
        if low in ("images", "images_train", "img", "imgs", "images_all", "jpgs", "jpeg", "pngs"):
            img_dirs.append(dp)
        if low in ("labels", "labels_train", "ann", "annotations", "labels_yolo", "yolo"):
            label_dirs.append(dp)
    # also detect train/valid/test structures by keywords
    for f in files:
        if f.endswith(".json"):
            # maybe COCO annotation JSON
            cand = os.path.join(root, f)
            try:
                with open(cand,'r') as fh:
                    j = json.load(fh)
                if "annotations" in j and "images" in j:
                    images_root = os.path.join(root, "images")
                    # if images present, convert
                    br = os.path.join(root, "labels_coco_converted")
                    coco_to_yolo(cand, images_root if os.path.exists(images_root) else root, br)
                    label_dirs.append(br)
            except Exception:
                pass

# fallback: if any folder directly contains image files (jpg/png) treat as images dir
for root, dirs, files in os.walk(UNZIP_ROOT):
    imgs_here = [f for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if imgs_here:
        if root not in img_dirs:
            img_dirs.append(root)

# Also include top-level Roboflow standard paths (some zip extract direct)
for p in glob.glob(os.path.join(UNZIP_ROOT, "*")):
    for sub in ("train/images","train/img","valid/images","valid/img","images"):
        cand = os.path.join(p, sub)
        if os.path.exists(cand) and cand not in img_dirs:
            img_dirs.append(cand)
    for sub in ("train/labels","valid/labels","labels"):
        cand = os.path.join(p, sub)
        if os.path.exists(cand) and cand not in label_dirs:
            label_dirs.append(cand)

img_dirs = list(dict.fromkeys(img_dirs))
label_dirs = list(dict.fromkeys(label_dirs))
print("Found image dirs (examples):", img_dirs[:6])
print("Found label dirs (examples):", label_dirs[:6])

# 6) Merge all images and labels into MERGED_DIR/images & MERGED_DIR/labels (rename files sequentially)
os.makedirs(os.path.join(MERGED_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(MERGED_DIR, "labels"), exist_ok=True)

def copy_and_normalize(img_dirs, label_dirs, merged_images_dir, merged_labels_dir, start_index=0):
    idx = start_index
    for img_dir in img_dirs:
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*"))):
            if not img_path.lower().endswith(('.jpg','.jpeg','.png')): continue
            ext = Path(img_path).suffix
            new_name = f"{idx:08d}{ext}"
            dst_img = os.path.join(merged_images_dir, new_name)
            if not os.path.exists(dst_img):
                try:
                    shutil.copy2(img_path, dst_img)
                except Exception as e:
                    print("Copy failed:", img_path, e); continue
            # try find a label in label_dirs with same stem
            found = None
            stem = Path(img_path).stem
            for lbl_root in label_dirs:
                cand = os.path.join(lbl_root, stem + ".txt")
                if os.path.exists(cand):
                    found = cand; break
            # if found, remap classes to 0 and write new label
            if found:
                text = open(found,'r').read().strip()
                if text:
                    lines = []
                    for line in text.splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            coords = parts[1:]
                            lines.append("0 " + " ".join(coords))
                    open(os.path.join(merged_labels_dir, Path(new_name).stem + ".txt"), "w").write("\n".join(lines))
            idx += 1
    return idx

print("Merging datasets (this may take a few minutes)...")
total = copy_and_normalize(img_dirs, label_dirs, os.path.join(MERGED_DIR,"images"), os.path.join(MERGED_DIR,"labels"))
print("Merged images count:", total)

# 7) Create train/val/test splits (80/10/10)
all_imgs = sorted(glob.glob(os.path.join(MERGED_DIR,"images","*")))
random.seed(42)
random.shuffle(all_imgs)
n = len(all_imgs)
if n == 0:
    raise RuntimeError("No images found after merge. Check that your ZIPs contained images.")
train_end = int(0.8 * n)
val_end = int(0.9 * n)

def make_split(split_name, img_list):
    timg = os.path.join(MERGED_DIR, split_name, "images"); os.makedirs(timg, exist_ok=True)
    tlbl = os.path.join(MERGED_DIR, split_name, "labels"); os.makedirs(tlbl, exist_ok=True)
    for p in img_list:
        fname = Path(p).name
        shutil.copy2(p, os.path.join(timg, fname))
        lbl_src = os.path.join(MERGED_DIR, "labels", Path(p).stem + ".txt")
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, os.path.join(tlbl, Path(lbl_src).name))

make_split("train", all_imgs[:train_end])
make_split("val", all_imgs[train_end:val_end])
make_split("test", all_imgs[val_end:])
print(f"Created splits. Train:{train_end}, Val:{val_end-train_end}, Test:{n-val_end}")

# 8) Write dataset.yaml
dataset_yaml = {
    "path": MERGED_DIR,
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": NUM_CLASSES,
    "names": CLASS_NAMES
}
with open(os.path.join(MERGED_DIR, "dataset.yaml"), "w") as f:
    json.dump(dataset_yaml, f, indent=2)
print("Wrote dataset.yaml to", os.path.join(MERGED_DIR, "dataset.yaml"))

# 9) Train: helper which resumes if checkpoint exists
from ultralytics import YOLO
import torch
print("CUDA:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

PROJECT = DRIVE_PROJECT_DIR + "/runs_yolov8"
os.makedirs(PROJECT, exist_ok=True)

def train_stage(start_ckpt, model_pretrained, imgsz, epochs, batch, project, run_name):
    # if a checkpoint path is provided it will be used to initialize the model
    if start_ckpt and os.path.exists(start_ckpt):
        print("Resuming from checkpoint:", start_ckpt)
        model = YOLO(start_ckpt)
    else:
        print("Starting from pretrained:", model_pretrained)
        model = YOLO(model_pretrained)
    # call train
    model.train(
        data=os.path.join(MERGED_DIR,"dataset.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=run_name,
        save=True,
        save_period=SAVE_PERIOD,
        workers=2,
        device=0,
        exist_ok=True
    )
    # return path to best or last
    weights_dir = os.path.join(project, run_name, "weights")
    best = os.path.join(weights_dir, "best.pt")
    last = os.path.join(weights_dir, "last.pt")
    if os.path.exists(best): return best
    if os.path.exists(last): return last
    return None

# Decide resume logic: check for existing runs in Drive_project
stage1_run = "rooftop_yolov8_stage1"
stage2_run = "rooftop_yolov8_stage2"

# If user previously ran, find last checkpoint
def find_last_ckpt(project, run_name):
    wdir = os.path.join(project, run_name, "weights")
    cand_last = os.path.join(wdir, "last.pt")
    cand_best = os.path.join(wdir, "best.pt")
    if os.path.exists(cand_last): return cand_last
    if os.path.exists(cand_best): return cand_best
    return None

# Stage1: warmup
ckpt1 = find_last_ckpt(PROJECT, stage1_run)
if ckpt1:
    print("Found existing Stage1 checkpoint:", ckpt1)
ckpt1_out = train_stage(ckpt1, WARMUP["model"], WARMUP["imgsz"], WARMUP["epochs"], WARMUP["batch"], PROJECT, stage1_run)
print("Stage1 output checkpoint:", ckpt1_out)

# Stage2: fine-tune (resume from stage1 best)
if ckpt1_out:
    ckpt2 = find_last_ckpt(PROJECT, stage2_run) or ckpt1_out
    ckpt2_out = train_stage(ckpt2, ckpt1_out, FINETUNE["imgsz"], FINETUNE["epochs"], FINETUNE["batch"], PROJECT, stage2_run)
    print("Stage2 output checkpoint:", ckpt2_out)
else:
    print("Stage1 produced no checkpoint; cannot proceed to Stage2 automatically.")

# 10) Copy final best.pt to Drive final_models folder and /content
def copy_final(project, run_name, dest_drive_folder):
    wdir = os.path.join(project, run_name, "weights")
    best = os.path.join(wdir, "best.pt")
    last = os.path.join(wdir, "last.pt")
    src = best if os.path.exists(best) else last if os.path.exists(last) else None
    if not src:
        print("No trained checkpoint found in", wdir)
        return None
    os.makedirs(dest_drive_folder, exist_ok=True)
    dst = os.path.join(dest_drive_folder, Path(src).name)
    shutil.copy2(src, dst)
    shutil.copy2(src, "/content/" + Path(src).name)
    print("Copied final checkpoint to:", dst, "and to /content/")
    return dst

FINAL_DIR = os.path.join(DRIVE_PROJECT_DIR, "final_models")
final = copy_final(PROJECT, stage2_run, FINAL_DIR)
print("Pipeline finished. Final model:", final)
# --------------------------------------------------------------------------------------------------------------
