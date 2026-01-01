# CelebA Dataset Setup

The **CelebFaces Attributes (CelebA)** dataset is required for face-related notebooks. This dataset is not included in the repository due to its size (~1.3GB).

---

## Required For

| Notebook | Description |
|----------|-------------|
| `v1/notebooks/03_05_vae_faces_train.ipynb` | VAE training on faces |
| `v1/notebooks/03_06_vae_faces_analysis.ipynb` | VAE face analysis |
| `v1/notebooks/04_03_wgangp_faces_train.ipynb` | WGAN-GP on faces |

---

## Option 1: Automated Download (Recommended)

Use the provided download script:

```bash
# From project root
cd v1
bash scripts/download_celeba_kaggle.sh
```

**Prerequisites:**
- Kaggle credentials configured in `.env`:
  ```env
  KAGGLE_USERNAME=your_username
  KAGGLE_KEY=your_api_key
  ```

The script will:
1. Download from Kaggle
2. Extract to correct directory
3. Verify the structure

---

## Option 2: Manual Download

### 1. Download the Dataset

Download `img_align_celeba.zip` from:
- **Kaggle** (Recommended): [https://www.kaggle.com/datasets/jessicali9530/celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- **Official Website**: [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

### 2. Setup Directory Structure

The notebooks expect this structure for Keras `flow_from_directory`:

```
Generative_Deep_Learning/
└── v1/
    └── data/
        └── img_align_celeba/
            └── images/          <-- Create this subdirectory!
                ├── 000001.jpg
                ├── 000002.jpg
                └── ... (202,599 images total)
```

### 3. Extract and Organize

```bash
# Create directory
mkdir -p v1/data/img_align_celeba/images

# Extract (adjust path to your download location)
unzip ~/Downloads/img_align_celeba.zip -d v1/data/img_align_celeba/

# If images extracted to a subfolder, move them:
mv v1/data/img_align_celeba/img_align_celeba/* v1/data/img_align_celeba/images/
rmdir v1/data/img_align_celeba/img_align_celeba
```

---

## Verification

The notebooks look for images using this pattern:

```python
from glob import glob
import os

# From v1/notebooks directory
images = glob(os.path.join('../data/img_align_celeba/', '*/*.jpg'))
print(f"Found {len(images)} images")
```

Expected output: `Found 202599 images`

### Quick Check

```bash
# Count images
ls -1 v1/data/img_align_celeba/images/*.jpg | wc -l
# Should output: 202599
```

---

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total Images | 202,599 |
| Original Size | 178 × 218 pixels |
| Processed Size | 128 × 128 (in notebooks) |
| File Format | JPEG |
| Total Size | ~1.3 GB |

---

## Troubleshooting

### "No images found"

1. Check the directory structure matches exactly
2. Ensure images are in `images/` subdirectory (not directly in `img_align_celeba/`)
3. Verify file extension is `.jpg` not `.jpeg`

### Download Issues on Kaggle

1. Accept the dataset's terms on Kaggle website first
2. Verify Kaggle API credentials: `kaggle datasets list`
3. Check available disk space (~2GB needed for download + extraction)

---

## Related Documentation

- **[GPU_SETUP.md](GPU_SETUP.md)** - Batch size recommendations for CelebA
- **[NOTEBOOK_STANDARDIZATION.md](NOTEBOOK_STANDARDIZATION.md)** - Notebook workflow
- **[../v1/AGENTS.md](../v1/AGENTS.md)** - V1 notebook conventions
