# CelebA Dataset Setup

The **CelebFaces Attributes (CelebA)** dataset is required for the `v1/notebooks/03_05_vae_faces_train.ipynb` notebook. This dataset is not included in the repository due to its size.

## 1. Download the Dataset

You can download `img_align_celeba.zip` from:
*   **Kaggle** (Recommended): [https://www.kaggle.com/datasets/jessicali9530/celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
*   **Official Website**: [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## 2. Setup Directory Structure

The notebook expects the following directory structure to work with Keras `flow_from_directory`:

```
Generative_Deep_Learning/
├── data/
│   └── img_align_celeba/
│       └── images/          <-- Create this subdirectory!
│           ├── 000001.jpg
│           ├── 000002.jpg
│           └── ...
```

### Steps:
1.  Create the directory: `mkdir -p data/img_align_celeba/images`
2.  Download `img_align_celeba.zip`.
3.  Unzip the content directly into `data/img_align_celeba/images/`.
    *   Ensure the `.jpg` files are directly inside `images/`.
    *   If the zip extracts to a folder (e.g., `img_align_celeba`), move the contents so the final path is `data/img_align_celeba/images/[file].jpg`.

## 3. Verification

The notebook looks for images using this pattern:
`glob(os.path.join('./data/img_align_celeba/', '*/*.jpg'))`

This will match `data/img_align_celeba/images/*.jpg`.
