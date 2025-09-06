import os
import shutil
import stat
import hashlib
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from PIL import Image
from flask import Flask, render_template, request, send_file

# =============================
# NLTK Resources
# =============================
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

# =============================
# Configurations
# =============================
dataset_dir = "dataset"
cnn_dataset_dir = "cnn_dataset"
yolo_dataset_dir = "yolo_dataset"

IMG_SIZE = (256, 256)
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # Train, Test, Val

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Synonyms
SYNONYMS = {
    "pics": "images",
    "photos": "images",
    "cat": "feline",
    "dog": "canine",
    "haircut": "hairstyle",
    "cut": "hairstyle",
}

# =============================
# NLP Preprocessing
# =============================
def preprocess_query(query: str) -> str:
    tokens = nltk.word_tokenize(query.lower())
    cleaned_tokens = []
    for word in tokens:
        if word in SYNONYMS:
            word = SYNONYMS[word]
        word = lemmatizer.lemmatize(word)
        if word not in stop_words and word.isalnum():
            cleaned_tokens.append(word)
    return " ".join(cleaned_tokens)

# =============================
# Image Functions
# =============================
def download_images(query, save_dir, limit=50):
    os.makedirs(save_dir, exist_ok=True)
    crawlers = [
        BingImageCrawler(storage={'root_dir': save_dir}, downloader_threads=4, parser_threads=2),
        GoogleImageCrawler(storage={'root_dir': save_dir}, downloader_threads=4, parser_threads=2)
    ]
    count = 0
    for crawler in crawlers:
        try:
            crawler.crawl(keyword=query, max_num=limit - count, file_idx_offset=count)
            count = len([f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))])
            if count >= limit:
                break
        except Exception as e:
            print(f"⚠️ Error in {crawler.__class__.__name__}: {e}")
    print(f"📦 Collected {count}/{limit} images for '{query}'")

def clean_and_resize_images(folder, size=(256, 256)):
    seen_hashes = set()
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in seen_hashes:
                os.remove(file_path)
                continue
            seen_hashes.add(file_hash)
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                img = img.resize(size, Image.LANCZOS)
                img.save(file_path, "JPEG", quality=90)
        except Exception as e:
            os.remove(file_path)
            print(f"⚠️ Removed bad file {file_path}: {e}")

def split_dataset_classification(class_dir, output_dir, split_ratios=(0.7, 0.2, 0.1)):
    files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    random.shuffle(files)
    n = len(files)
    n_train = int(n * split_ratios[0])
    n_test = int(n * split_ratios[1])
    splits = {
        "train": files[:n_train],
        "test": files[n_train:n_train+n_test],
        "val": files[n_train+n_test:]
    }
    for split, split_files in splits.items():
        split_dir = os.path.join(output_dir, split, os.path.basename(class_dir))
        os.makedirs(split_dir, exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(split_dir, f))

def split_dataset_detection(class_dir, output_dir, class_id=0, split_ratios=(0.8, 0.2)):
    files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    random.shuffle(files)
    n = len(files)
    n_train = int(n * split_ratios[0])
    splits = {
        "train": files[:n_train],
        "val": files[n_train:]
    }
    for split, split_files in splits.items():
        img_dir = os.path.join(output_dir, "images", split)
        label_dir = os.path.join(output_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(img_dir, f))
            label_path = os.path.splitext(f)[0] + ".txt"
            with open(os.path.join(label_dir, label_path), "w") as lf:
                lf.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# =============================
# Safe Folder Deletion (Windows)
# =============================
def remove_readonly(func, path, excinfo):
    # Make file writable and retry
    os.chmod(path, stat.S_IWRITE)
    func(path)

def safe_delete(folder):
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder, onerror=remove_readonly)
        except PermissionError as e:
            print(f"⚠️ Could not delete {folder}: {e}")

# =============================
# Build Dataset
# =============================
def build_dataset(user_query, num_images=100, mode="classification", class_id=0):
    processed_query = preprocess_query(user_query)
    save_dir = os.path.join(dataset_dir, processed_query.replace(" ", "_"))
    download_images(processed_query, save_dir, limit=num_images)
    clean_and_resize_images(save_dir, size=IMG_SIZE)

    if mode == "classification":
        split_dataset_classification(save_dir, cnn_dataset_dir, split_ratios=SPLIT_RATIOS)
    elif mode == "detection":
        split_dataset_detection(save_dir, yolo_dataset_dir, class_id=class_id)

# =============================
# Flask App
# =============================
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        mode = request.form["mode"]
        queries = request.form["queries"]
        num_images = int(request.form["num_images"])
        
        # Clean old datasets before making new one
        for folder in [dataset_dir, cnn_dataset_dir, yolo_dataset_dir]:
            safe_delete(folder)

        os.makedirs(dataset_dir, exist_ok=True)

        query_list = [q.strip() for q in queries.split(",")]
        for idx, q in enumerate(query_list):
            build_dataset(q, num_images=num_images, mode=mode, class_id=idx)

        zip_name = "_".join([preprocess_query(q).replace(" ", "_") for q in query_list])
        if mode == "classification":
            shutil.make_archive(zip_name, 'zip', cnn_dataset_dir)
        else:
            shutil.make_archive(zip_name, 'zip', yolo_dataset_dir)

        return send_file(f"{zip_name}.zip", as_attachment=True)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
