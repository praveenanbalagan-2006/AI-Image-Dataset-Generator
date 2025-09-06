Hey there! 👋 This project helps you create high-quality image datasets for your AI or computer vision experiments—without manually searching and downloading hundreds of images.

What makes it special? It uses NLP magic to understand your search queries, expand them, and fetch images more intelligently.

🌟 What It Can Do

Smarter Search Queries: Automatically expands your keywords using NLP so you don’t miss any relevant images.

Clean & Organized: Filters out irrelevant words, categorizes images neatly into train/val/test folders.

Flexible Categories: Works for anything—face shapes, hairstyles, objects, or whatever you want to classify.

Ready-to-Use Dataset: Images are downloaded, preprocessed, and ready for your AI models.

🛠️ Tools & Libraries Behind the Scenes

NLP Stuff

NLTK – Tokenizes your queries, removes stopwords, and helps clean text.

spaCy – Optional, but great for understanding categories or named entities.

TextBlob – For simple word expansions or corrections.

Image Stuff

icrawler – Automatically fetches images from Google, Bing, etc.

Pillow (PIL) – Resizes, converts, and prepares images.

BeautifulSoup4 – Parses metadata or captions if needed.

Helpers

hashlib – Avoid duplicates by hashing images.

os / shutil / pathlib – Keeps your dataset structured and tidy.
