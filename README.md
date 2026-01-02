
# malayalam-news-dataset-crawler
A fast, parallel web crawler for collecting and building labeled Malayalam news and fact-check datasets for NLP and machine learning research.

📰 Malayalam News Dataset Crawler (Fast)

A high-performance, multi-threaded web crawler for collecting Malayalam real news and fake/fact-check articles and exporting them into a clean, ML-ready CSV dataset.

Designed specifically for:

🧠 Fake News Detection (Malayalam)

📊 NLP & Data Science projects

🤖 Machine Learning / Deep Learning pipelines

🚀 Features

⚡ Fast & Parallel Crawling using ThreadPoolExecutor

📰 Collects Real News from trusted Malayalam news portals

❌ Collects Fake / Fact-Check News from verification sites

🧹 Advanced Text Cleaning

URL removal

Emoji removal

Unicode normalization (Malayalam-safe)

🧠 Duplicate Detection using SHA-256 hashing

🤖 Synthetic Fake News Generation (auto-balancing)

📄 Outputs:

CSV dataset

Manifest JSON (metadata & counts)

🤝 robots.txt respected by default

🧵 Background CSV writer with checkpointing

🧪 CLI configurable (targets, workers, crawl depth)


malayalam-news-dataset-crawler/
│
├── crawler.py                  # Main crawler script
├── outputs/
│   ├── malayalam_dataset_fast.csv
│   ├── malayalam_dataset_fast.manifest.json
│
├── README.md
├── requirements.txt
└── .gitignore

| Column Name      | Description                  |
| ---------------- | ---------------------------- |
| `id`             | Unique record ID             |
| `headline`       | Article headline             |
| `body`           | Full cleaned article content |
| `summary`        | Short summary / correction   |
| `source`         | News source domain           |
| `url`            | Original article URL         |
| `published_date` | ISO-8601 timestamp           |
| `label`          | `1 = Real`, `0 = Fake`       |
| `synthetic`      | `true/false`                 |


🌐 Sources Used
✅ Real News

manoramaonline.com

mathrubhumi.com

asianetnews.com

indianexpress.com (Malayalam)

mediaoneonline.com

news18 Malayalam

twentyfournews.com

❌ Fake / Fact-Check

altnews.in

boomlive.in

factly.in

mathrubhumi fact-check

news18 fake-news tag


⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/malayalam-news-dataset-crawler.git
cd malayalam-news-dataset-crawler

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt


📦 requirements.txt

Create a file called requirements.txt:

requests
beautifulsoup4
lxml
python-dateutil
tqdm

▶️ Usage
Basic Run
python crawler.py

Custom Configuration
python crawler.py \
  --out outputs/malayalam.csv \
  --target-real 1500 \
  --target-fake 1500 \
  --workers 30 \
  --crawl-depth 2


| Argument              | Description                  |
| --------------------- | ---------------------------- |
| `--out`               | Output CSV path              |
| `--target-real`       | Number of real news articles |
| `--target-fake`       | Number of fake articles      |
| `--workers`           | Parallel threads             |
| `--crawl-depth`       | BFS depth                    |
| `--max-per-site`      | Limit per domain             |
| `--save-raw`          | Save raw HTML                |
| `--no-respect-robots` | Disable robots.txt           |

📑 Output Files

malayalam_dataset_fast.csv → ML-ready dataset

malayalam_dataset_fast.manifest.json → crawl metadata
⚠️ Legal & Ethical Notice

This crawler respects robots.txt by default

Intended for research and educational use only

Do not use collected data for commercial redistribution

Follow each website’s terms of service

🧩 Future Improvements

 Social media crawling (Telegram, Twitter)

 Language detection validation

 Named Entity Recognition (NER)

 Transformer-ready dataset formatting

 Streamlit dashboard

🧑‍💻 Author

Dijo (B.Tech AI & DS)
Focused on Malayalam NLP, Fake News Detection & AI Systems

