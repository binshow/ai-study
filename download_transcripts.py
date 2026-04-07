import json
import subprocess
import sys
import os

videos = [
    ("VuQUF1VVX40", "第0講"),
    ("TigfpYPJk1s", "第1講"),
    ("lVdajtNpaGI", "第2講"),
    ("8iFvM7WUUs8", "第3講"),
    ("dWQVY_h0YXU", "第4講"),
    ("Taj1eHmZyWw", "第5講"),
    ("mPWvAN4hzzY", "第6講"),
    ("YJoegm7kiUM", "第7講"),
    ("EnWz5XuOnIQ", "第8講"),
    ("ccqCDD9LqCA", "第9講"),
    ("CbIPjrOj2Tc", "第10講")
]

os.makedirs("GenAI_ML_2025_Notes", exist_ok=True)

cli_path = "/Users/shengbinbin/Library/Python/3.9/bin/youtube_transcript_api"

for vid, title in videos:
    try:
        # Run CLI command
        result = subprocess.run([cli_path, vid, "--languages", "zh-TW", "zh-HK", "zh-CN", "zh", "en", "--format", "json"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed {title}: {result.stderr}")
            continue
        
        data = json.loads(result.stdout)
        # The CLI returns a list for the requested videos, or a dict. 
        # Usually it returns a JSON object where keys are video IDs if multiple, or just the list if one video? Let's check format.
        # From head output, it starts with [[{"text": ...
        # It's a list of lists.
        text = " ".join([item['text'] for item in data[0]])
        with open(f"GenAI_ML_2025_Notes/{title}_transcript.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Downloaded transcript for {title}")
    except Exception as e:
        print(f"Failed {title}: {e}")
