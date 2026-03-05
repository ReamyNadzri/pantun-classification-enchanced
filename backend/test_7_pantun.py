"""
Test 7 new pantun across 3 models: TextCNN, MalayBERT, SVM 90-10
"""
import requests
import json
import time

API_URL = "http://localhost:5000/api/classify"

PANTUN_SAMPLES = [
    {
        "no": 21,
        "pantun": "Makan sirih berpinang tidak, Pinang ada di bawah tangga; Makan sirih mengenyang tidak, Kerana budi dan bahasa."
    },
    {
        "no": 22,
        "pantun": "Kain bersuji di atas atap, Sudah dikelim baru dipakai; Kalau sudi tuanku santap, Sirihnya kering pinangnya kotai."
    },
    {
        "no": 23,
        "pantun": "Ada sirih ada pinang, Nanti gambir dengan kapur; Sudah dipilih sudah dipinang, Hanya menanti ijab kabul."
    },
    {
        "no": 24,
        "pantun": "Ambil buluh di rumpun jering, Jering bersanggit dibatang pinang; Kalau jalak lawan biring, Belum tentu kalah menang."
    },
    {
        "no": 25,
        "pantun": "Apa guna pokok pinang, Daun hijau kuning padi; Apa diingat apa dikenang, Gunung dikejar takkan lari."
    },
    {
        "no": 26,
        "pantun": "Bangun tidur lalu berenang, Naik rakit batang pinang; Habis tali sambung benang, Kumbang terbang bukan senang."
    },
    {
        "no": 27,
        "pantun": "Hujan reda ribut berdengung, Tanam pinang di kaki tangga; Jangan dinda duduk termenung, Sapu tangan pengganti kanda."
    }
]

MODELS = [
    ("TextCNN",   "textcnn"),
    ("MalayBERT", "malaybert"),
    ("SVM 90/10", "svm_90-10_no_pembayang"),
]

def classify(pantun_text, model_key):
    try:
        resp = requests.post(API_URL, json={"pantun": pantun_text, "model": model_key, "top_k": 3}, timeout=90)
        if resp.status_code == 200:
            return resp.json().get("predictions", [])
        else:
            return [{"theme": f"ERROR {resp.status_code}", "confidence": 0}]
    except Exception as e:
        return [{"theme": f"EXCEPTION: {str(e)[:30]}", "confidence": 0}]

results = []
print("Running...")
for sample in PANTUN_SAMPLES:
    row = {"no": sample["no"], "pantun": sample["pantun"], "models": {}}
    for model_name, model_key in MODELS:
        preds = classify(sample["pantun"], model_key)
        row["models"][model_name] = preds
    results.append(row)
    print(f"  Done #{sample['no']}")

print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
