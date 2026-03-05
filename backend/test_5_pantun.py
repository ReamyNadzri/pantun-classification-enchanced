"""
Test 5 pantun across 3 models: TextCNN, MalayBERT, SVM 90-10
"""
import requests
import json

API_URL = "http://localhost:5000/api/classify"

PANTUN_SAMPLES = [
    {
        "label": "1. PERCINTAAN",
        "pantun": "Bunga melur di tepi paya, mekar indah sepanjang hari. Hati rindu tidak tersaya, jauh di mata dekat di hati."
    },
    {
        "label": "2. NASIHAT DAN PENDIDIKAN",
        "pantun": "Buah manggis di dalam peti, bawa ke pasar dijual orang. Melentur buluh biarlah dari muda, melentur sudah tidak payang."
    },
    {
        "label": "3. ADAT DAN RESAM",
        "pantun": "Limau purut di tepi perigi, batang pepaya di tepi kali. Adat lembaga janganlah rugi, biar mati adat jangan sekali."
    },
    {
        "label": "4. AGAMA DAN KEPERCAYAAN",
        "pantun": "Padi masak beras melimpah, tuaian menjadi banyak. Sembahyang jangan dilupakan, tiang agama itulah tegak."
    },
    {
        "label": "5. JENAKA DAN PERMAINAN",
        "pantun": "Pergi ke pasar beli kelapa, kelapa dibuat santan lemak. Orang tua bergelak gelak, makan durian sampai muntah."
    },
]

MODELS = [
    ("TextCNN",       "textcnn"),
    ("MalayBERT",     "malaybert"),
    ("SVM 90/10",     "svm_90-10_no_pembayang"),
]

def classify(pantun_text, model_key):
    try:
        resp = requests.post(API_URL, json={"pantun": pantun_text, "model": model_key, "top_k": 3}, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            preds = data.get("predictions", [])
            return preds
        else:
            return [{"theme": f"ERROR {resp.status_code}", "confidence": 0}]
    except Exception as e:
        return [{"theme": f"EXCEPTION: {e}", "confidence": 0}]

print("=" * 80)
print("  MALAY PANTUN CLASSIFICATION — LIVE DEMO")
print("  Testing 5 Pantun × 3 Models")
print("=" * 80)

results = []
for sample in PANTUN_SAMPLES:
    print(f"\n{sample['label']}")
    print(f"  Pantun: {sample['pantun'][:70]}...")
    row = {"label": sample["label"], "pantun": sample["pantun"], "models": {}}
    for model_name, model_key in MODELS:
        preds = classify(sample["pantun"], model_key)
        row["models"][model_name] = preds
        top = preds[0] if preds else {}
        print(f"  [{model_name:15s}] → {top.get('theme','?'):45s} ({top.get('confidence', 0):.1f}%)")
    results.append(row)

print("\n" + "=" * 80)
print("  FULL RESULTS (JSON)")
print("=" * 80)
print(json.dumps(results, ensure_ascii=False, indent=2))
