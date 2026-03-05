"""
Test 20 user-composed pantun across 3 models: TextCNN, MalayBERT, SVM 90-10
"""
import requests
import json

API_URL = "http://localhost:5000/api/classify"

PANTUN_SAMPLES = [
    {
        "no": 1,
        "expected": "Percintaan",
        "pantun": "Pulau pandan jauh ke tengah, Gunung Daik bercabang tiga, Hancur badan dikandung tanah, Budi yang baik dikenang juga."
    },
    {
        "no": 2,
        "expected": "Percintaan",
        "pantun": "Kalau ada sumur di ladang, Bolehlah kita menumpang mandi, Kalau ada umur yang panjang, Bolehlah kita berjumpa lagi."
    },
    {
        "no": 3,
        "expected": "Persahabatan",
        "pantun": "Berburu ke padang datar, Dapat rusa berbelang kaki, Berguru kepalang ajar, Bagai bunga kembang tak jadi."
    },
    {
        "no": 4,
        "expected": "Pendidikan",
        "pantun": "Anak ayam turun sepuluh, Mati seekor tinggal sembilan, Tuntutlah ilmu bersungguh-sungguh, Supaya hidup penuh kemuliaan."
    },
    {
        "no": 5,
        "expected": "Nasihat",
        "pantun": "Tinggi-tinggi pohon kelapa, Nampak dari tepi perigi, Baik-baik membawa diri, Supaya selamat hidup negeri."
    },
    {
        "no": 6,
        "expected": "Nasihat",
        "pantun": "Buah cempedak di luar pagar, Ambil galah tolong jolokkan, Saya budak baru belajar, Kalau salah tolong tunjukkan."
    },
    {
        "no": 7,
        "expected": "Adab",
        "pantun": "Kalau memancing ikan tenggiri, Jangan lupa membawa jala, Kalau bercakap sesama negeri, Bahasa baik hiasan diri."
    },
    {
        "no": 8,
        "expected": "Adab",
        "pantun": "Pergi ke hutan mencari rotan, Rotan tua dibuat titi, Jika hidup penuh kesopanan, Orang memandang penuh hormati."
    },
    {
        "no": 9,
        "expected": "Pendidikan",
        "pantun": "Pergi ke sawah menanam padi, Padi masak kuning warnanya, Jika rajin mencari budi, Ilmu banyak manfaatnya."
    },
    {
        "no": 10,
        "expected": "Pendidikan",
        "pantun": "Kalau tuan pergi ke kedai, Belikan saya kain pelikat, Kalau tuan pandai pandai, Ilmu dicari janganlah malas."
    },
    {
        "no": 11,
        "expected": "Alam",
        "pantun": "Anak ikan dimakan ikan, Ikan besar di dalam perahu, Alam Tuhan penuh keindahan, Tempat manusia mencari ilmu."
    },
    {
        "no": 12,
        "expected": "Alam",
        "pantun": "Burung merpati terbang ke hulu, Hinggap sebentar di dahan sena, Hutan rimba jangan diganggu, Tempat makhluk hidup bersama."
    },
    {
        "no": 13,
        "expected": "Keagamaan",
        "pantun": "Pergi ke masjid waktu pagi, Membawa sejadah di tangan, Jika hidup mahu diberkati, Jangan lupa pada Tuhan."
    },
    {
        "no": 14,
        "expected": "Keagamaan",
        "pantun": "Anak dagang pergi ke seberang, Membawa bekal seikat padi, Jika hidup hendaklah tenang, Taat kepada Ilahi Rabbi."
    },
    {
        "no": 15,
        "expected": "Kasih ibu bapa",
        "pantun": "Air jernih di dalam telaga, Tempat rusa datang minum, Kasih ibu sepanjang masa, Kasih ayah sepanjang umur."
    },
    {
        "no": 16,
        "expected": "Kasih ibu bapa",
        "pantun": "Anak ayam turun ke laman, Mematuk padi di tepi pagar, Jasa ayah bunda jangan dilupakan, Kerana mereka pembawa sabar."
    },
    {
        "no": 17,
        "expected": "Kehidupan",
        "pantun": "Pisang emas dibawa belayar, Masak sebiji di atas peti, Hutang emas boleh dibayar, Hutang budi dibawa mati."
    },
    {
        "no": 18,
        "expected": "Kehidupan",
        "pantun": "Pergi ke kebun memetik kelapa, Kelapa muda airnya segar, Jika hidup banyak berjasa, Nama dikenang sepanjang zaman."
    },
    {
        "no": 19,
        "expected": "Budi bahasa",
        "pantun": "Kalau pergi ke Tanjung Keramat, Singgah sebentar membeli bunga, Jika hidup hendak selamat, Budi bahasa jangan dilupa."
    },
    {
        "no": 20,
        "expected": "Budi",
        "pantun": "Anak merbah di pohon cemara, Terbang rendah mencari makan, Orang berbudi kita berbahasa, Orang memberi kita merasa."
    },
]

MODELS = [
    ("TextCNN",   "textcnn"),
    ("MalayBERT", "malaybert"),
    ("SVM 90/10", "svm_90-10_no_pembayang"),
]

# Map user expected themes to nearest system category
THEME_MAP = {
    "Percintaan":     "PANTUN PERCINTAAN",
    "Persahabatan":   "PANTUN KEMBARA DAN PERANTAUAN",  # closest
    "Pendidikan":     "PANTUN NASIHAT DAN PENDIDIKAN",
    "Nasihat":        "PANTUN NASIHAT DAN PENDIDIKAN",
    "Adab":           "PANTUN BUDI",
    "Alam":           "PANTUN KIAS DAN IBARAT",  # closest
    "Keagamaan":      "PANTUN AGAMA DAN KEPERCAYAAN",
    "Kasih ibu bapa": "PANTUN BUDI",
    "Kehidupan":      "PANTUN PERIBAHASA DAN PERBILANGAN",  # closest
    "Budi bahasa":    "PANTUN BUDI",
    "Budi":           "PANTUN BUDI",
}

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
    row = {"no": sample["no"], "expected": sample["expected"], "pantun": sample["pantun"][:60]+"...", "models": {}}
    for model_name, model_key in MODELS:
        preds = classify(sample["pantun"], model_key)
        row["models"][model_name] = preds
    results.append(row)
    print(f"  Done #{sample['no']}")

print(json.dumps({"theme_map": THEME_MAP, "results": results}, ensure_ascii=False, indent=2))
