"""
Malay Pantun NLP Preprocessing Pipeline
Following Chapter 3 methodology:
1. Text Segmentation (pembayang/maksud separation)
2. Case Folding
3. Tokenization
4. Stopword Removal
5. Stemming
6. Special Character Removal
"""

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Initialize Malay stemmer (PySastrawi)
_stemmer_factory = StemmerFactory()
stemmer = _stemmer_factory.create_stemmer()

# Malay stopwords (comprehensive list)
MALAY_STOPWORDS = {
    "ada", "adalah", "adanya", "adapun", "agak", "agaknya", "agar", "akan",
    "akankah", "akhir", "akhirnya", "aku", "akulah", "amat", "amatlah",
    "anda", "andalah", "antar", "antara", "antaranya", "apa", "apabila",
    "apakah", "apalagi", "apatah", "atau", "ataukah", "ataupun", "bagai",
    "bagaimanakah", "bagaimanapun", "bagi", "bagian", "bagimu", "baginda",
    "bahawa", "bahawasanya", "bahkan", "bahwa", "bahwasanya", "baik",
    "bakal", "bakalan", "balik", "banyak", "banyaknya", "bapak", "baru",
    "bawah", "beberapa", "begini", "beginian", "beginikah", "beginilah",
    "begitu", "begitukah", "begitulah", "begitupun", "bekas", "belakang",
    "beliau", "belum", "belumkah", "belumlah", "benar", "benarkah",
    "benarlah", "berada", "berakhir", "berakhirlah", "berakhirnya",
    "berapa", "berapakah", "berapalah", "berapapun", "berarti", "berawal",
    "berbagai", "berdatangan", "beri", "berikan", "berikut", "berikutnya",
    "berjumlah", "berkali", "berkata", "berkehendak", "berkeinginan",
    "berkenaan", "berlainan", "berlalu", "berlangsung", "berlebihan",
    "bermacam", "bermaksud", "bermula", "bersama", "bersiap", "bertanya",
    "berturut", "bertutur", "berupa", "besar", "betul", "betulkah",
    "biasa", "biasanya", "bila", "bilakah", "bilamana", "bisa", "bisakah",
    "boleh", "bolehkah", "bolehlah", "buat", "bukan", "bukankah",
    "bukanlah", "bukannya", "bulan", "bung", "cara", "caranya", "cuma",
    "cukup", "cukupkah", "cukuplah", "dahulu", "dalam", "dan", "dapat",
    "dari", "daripada", "datang", "dekat", "demi", "demikian",
    "demikianlah", "dengan", "depan", "di", "dia", "diakhiri",
    "diakhirinya", "dialah", "diantara", "diantaranya", "diberi",
    "diberikan", "diberikannya", "dibuat", "dibuatnya", "didapat",
    "didapati", "digunakan", "diibaratkan", "dijawab", "dijawabnya",
    "dikarenakan", "dikatakan", "dikatakannya", "dikerjakan", "dilakukan",
    "diluar", "dimaksud", "dimaksudkan", "dimaksudkannya", "diminta",
    "dimulai", "dimulailah", "dimulainya", "dini", "diperbuat",
    "diperbuatnya", "dipergunakan", "diperkirakan", "diperlukan",
    "dipersoalkan", "dipertanyakan", "dipunyai", "diri", "dirinya",
    "ditambahkan", "ditandaskan", "ditanya", "ditanyai", "ditanyakan",
    "ditentukan", "dituturkan", "dituturkannya", "diucapkan",
    "diucapkannya", "diungkapkan", "dong", "dua", "dulu", "empat",
    "engkau", "engkaulah", "enggak", "enggaknya", "entah", "entahlah",
    "guna", "gunakan", "hal", "hampir", "hanya", "hanyalah", "hari",
    "harus", "haruskah", "haruslah", "hendak", "hendaklah", "hendaknya",
    "hingga", "ia", "ialah", "ibarat", "ibaratkan", "ibaratnya", "ibu",
    "ikut", "ingat", "ini", "inikah", "inilah", "itu", "itukah",
    "itulah", "jadi", "jadilah", "jadinya", "jangan", "jangankan",
    "janganlah", "jauh", "jawab", "jawaban", "jawabnya", "jelas",
    "jelasnya", "jika", "jikalau", "juga", "jumlah", "jumlahnya",
    "justru", "kala", "kalau", "kalaulah", "kalaupun", "kali", "kalian",
    "kami", "kamilah", "kamu", "kamulah", "kan", "kapan", "kapankah",
    "kapanpun", "karena", "karenanya", "kasus", "kata", "katakan",
    "katakanlah", "katanya", "ke", "keadaan", "kebetulan", "kecil",
    "kedua", "keduanya", "keinginan", "kelamaan", "kelihatan",
    "kelihatannya", "kelima", "keluar", "kembali", "kemudian", "kemungkinan",
    "kemungkinannya", "kenapa", "kepada", "kepadanya", "kesampaian",
    "keseluruhan", "keseluruhannya", "keterlaluan", "ketika", "khususnya",
    "kini", "kinilah", "kira", "kiranya", "kita", "kitalah", "kok",
    "kurang", "lagi", "lagian", "lah", "lain", "lainnya", "lalu",
    "lama", "lamanya", "langsung", "lanjut", "lanjutnya", "lebih",
    "luar", "macam", "mahu", "maka", "makanya", "makin", "malah",
    "malahan", "mampu", "mampukah", "mana", "manakala", "manalagi",
    "masa", "masalah", "masalahnya", "masih", "masihkah", "masing",
    "mau", "maupun", "melainkan", "melakukan", "melalui", "melihat",
    "melihatnya", "memang", "memastikan", "memberi", "memberikan",
    "membuat", "memerlukan", "memihak", "memiliki", "meminta",
    "memintakan", "memisalkan", "memperbuat", "mempergunakan",
    "memperkirakan", "memperlihatkan", "mempersiapkan", "mempersoalkan",
    "mempertanyakan", "mempunyai", "memulai", "memungkinkan", "menaiki",
    "menambahkan", "menandaskan", "menanti", "menantikan", "menanya",
    "menanyai", "menanyakan", "mendapat", "mendapati", "mendapatkan",
    "mengatakan", "mengakhiri", "mengapa", "mengenai", "mengerjakan",
    "menggambarkan", "menggunakan", "menghendaki", "mengibaratkan",
    "mengibaratkannya", "mengingat", "mengingatkan", "menginginkan",
    "mengira", "mengucapkan", "mengucapkannya", "mengungkapkan",
    "menjadi", "menjawab", "menjelaskan", "menuju", "menunjuk",
    "menunjuki", "menunjukkan", "menunjuknya", "menurut", "menuturkan",
    "menyampaikan", "menyangkut", "menyatakan", "menyebutkan",
    "menyeluruh", "menyiapkan", "merasa", "mereka", "merekalah",
    "merupakan", "meski", "meskipun", "minta", "mirip", "misal",
    "misalkan", "misalnya", "mula", "mulai", "mulailah", "mulanya",
    "mungkin", "mungkinkah", "nah", "naik", "namun", "nanti", "nantinya",
    "nyaris", "nyatanya", "oleh", "olehnya", "pada", "padahal",
    "padanya", "paling", "panjang", "pantas", "para", "pasti",
    "pastilah", "penting", "pentingnya", "per", "percuma", "perlu",
    "perlukah", "perlunya", "pernah", "pernahkah", "pertama",
    "pertanyaan", "pertanyakan", "pihak", "pihaknya", "pukul", "pula",
    "pun", "punya", "rasa", "rasanya", "rata", "rupanya", "saat",
    "saatnya", "saja", "sajalah", "saling", "sama", "sambil", "sampai",
    "sana", "sangat", "sangatlah", "satu", "saya", "sayalah", "se",
    "sebab", "sebabnya", "sebagai", "sebagaimana", "sebagainya",
    "sebagian", "sebaik", "sebaiknya", "sebaliknya", "sebanyak",
    "sebegini", "sebegitu", "sebelum", "sebelumnya", "sebenarnya",
    "seberapa", "sebesar", "sebetulnya", "sebisanya", "sebuah", "sebut",
    "sebutlah", "sebutnya", "secara", "secukupnya", "sedang", "sedangkan",
    "sedemikian", "sedikit", "sedikitnya", "seenaknya", "segala",
    "segalanya", "segera", "seharusnya", "sehingga", "seingat",
    "sejak", "sejauh", "sejenak", "sejumlah", "sekadar", "sekadarnya",
    "sekali", "sekalian", "sekaligus", "sekalipun", "sekarang",
    "sekecil", "seketika", "sekiranya", "sekitar", "sekitarnya",
    "sekurang", "selain", "selaku", "selalu", "selama", "selanjutnya",
    "seluruh", "seluruhnya", "semacam", "semakin", "semampu", "semasa",
    "semasih", "semata", "sementara", "semisal", "semoga", "sempat",
    "semua", "semuanya", "semula", "sendiri", "sendirinya", "seolah",
    "seorang", "sepanjang", "sepantasnya", "sepatutnya", "seperti",
    "sepertinya", "sepihak", "sering", "seringnya", "serta", "serupa",
    "sesaat", "sesama", "sesampai", "sesegera", "sesekali", "seseorang",
    "sesuatu", "sesuatunya", "sesudah", "sesudahnya", "setelah",
    "setempat", "setengah", "seterusnya", "setiap", "setiba",
    "setidaknya", "setinggi", "seusai", "sewaktu", "siapa", "siapakah",
    "sini", "sinilah", "soal", "soalnya", "suatu", "sudah", "sudahkah",
    "sudahlah", "supaya", "tadi", "tadinya", "tahu", "tahun", "tak",
    "tambah", "tambahnya", "tampak", "tampaknya", "tandas", "tandasnya",
    "tanpa", "tanya", "tanyakan", "tanyanya", "tapi", "tentu",
    "tentulah", "tentunya", "tepat", "terakhir", "terasa", "terbanyak",
    "terdahulu", "terdapat", "terdiri", "terhadap", "terhadapnya",
    "teringat", "terjadi", "terjadilah", "terjadinya", "terkira",
    "terlalu", "terlebih", "terlihat", "termasuk", "ternyata", "tersampaikan",
    "tersebut", "tertentu", "tertuju", "terus", "terutama", "tetap",
    "tetapi", "tiada", "tiadanya", "tidak", "tidakkah", "tidaklah",
    "tiga", "tinggi", "toh", "tuju", "tunjuk", "turut", "tutur",
    "tuturnya", "ucap", "ucapnya", "ujar", "ujarnya", "umum", "umumnya",
    "ungkap", "ungkapnya", "untuk", "usah", "usai", "waduh", "wah",
    "wahai", "waktu", "walaupun", "wong", "yakni", "yaitu", "yang",
    # Additional Malay-specific stopwords
    "nye", "lah", "kah", "tah", "pun", "sahaja", "hanya",
    "ialah", "iaitu", "mahupun", "ataupun", "serta", "tetapi",
    "walau", "bagaimanapun", "namun", "sebaliknya", "oleh",
    "kerana", "sebab", "hingga", "sehingga", "supaya",
    "telah", "sedang", "akan", "masih", "belum", "sudah",
    "yang", "ini", "itu", "dan", "di", "ke", "dari",
}


def segment_pantun(text, use_pembayang=False):
    """
    Segment pantun into pembayang (lines 1-2) and maksud (lines 3-4).
    
    Args:
        text: Pantun text with lines separated by ';' or ','
        use_pembayang: If True, return all 4 lines. If False, return only maksud (lines 3-4).
    
    Returns:
        Segmented text for classification.
    """
    # Split by semicolon or comma
    lines = re.split(r'[;,]', text)
    lines = [line.strip() for line in lines if line.strip()]
    
    if len(lines) < 4:
        # If less than 4 lines, use all available text
        return ' '.join(lines)
    
    if use_pembayang:
        # Use all 4 lines
        return ' '.join(lines[:4])
    else:
        # Use only maksud (lines 3 and 4) - theme meaning resides here
        return ' '.join(lines[2:4])


def case_fold(text):
    """Convert text to lowercase."""
    return text.lower()


def tokenize(text):
    """Tokenize text into words."""
    # Remove special characters and numbers, keep only Malay letters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Split into tokens
    tokens = text.split()
    # Remove empty tokens
    return [t for t in tokens if t.strip()]


def remove_stopwords(tokens):
    """Remove Malay stopwords from token list."""
    return [t for t in tokens if t not in MALAY_STOPWORDS]


def stem_tokens(tokens):
    """Stem tokens to root words using Malay stemmer."""
    return [stemmer.stem(t) for t in tokens]


def preprocess_text(text, use_pembayang=False):
    """
    Full preprocessing pipeline for a single pantun text.
    
    Returns:
        preprocessed_text: The cleaned, processed text as a string
        steps: Dictionary of intermediate results for visualization
    """
    steps = {}
    
    # Step 1: Text Segmentation
    segmented = segment_pantun(text, use_pembayang)
    steps['segmentation'] = segmented
    
    # Step 2: Case Folding
    folded = case_fold(segmented)
    steps['case_folding'] = folded
    
    # Step 3: Tokenization
    tokens = tokenize(folded)
    steps['tokenization'] = tokens
    
    # Step 4: Stopword Removal
    no_stopwords = remove_stopwords(tokens)
    steps['stopword_removal'] = no_stopwords
    
    # Step 5: Stemming
    stemmed = stem_tokens(no_stopwords)
    steps['stemming'] = stemmed
    
    # Final: Join back to string
    result = ' '.join(stemmed)
    steps['final'] = result
    
    return result, steps


def preprocess_batch(texts, use_pembayang=False, return_steps=False):
    """
    Preprocess a batch of pantun texts.
    
    Args:
        texts: List of pantun text strings
        use_pembayang: Whether to include pembayang lines
        return_steps: Whether to return preprocessing steps
    
    Returns:
        List of preprocessed texts (and optionally steps)
    """
    results = []
    all_steps = []
    
    for text in texts:
        processed, steps = preprocess_text(text, use_pembayang)
        results.append(processed)
        if return_steps:
            all_steps.append(steps)
    
    if return_steps:
        return results, all_steps
    return results


if __name__ == "__main__":
    # Test the preprocessing pipeline
    test_pantun = "Ikan berenang di dalam lubuk; Ikan belida dadanya panjang; Adat pinang pulang ke tampuk; Adat sirih pulang ke gagang"
    
    print("=" * 60)
    print("PREPROCESSING PIPELINE TEST")
    print("=" * 60)
    print(f"\nOriginal: {test_pantun}")
    
    print("\n--- Without Pembayang (maksud only) ---")
    result, steps = preprocess_text(test_pantun, use_pembayang=False)
    for step_name, value in steps.items():
        print(f"  {step_name}: {value}")
    
    print("\n--- With Pembayang (all 4 lines) ---")
    result, steps = preprocess_text(test_pantun, use_pembayang=True)
    for step_name, value in steps.items():
        print(f"  {step_name}: {value}")
