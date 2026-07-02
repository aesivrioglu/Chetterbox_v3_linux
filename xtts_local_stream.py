"""
============================================================================
XTTS v2 (Coqui) — YEREL (Linux + RTX 3090 / CUDA) STREAMING TTFB scripti
============================================================================

AMAÇ
  chatterbox_local_stream.py ve elevenlabs_ttfb.py ile BİREBİR AYNI kıyas
  mantığı; tek fark model: burada aday TTS = Coqui XTTS v2. Amaç yine ElevenLabs
  (baz çıta ~182-188 ms) TTFB'sinin ALTINA inen, şirket localinde ağsız çalışan
  bir Türkçe TTS bulmak. Bu script arayüzsüz (headless) bir Linux makinede,
  yerel RTX 3090 (Ampere) üzerinde çalışır ve GitHub'dan çekilip tek komutla
  koşacak şekilde OLABİLDİĞİNCE EKSİKSİZ + savunmacı yazılmıştır.

NEDEN XTTS v2 CHATTERBOX'TAN HIZLI OLMALI (senin tahminini doğruluyor)
  Chatterbox'ın yüksek TTFB'sinin iki yapısal sebebi vardı:
    (a) Zero-shot referans, her ses parçasında CFM/vocoder içinde BAŞTAN işlenip
        atılıyordu (10 sn prompt -> her chunk'ta ~510 mel frame yeniden üretim).
    (b) Token'lar tek tek, yüksek Python/launch overhead'iyle işleniyordu.
  XTTS v2 (a)'yı YAPISAL olarak çözer: referans SES BİR KEZ işlenip
  `gpt_cond_latent` + `speaker_embedding`'e dönüştürülür (get_conditioning_latents)
  ve her istekte SADECE bu hazır latent'ler kullanılır — referans TEKRAR
  işlenMEZ. Ayrıca `inference_stream` NATIVE streaming'dir: GPT token üretir
  üretmez `stream_chunk_size` token biriktiğinde HiFi-GAN ile decode edip parça
  yield eder. Yani "ilk ses"e kadar geçen süre kısadır.

ADİL KIYAS (üç script de AYNI birim)
  - Aynı Türkçe cümle (BENCH_TEXT) ve aynı konuşma (CONVERSATION).
  - AYNI BİRİM: "ilk FIRST_AUDIO_MS=200 ms sesin dinleyicinin eline geçmesi".
    ElevenLabs baytları 200 ms'ye çevirip damgalıyordu; burada da üretilen ses
    ÖRNEKLERİNİ biriktirip 200 ms dolunca damgalıyoruz -> elmayla elma.
  - Aynı RUNS=10, WARMUP=2.
  - Aynı referans ses dosyası (sayfa-33.wav) -> ses klonlama da aynı girdiden.

⚠️ LİSANS (üretim kararı için ÖNEMLİ — README-chatterbox.md'de de var)
  XTTS v2 "Coqui Public Model License (CPML)" ile gelir ve TİCARİ KULLANIMA
  KAPALIDIR. Bu script yalnızca KIYAS/POC içindir. ElevenLabs YERİNE ticari
  üretimde kullanmak istiyorsan XTTS v2 uygun DEĞİLDİR; Chatterbox (MIT) ticari
  kullanıma açıktır. Karşılaştırmayı bilerek yapıyoruz ama bu kısıtı unutma.
  Aşağıdaki COQUI_TOS_AGREED=1, model indirmesi için CPML'i kabul eder.

----------------------------------------------------------------------------
KURULUM (Linux, RTX 3090 — AYRI venv; chatterbox ile ÇAKIŞIR!)
----------------------------------------------------------------------------
  # XTTS'in torch/transformers pin'leri chatterbox'ınkinden FARKLIDIR ->
  # MUTLAKA ayrı bir sanal ortam kullan:
  python3.11 -m venv .venv-xtts && source .venv-xtts/bin/activate
  pip install -U pip
  pip install -r requirements-xtts.txt      # coqui-tts kendi torch'unu çözer
  # RTX 3090 (Ampere) CUDA wheel'i otomatik gelir; CPU wheel gelirse:
  #   pip install --force-reinstall torch torchaudio --index-url \
  #       https://download.pytorch.org/whl/cu124

ÇALIŞTIRMA
  # Referans sesi (Türkçe, 6-10 sn temiz) makineye koy; env ile de verilebilir:
  REF_WAV=/yol/sayfa-33.wav OUT_DIR=out python xtts_local_stream.py
  # İlk çalıştırma XTTS v2 modelini (~1.8 GB) indirir; sonra cache'ten gelir.
  # GPU kontrolü:  python -c "import torch; print(torch.cuda.get_device_name(0))"
----------------------------------------------------------------------------
"""
import os

# Model indirmesi headless'ta interaktif lisans sorusu SORAMAZ -> CPML'i baştan
# kabul et (yukarıdaki lisans notunu okudun). Kullanıcı zaten ayarladıysa bozma.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import statistics
import time

import numpy as np
import torch
import torchaudio as ta

# ===========================================================================
# 0. AYARLAR  (chatterbox_local_stream.py / elevenlabs_ttfb.py ile AYNI — kıyas için)
# ===========================================================================
LANG    = "tr"
_HERE   = os.path.dirname(os.path.abspath(__file__))
REF_WAV = os.getenv("REF_WAV", os.path.join(_HERE, "sayfa-33.wav"))
OUT_DIR = os.getenv("OUT_DIR", _HERE)
os.makedirs(OUT_DIR, exist_ok=True)

# --- ADİL KIYAS BİRİMİ: "ilk duyulabilir X ms ses" (üç script de AYNI) ---
FIRST_AUDIO_MS = 200

# Ölçüm (chatterbox / elevenlabs ile AYNI)
RUNS     = 10
WARMUP   = 2
SAVE_WAV = True
MEASURE_CER = True          # faster-whisper ile Türkçe doğruluk (kaliteyi de kıyasla)

# Canlı görüşme simülasyonu (diğer scriptlerle AYNI cümleler)
CONVERSATION = [
    "Merhaba, ben Ahmet. Ürün demosu için on dört Temmuz Pazartesi "
    "saat on beş sıfır sıfır'da size randevu oluşturdum, uygun mudur?",
    "Anladım, size uygun başka bir gün ve saat önerebilir miyim?",
    "Harika, o zaman görüşmemizi teyit ediyorum. İyi günler dilerim.",
]
BENCH_TEXT = CONVERSATION[0]   # ElevenLabs TEXT ile AYNI olmalı

# ===========================================================================
# 0.5 XTTS OPTİMİZASYON / KALİTE AYARLARI (gerekçe: README-chatterbox.md)
# ===========================================================================
# --- TTFB'yi düşüren asıl kaldıraç: stream_chunk_size ---
# GPT kaç yeni token biriktirince ilk parça decode edilip yield edilsin.
# KÜÇÜK = daha düşük TTFB ama parça başına overhead artar; ÇOK küçükte (<~10)
# HiFi-GAN parça sınırında ses bozulabilir. 20 güvenli varsayılan; TTFB'yi
# düşürmek için 10-15 dene ve out_bench.wav + CER ile kaliteyi doğrula.
STREAM_CHUNK_SIZE = 20
OVERLAP_WAV_LEN   = 1024        # parça birleştirme örtüşmesi (stok varsayılan)

# --- REFERANS SESİ OPTİMİZASYONU (BİR KEZ; TTFB'yi doğrudan etkilemez) ---
# XTTS referansı istek başına DEĞİL, başlangıçta BİR KEZ latent'e çevirir. Bu
# yüzden burada amaç kaliteyi korurken kurulum maliyetini makul tutmak. 6 sn
# temiz Türkçe referans, hızlı ve kaliteli klon için yeterlidir.
REF_GPT_COND_SEC = 6            # GPT koşullama için kullanılan referans uzunluğu (sn)
REF_MAX_SEC      = 10           # konuşmacı embedding'i için üst sınır (sn)
SOUND_NORM_REFS  = False        # referansı ses-normalize et (gürültülü kayıtta True dene)
CACHE_LATENTS    = True         # latent'leri REF_WAV yanına .pt olarak önbelleğe al (tekrar koşularda anında)

# --- GPT örnekleme (kalite; XTTS v2 varsayılanlarına yakın) ---
TEMPERATURE        = 0.7
LENGTH_PENALTY     = 1.0
REPETITION_PENALTY = 5.0
TOP_K              = 50
TOP_P              = 0.85
SPEED              = 1.0

# ===========================================================================
# 1. CİHAZ + DTYPE (RTX 3090 = Ampere -> TF32 açık, fp32 matmul'ları hızlanır)
# ===========================================================================
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    gpu_name = torch.cuda.get_device_name(0)
else:
    device = "cpu"
    gpu_name = "CPU (GPU bulunamadı — XTTS CPU'da çok yavaştır; CUDA/torch kurulumunu kontrol et!)"
print(f"[cihaz] {device} | {gpu_name} | dtype=fp32 (Ampere'de TF32 açık)")

def _sync():
    if device == "cuda":
        torch.cuda.synchronize()

if not (REF_WAV and os.path.exists(REF_WAV)):
    raise SystemExit(
        f"[hata] Referans ses yok: '{REF_WAV}'. XTTS ses klonlama için bir referans "
        "ŞART. Türkçe, 6-10 sn temiz bir .wav koy veya REF_WAV=/yol/ref.wav ver."
    )

# ===========================================================================
# 2. MODEL YÜKLE (bir kez) — TTS.api auto-download + düşük seviye Xtts erişimi
# ===========================================================================
try:
    from TTS.api import TTS
except Exception as e:  # noqa: BLE001
    raise SystemExit(
        "[hata] coqui-tts import edilemedi. AYRI venv'de kur:\n"
        "  pip install -r requirements-xtts.txt\n"
        f"(orijinal hata: {e})"
    )

print("[bilgi] XTTS v2 yükleniyor (ilk sefer ~1.8 GB indirir)...")
_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
model = _tts.synthesizer.tts_model          # düşük seviye Xtts örneği (inference_stream burada)
SR    = _tts.synthesizer.output_sample_rate  # XTTS v2 -> 24000
FIRST_AUDIO_SAMPLES = round(FIRST_AUDIO_MS / 1000 * SR)
print(f"[bilgi] örnekleme hızı: {SR} Hz | ilk-ses eşiği: {FIRST_AUDIO_SAMPLES} örnek ({FIRST_AUDIO_MS} ms)")

# ===========================================================================
# 2.5 REFERANS -> LATENT (BİR KEZ). İstek başına TEKRAR işlenMEZ.
#     Chatterbox'ın her chunk'ta referansı yeniden işleme sorununun XTTS'teki
#     yapısal çözümü tam da budur.
# ===========================================================================
def load_latents():
    cache = REF_WAV + ".xtts_latents.pt"
    if CACHE_LATENTS and os.path.exists(cache):
        try:
            d = torch.load(cache, map_location=device)
            print(f"[bilgi] Referans latent'leri önbellekten yüklendi: {cache}")
            return d["gpt_cond_latent"].to(device), d["speaker_embedding"].to(device)
        except Exception as e:  # noqa: BLE001
            print(f"[uyarı] Latent önbelleği okunamadı ({e}); yeniden hesaplanıyor.")
    t0 = time.perf_counter()
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[REF_WAV],
        gpt_cond_len=REF_GPT_COND_SEC,
        gpt_cond_chunk_len=REF_GPT_COND_SEC,
        max_ref_length=REF_MAX_SEC,
        sound_norm_refs=SOUND_NORM_REFS,
    )
    print(f"[bilgi] Referans '{os.path.basename(REF_WAV)}' BİR KEZ latent'e çevrildi "
          f"({(time.perf_counter()-t0)*1000:.0f} ms; istek başına tekrar YOK).")
    if CACHE_LATENTS:
        try:
            torch.save({"gpt_cond_latent": gpt_cond_latent.detach().cpu(),
                        "speaker_embedding": speaker_embedding.detach().cpu()}, cache)
            print(f"[bilgi] Latent'ler önbelleğe yazıldı: {cache}")
        except Exception as e:  # noqa: BLE001
            print(f"[uyarı] Latent önbelleğe yazılamadı: {e}")
    return gpt_cond_latent, speaker_embedding

GPT_COND_LATENT, SPEAKER_EMB = load_latents()

# ===========================================================================
# 3. STREAMING ÜRETEÇ — XTTS native inference_stream
#    Ses parçalarını (1D CPU float tensor) üretildikçe yield eder.
# ===========================================================================
def _to_cpu_1d(x):
    """XTTS parçası (torch tensor VEYA numpy) -> 1D CPU float tensor. .cpu() zaten
    o parça için CUDA senkronu yapar (zamanlama güvenli)."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).reshape(-1).float()
    return x.detach().reshape(-1).float().cpu()

def generate_stream(text):
    with torch.inference_mode():
        stream = model.inference_stream(
            text,
            LANG,
            GPT_COND_LATENT,
            SPEAKER_EMB,
            stream_chunk_size=STREAM_CHUNK_SIZE,
            overlap_wav_len=OVERLAP_WAV_LEN,
            temperature=TEMPERATURE,
            length_penalty=LENGTH_PENALTY,
            repetition_penalty=REPETITION_PENALTY,
            top_k=TOP_K,
            top_p=TOP_P,
            speed=SPEED,
            enable_text_splitting=False,   # tek cümle -> segmentasyon gecikmesi ekleme
        )
        for chunk in stream:
            yield _to_cpu_1d(chunk)

# ===========================================================================
# 4. TEK CÜMLE ÖLÇÜMÜ (canlı görüşmede bir ajan repliği gibi)
#    ADİL TTFB = ilk FIRST_AUDIO_MS ms ses biriktiği an (ElevenLabs ile AYNI mantık).
# ===========================================================================
def speak_measure(text):
    _sync(); t0 = time.perf_counter()
    raw_first = None            # ilk parçanın geldiği an (bilgi; ElevenLabs 'ham ilk-bayt' muadili)
    ttfb = None                 # ilk 200 ms ses eline geçtiği an = ADİL TTFB
    first_chunk_samples = None
    nsamples = 0
    pieces = []
    for wav_chunk in generate_stream(text):
        _sync()
        now = time.perf_counter() - t0
        if raw_first is None:
            raw_first = now
            first_chunk_samples = wav_chunk.numel()
        nsamples += wav_chunk.numel()
        if ttfb is None and nsamples >= FIRST_AUDIO_SAMPLES:
            ttfb = now          # <-- ilk 200 ms hazır = ADİL TTFB
        pieces.append(wav_chunk)
    total = time.perf_counter() - t0
    audio = torch.cat(pieces).unsqueeze(0) if pieces else torch.zeros(1, 1)
    audio_s = audio.shape[-1] / SR
    if ttfb is None:            # yanıt 200 ms'den kısaysa: tüm ses gelince hazır
        ttfb = total
    first_chunk_ms = (first_chunk_samples or 0) / SR * 1000
    return ttfb, raw_first, total, audio_s, len(pieces), first_chunk_ms, audio

# ===========================================================================
# 5. ISINMA (ölçüm dışı — ilk CUDA derleme/autotune ortalamayı bozmasın)
# ===========================================================================
print(f"[bilgi] Isınıyor ({WARMUP} tur)...")
for _ in range(WARMUP):
    speak_measure(BENCH_TEXT)

# ===========================================================================
# 6. TTFB ÖLÇÜMÜ (ElevenLabs/Chatterbox ile kıyas — aynı cümle, aynı tur sayısı)
# ===========================================================================
print(f"\n[TTFB kıyası] cümle: \"{BENCH_TEXT[:48]}...\"  ({RUNS} tur)")
ttfbs, raws, totals, audio_lens, first_chunk_mss, last_audio = [], [], [], [], [], None
for i in range(RUNS):
    ttfb, raw_first, total, audio_s, nchunks, fc_ms, audio = speak_measure(BENCH_TEXT)
    last_audio = audio
    ttfbs.append(ttfb); raws.append(raw_first); totals.append(total)
    audio_lens.append(audio_s); first_chunk_mss.append(fc_ms)
    print(f"  tur {i+1:2}/{RUNS}: TTFB {ttfb*1000:6.0f} ms | ham ilk-parça {raw_first*1000:6.0f} ms "
          f"| tam {total*1000:6.0f} ms | ses {audio_s:.2f} sn | {nchunks} parça")

bench_wav = os.path.join(OUT_DIR, "out_bench_xtts.wav")
if SAVE_WAV and last_audio is not None:
    ta.save(bench_wav, last_audio.cpu(), SR)

# ===========================================================================
# 7. CANLI GÖRÜŞME SİMÜLASYONU (ajan cümle cümle konuşuyor)
# ===========================================================================
print(f"\n[canlı görüşme simülasyonu] {len(CONVERSATION)} replik")
for idx, line in enumerate(CONVERSATION, 1):
    ttfb, raw_first, total, audio_s, nchunks, fc_ms, audio = speak_measure(line)
    if SAVE_WAV:
        ta.save(os.path.join(OUT_DIR, f"xtts_turn_{idx}.wav"), audio.cpu(), SR)
    print(f"  replik {idx}: TTFB {ttfb*1000:6.0f} ms | tam {total*1000:6.0f} ms "
          f"| ses {audio_s:.2f} sn -> xtts_turn_{idx}.wav")

# ===========================================================================
# 8. RAPOR
# ===========================================================================
rtfs = [t / a for t, a in zip(totals, audio_lens) if a > 0]

print("\n" + "=" * 60)
print(f"model            : XTTS v2 (Coqui)")
print(f"GPU              : {gpu_name}")
print(f"optimizasyon     : stream_chunk {STREAM_CHUNK_SIZE} | ref {REF_GPT_COND_SEC} sn (1 kez) | TF32")
print(f"kıyas birimi     : ilk {FIRST_AUDIO_MS} ms ses — ElevenLabs/Chatterbox ile AYNI")
print(f"ADİL TTFB(median): {statistics.median(ttfbs)*1000:.0f} ms   <-- ElevenLabs'i GEÇMELİ (~182 ms)")
print(f"ADİL TTFB(min)   : {min(ttfbs)*1000:.0f} ms")
print(f"ham ilk-parça(med): {statistics.median(raws)*1000:.0f} ms   (bilgi; ilk sesin fiilen geldiği an)")
print(f"ilk parça süresi : ~{statistics.median(first_chunk_mss):.0f} ms ses/parça   "
      f"(>200 ise TTFB'yi düşürmek için STREAM_CHUNK_SIZE'ı azalt)")
print(f"tam    (median)  : {statistics.median(totals)*1000:.0f} ms")
print(f"RTF    (median)  : {statistics.median(rtfs):.2f}   "
      f"(tam süre / ses süresi; canlı akışın kopmaması için <1.0 ŞART)")
print("=" * 60)
print("KIYAS NOTU: Bu TTFB TAMAMEN yereldir (ağ yok) — saf hesaplama. ElevenLabs")
print("TTFB'si ise ağ+sunucu+model. Local avantajın tam da bu: ağ turu yok.")
print("XTTS referansı BİR KEZ latent'e çevrildi; istek başına yeniden işlenMEZ")
print("(Chatterbox'taki tekrar-işleme maliyeti burada YOK).")
print("LİSANS: XTTS v2 = CPML (TİCARİ DEĞİL). Üretim için Chatterbox (MIT) değerlendir.")

# ===========================================================================
# 9. (opsiyonel) TÜRKÇE DOĞRULUK — faster-whisper CER (kaliteyi de kıyasla)
# ===========================================================================
if MEASURE_CER and SAVE_WAV:
    try:
        import Levenshtein
        from faster_whisper import WhisperModel

        def cer(ref, hyp):
            ref, hyp = ref.lower().strip(), hyp.lower().strip()
            return Levenshtein.distance(ref, hyp) / max(len(ref), 1)

        asr_device = "cuda" if device == "cuda" else "cpu"
        asr_ct = "float16" if device == "cuda" else "int8"
        asr = WhisperModel("small", device=asr_device, compute_type=asr_ct,
                           local_files_only=False)
        segments, _info = asr.transcribe(bench_wav, language=LANG)
        hyp = "".join(seg.text for seg in segments)
        score = cer(BENCH_TEXT, hyp) * 100
        print(f"CER%             : {score:.1f}   (düşük = Türkçe daha net/doğru)")
        print(f"ASR duydu        : {hyp.strip()}")
    except Exception as e:  # noqa: BLE001
        print(f"[uyarı] CER ölçülemedi: {e}")
