"""
ElevenLabs STREAMING TTFB (ilk ses baytı) — KIYAS TABANI (baseline).

AMAÇ: Şu an prodüksiyonda kullanılan ElevenLabs modelinin CANLI görüşmedeki
gecikmesini ölçmek. Chatterbox'ı (localde) buradan DAHA DÜŞÜK gecikmeye
indirmek hedef. Yani bu script "aşılması gereken çıta"yı verir.

ADİL KIYAS İÇİN (chatterbox_colab_stream.py ile hizalı):
  - Aynı Türkçe cümle (TEXT).
  - AYNI BİRİM: "ilk FIRST_AUDIO_MS ms sesin eline geçmesi" (byte değil, SES SÜRESİ).
    Gelen baytları ses süresine çevirip aynı eşiği iki tarafta da kullanıyoruz;
    böylece format (mp3/pcm) farkı bu dönüşümle giderilir.
  - Aynı tur sayısı (RUNS) ve warmup.

FREE HESAP NOTU: pcm_* çıktısı ücretli tier ister. Free API'de OUTPUT_FORMAT
mp3 OLMALI (varsayılan mp3_44100_128, free'de çalışır). CBR mp3 olduğu için
bayt<->süre yaklaşık lineerdir; ölçüm ±birkaç ms yaklaşık ama adildir.

NOT (ağ payı): Buradaki süre = AĞ RTT + sunucu + model. Chatterbox tarafı saf
yereldir; bu local avantajı KASITLI korunur. Ağ payını görmek için:
`ping api.elevenlabs.io` -> RTT'yi süreden düşebilirsin.

KREDİ NOTU (free = 10.000 kredi/ay): her tur ~130 karakter; WARMUP(2)+RUNS(10)
= 12 istek ~= 1.500+ kredi/çalıştırma. Kotayı korumak için RUNS'ı düşürebilirsin.
"""
import os
import statistics
import time

from dotenv import load_dotenv
from elevenlabs import ElevenLabs

# .env dosyasını yükle (varsa). Değeri ayrıca OS ortam değişkeninden de okuyabilir;
# ikisi de varsa OS ortam değişkeni önceliklidir (override=False varsayılan).
load_dotenv()

# ---- Ayarlar ----
# API anahtarı ELEVENLABS_API_KEY değişkeninden okunur (kaynağa gömme -> sızma riski).
# Kaynak: proje kökündeki .env dosyası VEYA OS ortam değişkeni.
#   .env:  ELEVENLABS_API_KEY=sk_...
#   veya:  setx ELEVENLABS_API_KEY "sk_..."   (kalıcı, yeni terminal gerekir)
API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not API_KEY:
    raise SystemExit(
        "ELEVENLABS_API_KEY ayarlı değil. Proje kökünde .env dosyası oluştur:\n"
        "  ELEVENLABS_API_KEY=sk_...\n"
        'veya OS ortam değişkeni ayarla:  setx ELEVENLABS_API_KEY "sk_..."'
    )
# FREE HESAP NOTU: professional/library sesler (Doga=IuRRIAcbQK5AQk1XevPj,
# Jon=sB7vwSCyX0tQmU24cW2C) API'de ücretli plan ister (402 paid_plan_required).
# Free tier'da SADECE premade sesler çalışır. eleven_flash_v2_5 çok dilli olduğu
# için premade ses de Türkçe konuşur. Ücretli plana geçersen Doga'ya çevirebilirsin.
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Sarah (premade) - free tier'da çalışır

# Prodüksiyonda hangi modeli kullanıyorsan ONU yaz — çıta o olmalı.
#   eleven_flash_v2_5      : en düşük gecikme sınıfı (~75 ms MODEL, ağ hariç)
#   eleven_turbo_v2_5      : denge
#   eleven_multilingual_v2 : en iyi kalite, en yüksek gecikme
MODEL_ID = "eleven_flash_v2_5"

TEXT = ("Merhaba, ben Ahmet. Ürün demosu için on dört Temmuz Pazartesi "
        "saat on beş sıfır sıfır'da size randevu oluşturdum, uygun mudur?")

# ÇIKTI FORMATI — hesap tier'ına göre:
#   pcm_24000       : Chatterbox ile birebir aynı format ama ÜCRETLİ tier (Pro+) ister.
#   mp3_44100_128   : FREE tier'da çalışır (ElevenLabs varsayılanı). CBR (sabit bitrate)
#                     olduğu için bayt<->süre yaklaşık lineer -> "ilk X ms" mantığı geçerli.
#   mp3_22050_32    : her tier'da kesin çalışır (düşük kalite), en güvenli fallback.
# FREE HESAP: pcm KULLANMA -> mp3_44100_128 bırak.
OUTPUT_FORMAT = "mp3_44100_128"


def bytes_per_ms(fmt: str) -> float:
    """Formata göre 1 ms sesin kaç bayt olduğunu döndürür (bayt<->süre dönüşümü)."""
    if fmt.startswith("pcm_"):
        sr = int(fmt.split("_")[1])
        return sr * 2 / 1000                    # 16-bit mono -> 2 bayt/örnek
    if fmt.startswith("mp3_"):
        kbps = int(fmt.split("_")[2])           # örn. mp3_44100_128 -> 128 kbps (CBR)
        return kbps * 1000 / 8 / 1000           # 128 kbps -> 16 bayt/ms
    raise ValueError(f"Bilinmeyen format: {fmt}")


BYTES_PER_MS = bytes_per_ms(OUTPUT_FORMAT)

# --- ADİL KIYAS BİRİMİ: "ilk duyulabilir X ms ses" ---
# chatterbox_colab_stream.py'deki FIRST_AUDIO_MS ile AYNI değer olmalı. İlk byte'ta
# durmuyoruz; karşı tarafın ilk X ms sesi DUYMAYA başladığı ana kadar bayt biriktirip
# o anı damgalıyoruz -> Chatterbox ile birebir aynı mantık.
# NOT (mp3): CBR varsayımı + olası küçük başlık ofseti yüzünden ±birkaç ms yaklaşıktır;
# kesin sayı istersen pcm (ücretli) kullan. Free tier için bu yaklaşım yeterince adil.
FIRST_AUDIO_MS    = 200
FIRST_AUDIO_BYTES = int(FIRST_AUDIO_MS * BYTES_PER_MS)

# optimize_streaming_latency: SDK 2.56'da HÂLÂ destekleniyor ama ElevenLabs bunu
# "deprecated" işaretledi (gecikme optimizasyonu artık büyük ölçüde otomatik).
# Prodüksiyondaki en düşük gecikmeyi temsil etsin diye açık bırakıyoruz; SDK
# ileride kaldırırsa None yap. 0..4 (3 = yüksek optimizasyon, metin norm. açık).
OPT_LATENCY = 3

RUNS   = 10    # chatterbox_ttfb_stream.py ile AYNI
WARMUP = 2     # TLS/DNS/bağlantı kurulumu ölçüm dışı kalsın diye

client = ElevenLabs(api_key=API_KEY)


def measure_once():
    """Tek istek: (1) ilk X ms sesin geldiği an = ADİL TTFB, (2) ham ilk-bayt anı
    (bilgi amaçlı), (3) toplam süre."""
    t0 = time.perf_counter()
    stream = client.text_to_speech.stream(
        VOICE_ID,
        text=TEXT,
        model_id=MODEL_ID,
        output_format=OUTPUT_FORMAT,
        optimize_streaming_latency=OPT_LATENCY,
    )
    raw_first = None       # ham ilk bayt (Chatterbox ile kıyasta KULLANILMAZ, referans)
    ttfa = None            # ilk FIRST_AUDIO_MS ms ses eline geçtiği an = ADİL TTFB
    nbytes = 0
    for chunk in stream:            # HTTP isteği burada başlar, baytlar akar
        if not chunk:
            continue
        if raw_first is None:
            raw_first = time.perf_counter() - t0
        nbytes += len(chunk)
        if ttfa is None and nbytes >= FIRST_AUDIO_BYTES:
            ttfa = time.perf_counter() - t0     # <-- ilk X ms ses hazır = ADİL TTFB
    total = time.perf_counter() - t0
    if ttfa is None:                # yanıt X ms'den kısaysa: tüm ses gelince hazırdır
        ttfa = total
    return ttfa, raw_first, total, nbytes


print(f"[bilgi] model={MODEL_ID}  format={OUTPUT_FORMAT}  opt_latency={OPT_LATENCY}")
print(f"[bilgi] kıyas birimi: ilk {FIRST_AUDIO_MS} ms ses (={FIRST_AUDIO_BYTES} bayt) "
      f"— chatterbox_colab_stream.py ile AYNI olmalı")
print(f"[bilgi] Isınıyor ({WARMUP} tur)...")
for _ in range(WARMUP):
    measure_once()

ttfas, raws, totals = [], [], []
for i in range(RUNS):
    ttfa, raw_first, total, nbytes = measure_once()
    ttfas.append(ttfa); raws.append(raw_first); totals.append(total)
    print(f"  {i+1:2}/{RUNS}: TTFB(ilk {FIRST_AUDIO_MS}ms) {ttfa*1000:6.0f} ms | "
          f"ham ilk-bayt {raw_first*1000:6.0f} ms | tam {total*1000:6.0f} ms | {nbytes:>7} bayt")

print("\n" + "=" * 60)
print(f"model               : {MODEL_ID}")
print(f"ADİL TTFB (median)  : {statistics.median(ttfas)*1000:.0f} ms   <-- Chatterbox bunu GEÇMELİ")
print(f"ADİL TTFB (min)     : {min(ttfas)*1000:.0f} ms   (ağ en iyi durumda)")
print(f"ham ilk-bayt(median): {statistics.median(raws)*1000:.0f} ms   (bilgi amaçlı, kıyasta KULLANMA)")
print(f"tam       (median)  : {statistics.median(totals)*1000:.0f} ms")
print("=" * 60)
print(f"KIYAS: 'ADİL TTFB' = ilk {FIRST_AUDIO_MS} ms sesin eline geçme süresi; Chatterbox")
print("da AYNI süre için ölçülür -> elmayla elma. Bu sayı AĞ+sunucu+model içerir;")
print("Chatterbox localde AĞSIZ çalışır ve local avantajı KASITLI olarak korunur.")
