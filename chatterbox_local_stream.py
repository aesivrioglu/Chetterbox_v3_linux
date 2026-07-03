"""
============================================================================
Chatterbox Multilingual TTS (v3) — YEREL (Linux + RTX/CUDA) STREAMING scripti (v3)
============================================================================

AMAÇ
  chatterbox_colab_stream.py ile BİREBİR AYNI ölçüm mantığı; tek fark ortam:
  bu script arayüzsüz (headless) bir Linux makinede, yerel RTX GPU
  (RTX 3090 / 4090) üzerinde çalışır. TTFB ve tam süre AYNI birimle (ilk
  FIRST_AUDIO_MS ms ses) ölçülür -> Colab ve RTX sonuçları elmayla elma.

v2 — YAPISAL HIZ OPTİMİZASYONLARI (bölüm 0.5; gerekçe: README-chatterbox.md)
    1) OPT_REF_TRIM_SEC : s3gen referans prompt'u 10 sn -> 3 sn (İLK parça).
    2) OPT_CFM_STEPS    : CFM (flow matching) Euler adımı 10 -> 6 (İLK parça).
    3) OPT_FLOW_AUTOCAST: CFM+encoder fp16/bf16 çalışır (HiFT fp32 kalır).
  SONUÇ: RTX 4090'da TTFB medyanı ~140 ms (2026-07) -> ElevenLabs çıtası
  (~182 ms) GEÇİLDİ. Bu yüzden v3'ün odağı SES KALİTESİ.

v3 — SES KALİTESİ (bölüm 0.6; KIYAS BİRİMİ ve İLK PARÇA HIZ YOLU DEĞİŞMEDİ)
    4) OPT_REF_PREPROCESS : referans ses modele daha iyi 'yedirilir' — baş/son
       sessizlik kırpılır, >0.5 sn iç duraklamalar sıkıştırılır, tepe -3 dBFS'e
       normallenir. 3 sn'lik akustik prompt böylece tamamen GERÇEK konuşmayla
       dolar (sessizlik prompt bütçesi çalar); x-vector daha temiz klipten çıkar.
    5) OPT_CTX_TOKENS     : parçalar (ilki hariç) önceki son 16 token'la
       birlikte vocode edilir, bağlama düşen ses atılır -> parça sınırında
       tını/zarf sürekliliği (CosyVoice'un streaming yaklaşımı). TTFB'ye etki YOK
       (ilk parçada bağlam yok).
    6) OPT_JOINT_FADE_MS  : parça eklerinde 5 ms kosinüs mikro-fade -> HiFT faz
       süreksizliğinden doğan 'klik'ler yok olur. Gecikme EKLEMEZ (parça
       uzunlukları değişmez, tutma/bekletme yok).
    7) OPT_*_REST         : TTFB'yi yalnız İLK parça belirler; SONRAKİ parçalar
       TAM referans + CFM 10 adımla üretilir (ilk parça v2 hız ayarında kalır).
       Sesin ~%95'i tam kalite yolundan çıkar; RTF < 1.0 kaldıkça bedavadır.
    8) OPT_T3_BF16_WEIGHTS: T3 Llama ağırlıkları bf16'ya çevrilir (yalnız
       Ampere+). Autocast zaten bf16 HESAPLIYORDU; ağırlık da bf16 olunca token
       başına tekrarlanan fp32->bf16 ağırlık kopyaları kalkar -> decode hızlanır.
       Örnekleme yine fp32 logits üzerinde yapılır (dağılım pratikte aynı).
  ÖLÇÜM DİSİPLİNİ: bench turlarında ses SAKLANMAZ (kayıt maliyeti sıfır);
  ölçüm bitince İSTATİSTİĞE DAHİL OLMAYAN 1 ek koşu out_bench.wav'ı üretir,
  CER bu kayıttan ölçülür. Görüşme simülasyonu turn_*.wav üretmeye devam eder
  (oradaki TTFB'ler bilgi amaçlıdır; kıyas bench'i 6. bölümdür).

ADİL KIYAS
  FIRST_AUDIO_MS, PRESET, CONVERSATION, BENCH_TEXT, RUNS, WARMUP ve İLK PARÇA
  bayrakları Colab scripti ile AYNI tutulmuştur. v3 kalite bayrakları ilk parça
  yoluna dokunmadığından TTFB kıyası bozulmaz; toplam süre/RTF ise REST
  parçaların daha kaliteli (daha pahalı) üretilmesi yüzünden v2'den yüksektir.

DOĞRULUK: Streaming döngüsü t3.py -> T3.inference() ile BİREBİR aynıdır.
  Tek istisna: top_p=1.0 iken TopP warper matematiksel KİMLİKTİR ve atlanır;
  örnekleme dağılımı stokla AYNI kalır. v3 kalite katmanları yalnız vocoder
  (s3gen) tarafına dokunur; T3 token dağılımı değişmez.

----------------------------------------------------------------------------
KURULUM (Linux, RTX 3090/4090 — bir kez; ayrıntı için requirements-chatterbox.txt)
----------------------------------------------------------------------------
    python3.11 -m venv .venv && source .venv/bin/activate
    pip install -U pip
    pip install -r requirements-chatterbox.txt
    # RTX 3090/4090 (Ampere/Ada) için torch 2.6.0'ın CUDA (cu124) wheel'i uyumludur;
    # CPU wheel gelirse:
    #   pip install --force-reinstall torch==2.6.0 torchaudio==2.6.0 \
    #       --index-url https://download.pytorch.org/whl/cu124

ÇALIŞTIRMA
    REF_WAV=/yol/sayfa-33.wav OUT_DIR=out python chatterbox_local_stream.py
    # GPU kontrolü:  python -c "import torch; print(torch.cuda.get_device_name(0))"

NOT (sürüm farkları): `t3_model`, `flow_inference`/`hift_inference` ve
`n_cfm_timesteps` bazı chatterbox build'lerinde YOKTUR. Hepsi imza/hasattr
kontrolüyle tespit edilir; desteklenmeyen optimizasyon uyarı basıp KENDİNİ
KAPATIR, script stok yola düşerek yine çalışır.
----------------------------------------------------------------------------
"""
import inspect
import math
import os
import statistics
import time

import torch
import torch.nn.functional as F
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm
from chatterbox.models.s3tokenizer import drop_invalid_tokens, S3_TOKEN_RATE
from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
from transformers.generation.logits_process import (
    TopPLogitsWarper, MinPLogitsWarper, RepetitionPenaltyLogitsProcessor,
)

# ===========================================================================
# 0. AYARLAR  (Colab scripti ile AYNI olmalı — kıyas için)
# ===========================================================================
LANG    = "tr"
# Referans ses yolu: env > script klasöründeki sayfa-33.wav.
_HERE   = os.path.dirname(os.path.abspath(__file__))
REF_WAV = os.getenv("REF_WAV", os.path.join(_HERE, "sayfa-33.wav"))
OUT_DIR = os.getenv("OUT_DIR", _HERE)                # wav çıktıları buraya
os.makedirs(OUT_DIR, exist_ok=True)

# Ajan preset'i (mtl_tts.generate varsayılanlarıyla uyumlu; sadece exagg/cfg/temp ayarlı)
# Kalite notu: bu değerler T3 ÖRNEKLEME dağılımını belirler; kıyas sürekliliği
# için v2 ile aynı bırakıldı. Denemeye değer alternatifler (Chatterbox önerileri):
#   - referans konuşmacı HIZLI konuşuyorsa cfg_weight'i 0.3'te tut (tempo düzelir),
#     değilse 0.5 stok değeri referansa bağlılığı artırır.
#   - temperature 0.6 ajan kararlılığı içindir; 0.8 stok değeri prozodiyi
#     canlandırır ama CER oynaklığını artırabilir.
PRESET = dict(exaggeration=0.4, cfg_weight=0.3, temperature=0.6,
              repetition_penalty=1.2, min_p=0.05, top_p=1.0)

# --- ADİL KIYAS BİRİMİ: "ilk duyulabilir X ms ses" ---
# elevenlabs_ttfb.py ve chatterbox_colab_stream.py'deki FIRST_AUDIO_MS ile AYNI.
FIRST_AUDIO_MS = 200
# 25 token/sn -> 200 ms ~= 5 token. Chatterbox ilk parçayı bu kadar ses birikince verir.
# NOT: ~120 ms (3 token) altına inme; vocoder çok az token'da kararsızlaşır.
FIRST_CHUNK_TOKENS = max(1, round(FIRST_AUDIO_MS / 1000 * S3_TOKEN_RATE))
CHUNK_TOKENS       = 40      # sonraki parçalar (TTFB'yi etkilemez, akış sürekliliği için)
MAX_NEW_TOKENS     = 1000

# Ölçüm
RUNS     = 10               # elevenlabs_ttfb.py / colab ile AYNI
WARMUP   = 2
SAVE_WAV = True
MEASURE_CER = True          # faster-whisper ile Türkçe doğruluk (kaliteyi de kıyasla)

# Canlı görüşme simülasyonu (Colab scripti ile AYNI cümleler)
CONVERSATION = [
    "Merhaba, ben Ahmet. Ürün demosu için on dört Temmuz Pazartesi "
    "saat on beş sıfır sıfır'da size randevu oluşturdum, uygun mudur?",
    "Anladım, size uygun başka bir gün ve saat önerebilir miyim?",
    "Harika, o zaman görüşmemizi teyit ediyorum. İyi günler dilerim.",
]
# TTFB kıyas cümlesi (ElevenLabs scriptindeki TEXT ile AYNI olmalı)
BENCH_TEXT = CONVERSATION[0]

# ===========================================================================
# 0.5 YAPISAL HIZ BAYRAKLARI (v2 — gerekçeler: README-chatterbox.md)
#     Colab scripti ile AYNI değerler — kıyas için. Kıyas birimine DOKUNMAZLAR;
#     yalnızca aynı 200 ms sesin daha hızlı üretilmesini sağlarlar.
#     v3'ten itibaren bu ikisi yalnız İLK parçaya uygulanır (REST bayrakları
#     sonraki parçaları tam kalitede üretir — bölüm 0.6 / 7).
# ===========================================================================
# 1) s3gen referans prompt'u (stok: 10 sn). CFM+encoder maliyeti (referans+parça)
#    dizi uzunluğuyla orantılıdır ve referans kısmı HER parçada yeniden üretilip
#    ATILIR (flow.py: feat[:, :, mel_len1:]). 3 sn -> ilk parça ~3x hızlı.
#    Konuşmacı kimliği (x-vector 'embedding') TAM klipten gelmeye devam eder.
OPT_REF_TRIM_SEC = 3.0        # None/0 = kapalı (stok 10 sn)
# 2) CFM (flow matching) Euler adım sayısı (stok: 10; adet başına 2 estimator
#    geçişi, dahili CFG yüzünden). Maliyet adımla lineer. Kulağa+CER'e göre 4-10.
OPT_CFM_STEPS = 6             # None = build varsayılanı (10)
# 3) CFM+encoder'ı autocast ile çalıştır (T4: fp16, Ampere+: bf16). HiFT vocoder
#    her durumda fp32 kalır (ses güvenliği). Ağırlık dönüşümü YOK -> sürüm-güvenli.
OPT_FLOW_AUTOCAST = True      # False = stok (s3gen tamamı fp32)

# ===========================================================================
# 0.6 KALİTE BAYRAKLARI (v3 — gerekçeler: README-chatterbox.md)
#     TTFB'yi yalnız İLK parça belirler; bu bayraklar ilk parçanın HIZ yoluna
#     dokunmaz. Hepsi bağımsız kapatılabilir.
# ===========================================================================
# 4) Referans ön-işleme: baş/son sessizliği kırp, >0.5 sn iç duraklamaları
#    ~0.25 sn'ye sıkıştır, tepeyi -3 dBFS'e normalle. 3 sn'lik akustik prompt
#    böylece tamamen gerçek konuşmayla dolar; CFM prompt'u sağlıklı seviyede
#    olur. Çıktı: OUT_DIR/ref_processed.wav (orijinal dosyaya DOKUNULMAZ).
OPT_REF_PREPROCESS = True
# 5) Bağlamlı vocode: her parça (ilki hariç) önceki son N token'la birlikte
#    CFM'den geçirilir, bağlama düşen ses atılır -> parça sınırında tını/zarf
#    sürekliliği. Maliyet: parça başına +N token CFM (TTFB'ye etkisi YOK).
#    0 = kapalı. 4'ün altına indirme: s3gen'in trim_fade bölgesi atılan bağlam
#    içinde kalmalı, yoksa parça başı fade'i duyulur.
OPT_CTX_TOKENS = 16
# 6) Parça eki mikro-fade'i: her ekte önceki parçanın son / yeni parçanın ilk
#    birkaç ms'ine kosinüs rampası -> HiFT faz süreksizliği 'klik'leri yok olur.
#    Tutma/bekletme yok -> gecikme eklemez. 0 = kapalı.
OPT_JOINT_FADE_MS = 5
# 7) İlk parça hızlı / devamı kaliteli: SONRAKİ parçalar için referans uzunluğu
#    ve CFM adımı. None = TAM referans / build varsayılanı (10 adım).
#    T4 gibi yavaş GPU'da RTF > 1.0 olursa bunları v2 değerlerine (3.0 / 6) indir.
#    Sınırda tını kayması duyarsan ya OPT_REF_TRIM_SEC'i yükselt (TTFB'ye +ms)
#    ya da OPT_REF_TRIM_SEC_REST = 3.0 yap (iki yol da tutarlılığı artırır).
OPT_REF_TRIM_SEC_REST = None  # None = TAM referans (ilk parça OPT_REF_TRIM_SEC kullanır)
OPT_CFM_STEPS_REST    = 10    # ilk parça OPT_CFM_STEPS (6) kullanır
# 8) T3 Llama ağırlıklarını bf16'ya çevir (yalnız Ampere+ / bf16 yolunda).
#    Autocast zaten bf16 hesaplıyordu; ağırlık da bf16 olunca token başına
#    tekrarlanan fp32->bf16 ağırlık kopyaları kalkar -> decode hızlanır.
#    Örnekleme fp32 logits üzerinde kalır; bf16 yuvarlaması autocast ile aynıdır.
OPT_T3_BF16_WEIGHTS = True

# ===========================================================================
# 1. CİHAZ + DTYPE (Ampere+ -> bf16, Turing/T4 -> fp16). RTX 3090/4090 -> bf16.
# ===========================================================================
if torch.cuda.is_available():
    device = "cuda"
    major, _ = torch.cuda.get_device_capability()
    autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    gpu_name = torch.cuda.get_device_name(0)
else:
    device = "cpu"
    autocast_dtype = torch.float32
    gpu_name = "CPU (GPU bulunamadı — CUDA/torch kurulumunu kontrol et!)"
use_autocast = device == "cuda"
print(f"[cihaz] {device} | {gpu_name} | dtype={autocast_dtype if use_autocast else 'fp32'}")

def _sync():
    if device == "cuda":
        torch.cuda.synchronize()

# ===========================================================================
# 1.5 REFERANS ÖN-İŞLEME (v3 — 'referansı daha iyi yedir')
#     Neden: prepare_conditionals akustik prompt'u klibin BAŞINDAN alır ve v2
#     onu 3 sn'ye kırpar. Baştaki sessizlik / uzun duraklamalar bu bütçeyi
#     çalar -> prompt yarı boş kalır, tını/prozodi kopyası zayıflar. Ayrıca
#     CFM prompt mel'i seviyeye duyarlıdır: cılız kayıt cılız çıktı üretir.
# ===========================================================================
def _preprocess_ref_wav(src, dst, sil_ratio=0.03, pad_s=0.10,
                        max_pause_s=0.50, keep_pause_s=0.25, xfade_s=0.010,
                        peak_dbfs=-3.0):
    """src'yi temizleyip dst'ye yazar ve dst'yi döndürür; herhangi bir aksilikte
    (okunamadı, tamamı sessiz, aşırı kısaldı) uyarı basıp src'yi döndürür.
      1) baş/son sessizlik kırpma  (eşik: tepe çerçeve RMS'inin %3'ü, ~-30 dB)
      2) >max_pause_s iç duraklamaları keep_pause_s'e sıkıştırma (10 ms crossfade)
      3) tepe genliği peak_dbfs'e normalleme
    """
    try:
        wav, sr = ta.load(src)
    except Exception as e:  # noqa: BLE001
        print(f"[uyarı] referans okunamadı ({e}); orijinal dosya kullanılacak.")
        return src
    x = wav.mean(0) if wav.shape[0] > 1 else wav[0]
    orig_s = x.numel() / sr
    frame, hop = int(0.02 * sr), int(0.01 * sr)
    if x.numel() < 2 * frame:
        return src
    rms = x.unfold(0, frame, hop).pow(2).mean(1).sqrt()
    thr = float(rms.max()) * sil_ratio
    speech = rms > thr
    if not bool(speech.any()):
        print("[uyarı] referansta konuşma bulunamadı; orijinal dosya kullanılacak.")
        return src
    nz = speech.nonzero().flatten()
    s0 = max(0, int(nz[0]) * hop - int(pad_s * sr))
    s1 = min(x.numel(), int(nz[-1]) * hop + frame + int(pad_s * sr))
    x = x[s0:s1]

    # --- iç duraklamaları sıkıştır (kesim yerlerinde xfade_s crossfade) ---
    rms = x.unfold(0, frame, hop).pow(2).mean(1).sqrt()
    speech = rms > thr
    max_pause_f = max(1, int(max_pause_s * sr / hop))
    keep_half = int(keep_pause_s * sr / 2)
    cuts, i, n = [], 0, int(speech.numel())
    while i < n:
        if bool(speech[i]):
            i += 1
            continue
        j = i
        while j < n and not bool(speech[j]):
            j += 1
        if (j - i) > max_pause_f:                     # uzun duraklama
            a = i * hop + keep_half                   # her iki yandan keep_half bırak
            b = min(x.numel(), j * hop) - keep_half
            if b - a > int(0.05 * sr):
                cuts.append((a, b))
        i = j
    if cuts:
        segs, prev = [], 0
        for a, b in cuts:
            segs.append(x[prev:a]); prev = b
        segs.append(x[prev:])
        xf = int(xfade_s * sr)
        y = segs[0]
        for s in segs[1:]:
            if xf and y.numel() > xf and s.numel() > xf:
                r = torch.linspace(0.0, 1.0, xf)
                y = torch.cat([y[:-xf], y[-xf:] * (1 - r) + s[:xf] * r, s[xf:]])
            else:
                y = torch.cat([y, s])
        x = y
    if x.numel() < sr:                                # <1 sn kaldıysa güvenli taraf
        print("[uyarı] ön-işleme sonrası referans çok kısaldı; orijinal kullanılacak.")
        return src

    peak = float(x.abs().max())
    if peak > 0:
        x = x * (10 ** (peak_dbfs / 20) / peak)
    try:
        ta.save(dst, x.unsqueeze(0), sr)
    except Exception as e:  # noqa: BLE001
        print(f"[uyarı] işlenmiş referans yazılamadı ({e}); orijinal kullanılacak.")
        return src
    print(f"[bilgi] referans ön-işleme: {orig_s:.1f} sn -> {x.numel() / sr:.1f} sn "
          f"(baş/son kırpıldı, {len(cuts)} uzun duraklama sıkıştırıldı, "
          f"tepe {peak_dbfs:.0f} dBFS) -> {dst}")
    return dst

REF_WAV_EFF = REF_WAV
if OPT_REF_PREPROCESS and REF_WAV and os.path.exists(REF_WAV):
    REF_WAV_EFF = _preprocess_ref_wav(REF_WAV, os.path.join(OUT_DIR, "ref_processed.wav"))

# ===========================================================================
# 2. MODEL + REFERANS (bir kez)
# ===========================================================================
# t3_model="v3" bazı chatterbox build'lerinde YOK -> imzayı kontrol et, varsa geç.
if "t3_model" in inspect.signature(ChatterboxMultilingualTTS.from_pretrained).parameters:
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
else:
    print("[uyarı] Bu chatterbox build'i t3_model parametresini desteklemiyor; "
          "varsayılan multilingual model yükleniyor.")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
if REF_WAV_EFF and os.path.exists(REF_WAV_EFF):
    model.prepare_conditionals(REF_WAV_EFF, exaggeration=PRESET["exaggeration"])
    print(f"[bilgi] Referans '{REF_WAV_EFF}' bir kez gömüldü.")
else:
    print(f"[uyarı] '{REF_WAV_EFF}' yok; varsayılan ses (conds.pt) kullanılacak.")

# ===========================================================================
# 2.5 OPTİMİZASYONLARI BUILD'E GÖRE ETKİNLEŞTİR (sürüm-güvenli)
#     Eski build'lerde flow_inference/hift_inference veya n_cfm_timesteps
#     olmayabilir -> imzayla tespit et, yoksa uyar ve stok yola düş.
# ===========================================================================
_HAS_SPLIT = hasattr(model.s3gen, "flow_inference") and hasattr(model.s3gen, "hift_inference")
if _HAS_SPLIT:
    _STEPS_OK = "n_cfm_timesteps" in inspect.signature(model.s3gen.flow_inference).parameters
else:
    _STEPS_OK = "n_cfm_timesteps" in inspect.signature(model.s3gen.inference).parameters
_FLOW_AUTOCAST = bool(OPT_FLOW_AUTOCAST and use_autocast and _HAS_SPLIT)
_CFM_STEPS_FIRST = OPT_CFM_STEPS if (OPT_CFM_STEPS and _STEPS_OK) else None
_CFM_STEPS_REST  = OPT_CFM_STEPS_REST if (OPT_CFM_STEPS_REST and _STEPS_OK) else None
if OPT_FLOW_AUTOCAST and use_autocast and not _HAS_SPLIT:
    print("[uyarı] Bu build'de flow/hift ayrı çağrılamıyor; flow autocast KAPATILDI "
          "(stok fp32 yol kullanılacak).")
if (OPT_CFM_STEPS or OPT_CFM_STEPS_REST) and not _STEPS_OK:
    print("[uyarı] Bu build n_cfm_timesteps desteklemiyor; CFM adımı varsayılanda (10) kalacak.")

def _trimmed_ref_dict(gen, seconds):
    """s3gen referans prompt'unu ilk `seconds` saniyeye indirir (stok kırpma da
    klibin BAŞINI alır -> aynı yön). Yalnızca CFM'in her parçada yeniden işlediği
    akustik prompt (prompt_token + prompt_feat) kısalır; konuşmacı embedding'i
    ('embedding', x-vector) TAM klipten kalır. Sığ kopya: model.conds.gen bozulmaz."""
    if not seconds:
        return gen
    try:
        n_tok = int(round(seconds * S3_TOKEN_RATE))          # 25 token/sn
        if n_tok >= int(gen["prompt_token"].shape[1]):
            return gen                                       # referans zaten kısa
        g = dict(gen)
        g["prompt_token"] = gen["prompt_token"][:, :n_tok]
        tl = gen["prompt_token_len"].clone(); tl[0] = n_tok
        g["prompt_token_len"] = tl
        g["prompt_feat"] = gen["prompt_feat"][:, :2 * n_tok]  # mel = 2 x token (embed_ref garantisi)
        return g
    except (KeyError, IndexError, RuntimeError) as e:  # beklenmedik build farkı -> stok
        print(f"[uyarı] Referans kırpılamadı ({e}); tam referans kullanılacak.")
        return gen

# İlk parça: v2 hız ayarı (TTFB yolu). Devam parçaları: tam kalite (v3).
REF_DICT_FIRST = _trimmed_ref_dict(model.conds.gen, OPT_REF_TRIM_SEC)
REF_DICT_REST  = _trimmed_ref_dict(model.conds.gen, OPT_REF_TRIM_SEC_REST)
_ref_first_sec = REF_DICT_FIRST["prompt_token"].shape[1] / S3_TOKEN_RATE
_ref_rest_sec  = REF_DICT_REST["prompt_token"].shape[1] / S3_TOKEN_RATE

# T3 ağırlıklarını bf16'ya çevir (yalnız bf16/autocast yolunda anlamlı ve güvenli).
_T3_BF16 = False
if OPT_T3_BF16_WEIGHTS and use_autocast:
    if autocast_dtype == torch.bfloat16:
        try:
            model.t3.tfmr.to(torch.bfloat16)
            _T3_BF16 = True
        except Exception as e:  # noqa: BLE001
            print(f"[uyarı] T3 bf16'ya çevrilemedi ({e}); fp32 ağırlık + autocast ile devam.")
    else:
        print("[uyarı] GPU Ampere öncesi (fp16 yolu) — taşma riski nedeniyle T3 "
              "ağırlıkları fp32 bırakıldı (autocast fp16 zaten aktif).")

_flow_dtype_desc = (("autocast " + ("bf16" if autocast_dtype == torch.bfloat16 else "fp16"))
                    if _FLOW_AUTOCAST else "fp32 (stok)")
OPT_DESC = (f"ilk[ref {_ref_first_sec:.1f} sn, CFM {_CFM_STEPS_FIRST or 10}] | "
            f"devam[ref {_ref_rest_sec:.1f} sn, CFM {_CFM_STEPS_REST or 10}] | "
            f"flow {_flow_dtype_desc} | ctx {OPT_CTX_TOKENS} tok | "
            f"fade {OPT_JOINT_FADE_MS} ms | "
            f"T3 {'bf16' if _T3_BF16 else 'fp32+autocast'} | "
            f"ref önişleme {'AÇIK' if REF_WAV_EFF != REF_WAV else 'kapalı'}")
print(f"[bilgi] optimizasyonlar: {OPT_DESC}")
print( "        (stok: ref 10.0 sn | CFM 10 | fp32 | ctx yok | fade yok)")

_TRIM_FADE = getattr(model.s3gen, "trim_fade", None)

# Parça eki mikro-fade rampaları (CPU'da, yield edilen kopyaya uygulanır).
_FADE_N = int(model.sr * OPT_JOINT_FADE_MS / 1000) if OPT_JOINT_FADE_MS else 0
if _FADE_N:
    _ramp = torch.linspace(0.0, 1.0, _FADE_N)
    _FADE_IN  = 0.5 - 0.5 * torch.cos(math.pi * _ramp)     # 0 -> 1 (kosinüs)
    _FADE_OUT = torch.flip(_FADE_IN, dims=[0])             # 1 -> 0

def vocode_chunk(new_tokens, ctx_tokens=None, first=False):
    """Token parçasını sese çevirir.
    first=True  -> v2 hız ayarları (kısa referans + az CFM adımı): TTFB yolu.
    first=False -> kalite ayarları (REF_DICT_REST + _CFM_STEPS_REST).
    ctx_tokens  -> önceki parçanın son token'ları bağlam olarak başa eklenir,
                   bağlama düşen ses kesilip atılır (sınır sürekliliği).
    Split yol: CFM+encoder autocast'te, HiFT fp32'de — işlem sırası ve trim_fade
    stok s3gen.inference ile BİREBİR. Eski build: stok s3gen.inference (fp32)."""
    ref_dict = REF_DICT_FIRST if first else REF_DICT_REST
    steps = _CFM_STEPS_FIRST if first else _CFM_STEPS_REST
    has_ctx = ctx_tokens is not None and ctx_tokens.numel() > 0
    tokens = torch.cat([ctx_tokens, new_tokens]) if has_ctx else new_tokens
    kw = {"n_cfm_timesteps": steps} if steps else {}
    if _HAS_SPLIT:
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=_FLOW_AUTOCAST):
            mels = model.s3gen.flow_inference(tokens, ref_dict=ref_dict,
                                              finalize=True, **kw)
        with torch.autocast(device_type=device, enabled=False):   # HiFT fp32 -> ses güvenli
            wav, _ = model.s3gen.hift_inference(mels.float())
        # trim_fade CFM'in parça başı artıklarını bastırır. Bağlam varken fade
        # bölgesi zaten atılacak bağlam sesine düşer -> uygulamak gereksiz.
        if _TRIM_FADE is not None and not has_ctx:
            wav[:, :_TRIM_FADE.shape[0]] *= _TRIM_FADE
    else:
        with torch.autocast(device_type=device, enabled=False):   # stok: tamamı fp32
            wav, _ = model.s3gen.inference(speech_tokens=tokens, ref_dict=ref_dict, **kw)
    if has_ctx:  # bağlama düşen kısmı at (örnek/token oranıyla — hop'tan bağımsız)
        total = ctx_tokens.numel() + new_tokens.numel()
        cut = int(round(wav.shape[-1] * ctx_tokens.numel() / total))
        wav = wav[:, cut:]
    return wav

# ===========================================================================
# 3. STREAMING ÜRETEÇ — T3.inference() döngüsü + parça parça vocoder
#    Ses parçalarını (1D CPU tensor) üretildikçe yield eder. `stage` dict'i
#    verilirse ilk parçanın TTFB bütçesini yazar: prefill / decode / vocode.
# ===========================================================================
def generate_stream(text, first_chunk=FIRST_CHUNK_TOKENS, chunk=CHUNK_TOKENS, stage=None):
    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=autocast_dtype, enabled=use_autocast
    ):
        t_gen0 = time.perf_counter()
        t3 = model.t3
        hp = t3.hp
        cond = model.conds.t3
        cfg_weight  = PRESET["cfg_weight"]
        temperature = PRESET["temperature"]

        # --- metin token'ları (mtl_tts.generate() ile birebir) ---
        norm = punc_norm(text)
        text_tokens = model.tokenizer.text_to_tokens(norm, language_id=LANG).to(device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)          # CFG için 2 dizi
        text_tokens = F.pad(text_tokens, (1, 0), value=hp.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=hp.stop_text_token)

        # --- ilk embed (t3.inference() ile birebir) ---
        initial_speech = hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _len_cond = t3.prepare_input_embeds(
            t3_cond=cond, text_tokens=text_tokens,
            speech_tokens=initial_speech, cfg_weight=cfg_weight,
        )

        patched = T3HuggingfaceBackend(
            config=t3.cfg, llama=t3.tfmr,
            speech_enc=t3.speech_emb, speech_head=t3.speech_head,
        )

        bos_token = torch.tensor([[hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = t3.speech_emb(bos_token) + t3.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = torch.cat([bos_embed, bos_embed])                       # CFG batch=2
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        generated_ids = bos_token.clone()
        predicted = []
        # top_p=1.0 -> TopP warper matematiksel KİMLİK (hiçbir token'ı elemez);
        # token başına gereksiz sort yapmamak için yalnızca top_p<1.0'da kur.
        top_p_warper = TopPLogitsWarper(top_p=PRESET["top_p"]) if PRESET["top_p"] < 1.0 else None
        min_p_warper = MinPLogitsWarper(min_p=PRESET["min_p"]) if PRESET["min_p"] > 0 else None
        rep_proc = RepetitionPenaltyLogitsProcessor(penalty=float(PRESET["repetition_penalty"]))

        output = patched(inputs_embeds=inputs_embeds, past_key_values=None,
                         use_cache=True, output_hidden_states=True, return_dict=True)
        past = output.past_key_values
        if stage is not None:
            _sync()
        t_prefill_end = time.perf_counter()
        if stage is not None:
            stage["prefill"] = t_prefill_end - t_gen0

        emitted_idx = 0
        next_threshold = first_chunk
        ctx_tail = None            # bağlamlı vocode için: vocode edilmiş son token'lar

        for i in range(MAX_NEW_TOKENS):
            logits_step = output.logits[:, -1, :]
            c = logits_step[0:1, :]
            u = logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=c.device, dtype=c.dtype)
            logits = (c + cfg * (c - u)).float()          # örnekleme kararlılığı için fp32
            logits = rep_proc(generated_ids[:1], logits)
            if temperature != 1.0:
                logits = logits / temperature
            if min_p_warper is not None:
                logits = min_p_warper(generated_ids[:1], logits)
            if top_p_warper is not None:
                logits = top_p_warper(generated_ids[:1], logits)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            is_eos = bool((next_token.view(-1) == hp.stop_speech_token).item())

            # --- yeterince token biriktiyse (ya da EOS) -> parçayı vocode et ---
            if len(predicted) >= next_threshold or is_eos:
                new_tokens = torch.cat(predicted[emitted_idx:], dim=1)[0]    # sadece yeni token'lar (1D)
                new_tokens = drop_invalid_tokens(new_tokens).to(device)
                if new_tokens.numel() > 0:
                    first_piece = emitted_idx == 0
                    if stage is not None and first_piece:
                        _sync()
                        t_decode_end = time.perf_counter()
                        stage["decode"] = t_decode_end - t_prefill_end
                    wav = vocode_chunk(new_tokens,
                                       ctx_tokens=ctx_tail if OPT_CTX_TOKENS else None,
                                       first=first_piece)
                    out = wav.squeeze(0).detach().float().cpu()              # cpu kopyası senkron
                    # parça eki mikro-fade'i (tutma yok -> gecikme eklemez)
                    if _FADE_N and out.numel() > 2 * _FADE_N:
                        if not first_piece:
                            out[:_FADE_N] *= _FADE_IN
                        out[-_FADE_N:] *= _FADE_OUT
                    if stage is not None and first_piece:
                        stage["vocode"] = time.perf_counter() - t_decode_end
                    yield out
                    if OPT_CTX_TOKENS:
                        ctx_tail = (new_tokens if ctx_tail is None
                                    else torch.cat([ctx_tail, new_tokens]))[-OPT_CTX_TOKENS:]
                emitted_idx = len(predicted)
                next_threshold = len(predicted) + chunk

            if is_eos:
                break

            next_embed = t3.speech_emb(next_token) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            next_embed = torch.cat([next_embed, next_embed])                # CFG
            output = patched(inputs_embeds=next_embed, past_key_values=past,
                             output_hidden_states=True, return_dict=True)
            past = output.past_key_values

# ===========================================================================
# 4. TEK CÜMLE ÖLÇÜMÜ (canlı görüşmede bir ajan repliği gibi)
# ===========================================================================
def speak_measure(text, collect=False):
    """Bir cümleyi stream et; TTFB, toplam süre, ses uzunluğu, parça sayısı,
    (collect=True ise) birleşik ses ve ilk parçanın faz bütçesini döndür.
    collect=False = ÖLÇÜM modu: parçalar saklanmaz, kayıt maliyeti sıfır
    (parçalar zaten CPU'ya kopyalanmış geliyor; burada yalnızca sayaç tutulur)."""
    stage = {}
    _sync(); t0 = time.perf_counter()
    ttfb = None
    pieces, n_samples, n_chunks = [], 0, 0
    for wav_chunk in generate_stream(text, stage=stage):
        _sync()
        now = time.perf_counter() - t0
        if ttfb is None:
            ttfb = now                          # <-- İLK ses baytı = TTFB
        n_samples += wav_chunk.shape[-1]
        n_chunks += 1
        if collect:
            pieces.append(wav_chunk)
    total = time.perf_counter() - t0
    audio = torch.cat(pieces).unsqueeze(0) if pieces else None
    audio_s = n_samples / model.sr
    return ttfb, total, audio_s, n_chunks, audio, stage

# ===========================================================================
# 5. ISINMA (ölçüm dışı — ilk CUDA derleme/autotune ortalamayı bozmasın)
# ===========================================================================
print(f"[bilgi] Isınıyor ({WARMUP} tur)...")
for _ in range(WARMUP):
    speak_measure(BENCH_TEXT)

# ===========================================================================
# 6. TTFB ÖLÇÜMÜ (ElevenLabs/Colab ile kıyas — aynı cümle, aynı tur sayısı)
#    Ölçüm turlarında KAYIT YAPILMAZ; örnek kaydı ölçümden SONRA ayrı koşuda.
# ===========================================================================
print(f"\n[TTFB kıyası] cümle: \"{BENCH_TEXT[:48]}...\"  ({RUNS} tur, kayıt kapalı)")
ttfbs, totals, audio_lens, stages = [], [], [], []
for i in range(RUNS):
    ttfb, total, audio_s, nchunks, _audio, stage = speak_measure(BENCH_TEXT)
    ttfbs.append(ttfb); totals.append(total); audio_lens.append(audio_s)
    stages.append(stage)
    print(f"  tur {i+1:2}/{RUNS}: TTFB {ttfb*1000:6.0f} ms | tam {total*1000:6.0f} ms "
          f"| ses {audio_s:.2f} sn | {nchunks} parça")

# --- kalite kaydı: İSTATİSTİĞE DAHİL DEĞİL, ölçüm bittikten sonra 1 ek koşu ---
bench_wav = os.path.join(OUT_DIR, "out_bench.wav")
if SAVE_WAV:
    print("[kalite kaydı] ölçüm bitti; istatistiğe dahil olmayan 1 ek koşuyla örnek alınıyor...")
    q_ttfb, _qt, q_audio_s, _qn, q_audio, _qs = speak_measure(BENCH_TEXT, collect=True)
    if q_audio is not None:
        ta.save(bench_wav, q_audio.cpu(), model.sr)
        print(f"  TTFB {q_ttfb*1000:6.0f} ms (bilgi amaçlı) | ses {q_audio_s:.2f} sn "
              f"-> {bench_wav}")

# ===========================================================================
# 7. CANLI GÖRÜŞME SİMÜLASYONU (ajan cümle cümle konuşuyor)
#    Amaç turn_*.wav örnekleri; TTFB'ler bilgi amaçlı (kıyas bench'i 6. bölüm).
#    Parça toplama CPU kopyaları üzerinde yapılır -> zamanlama yolunu etkilemez.
# ===========================================================================
print(f"\n[canlı görüşme simülasyonu] {len(CONVERSATION)} replik")
for idx, line in enumerate(CONVERSATION, 1):
    ttfb, total, audio_s, nchunks, audio, _stage = speak_measure(line, collect=SAVE_WAV)
    if SAVE_WAV and audio is not None:
        ta.save(os.path.join(OUT_DIR, f"turn_{idx}.wav"), audio.cpu(), model.sr)
    print(f"  replik {idx}: TTFB {ttfb*1000:6.0f} ms | tam {total*1000:6.0f} ms "
          f"| ses {audio_s:.2f} sn -> turn_{idx}.wav")

# ===========================================================================
# 8. RAPOR
# ===========================================================================
def _stage_med_ms(key):
    vals = [s[key] for s in stages if key in s]
    return statistics.median(vals) * 1000 if vals else float("nan")

rtfs = [t / a for t, a in zip(totals, audio_lens) if a > 0]

print("\n" + "=" * 60)
print(f"GPU              : {gpu_name}")
print(f"optimizasyon     : {OPT_DESC}")
print(f"kıyas birimi     : ilk {FIRST_AUDIO_MS} ms ses ({FIRST_CHUNK_TOKENS} token) "
      f"— ElevenLabs/Colab ile AYNI olmalı")
print(f"TTFB   (median)  : {statistics.median(ttfbs)*1000:.0f} ms   <-- ElevenLabs'i GEÇMELİ")
print(f"TTFB   (min)     : {min(ttfbs)*1000:.0f} ms")
print(f"TTFB bütçesi(med): prefill {_stage_med_ms('prefill'):.0f} ms | "
      f"ilk-{FIRST_CHUNK_TOKENS}-token decode {_stage_med_ms('decode'):.0f} ms | "
      f"ilk vocode {_stage_med_ms('vocode'):.0f} ms")
print(f"tam    (median)  : {statistics.median(totals)*1000:.0f} ms")
print(f"RTF    (median)  : {statistics.median(rtfs):.2f}   "
      f"(tam süre / ses süresi; canlı akışın kopmaması için <1.0 ŞART)")
print("=" * 60)
print("KIYAS NOTU: Bu TTFB TAMAMEN yereldir (ağ yok) — saf hesaplama. ElevenLabs")
print("TTFB'si ise ağ+sunucu+model. Local avantajın tam da bu: ağ turu yok.")
print("v3 kalite katmanları REST parçaları pahalılaştırır -> 'tam' ve RTF v2'den")
print("yüksek olabilir; RTF < 1.0 kaldığı sürece canlı akış etkilenmez.")
print("Kaliteyi out_bench.wav (kalite kaydı koşusu) + CER% ile değerlendir.")

# ===========================================================================
# 9. (opsiyonel) TÜRKÇE DOĞRULUK — faster-whisper CER (kaliteyi de kıyasla)
#    Not: kalite kaydı koşusunun çıktısı (out_bench.wav) üzerinden ölçülür.
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
