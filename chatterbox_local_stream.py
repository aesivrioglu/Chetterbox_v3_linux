"""
============================================================================
Chatterbox Multilingual TTS (v3) — YEREL (Linux + RTX/CUDA) STREAMING scripti (v2)
============================================================================

AMAÇ
  chatterbox_colab_stream.py ile BİREBİR AYNI ölçüm mantığı; tek fark ortam:
  bu script arayüzsüz (headless) bir Linux makinede, yerel RTX GPU (ör. RTX 3090)
  üzerinde çalışır. TTFB ve tam süre AYNI birimle (ilk FIRST_AUDIO_MS ms ses)
  ölçülür -> Colab (T4/L4/A100) ve RTX 3090 sonuçları elmayla elma kıyaslanır.

v2 — YAPISAL OPTİMİZASYONLAR (gerekçe ve ayrıntı: README-chatterbox.md)
  T4 stok ölçümü (medyan 1051 ms) TTFB bütçesini kabaca şöyle dağıttı:
  ~%60 s3gen vocode, ~%25 T3 decode, ~%15 prefill. Bu sürüm en büyük kalemi
  hedefleyen ÜÇ yapısal optimizasyon ekler (bölüm 0.5'teki bayraklar):
    1) OPT_REF_TRIM_SEC : s3gen'in HER parçada baştan işleyip attığı 10 sn'lik
       referans prompt'unu 3 sn'ye indirir (CFM dizisi ~510 -> ~170 frame).
    2) OPT_CFM_STEPS    : CFM (flow matching) Euler adım sayısı 10 -> 6.
    3) OPT_FLOW_AUTOCAST: CFM+encoder fp16/bf16 çalışır (HiFT vocoder fp32 kalır).
       RTX 3090 = Ampere -> bf16 seçilir; ayrıca allow_tf32 sayesinde fp32
       matmul/conv'lar da tensor core kullanır (T4'te bu yol YOK).
  KIYAS BİRİMİ DEĞİŞMEDİ: hâlâ "ilk 200 ms sesin hazır olma süresi" ölçülür —
  ElevenLabs tarafıyla birebir AYNI miktar ses için. Optimizasyonlar kaliteyi
  etkileyebileceğinden TTFB'yi HER ZAMAN CER% + out_bench.wav ile birlikte oku.
  Stok davranışa dönüş: üç bayrağı da None/False yap.

ADİL KIYAS
  FIRST_AUDIO_MS, PRESET, CONVERSATION, BENCH_TEXT, RUNS, WARMUP ve
  optimizasyon bayrakları Colab scripti ile AYNI tutulmuştur. Değiştirirsen
  üç tarafta da aynı değeri kullan, yoksa kıyas bozulur.

DOĞRULUK: Streaming döngüsü t3.py -> T3.inference() ile BİREBİR aynıdır.
  Tek istisna: top_p=1.0 iken TopP warper matematiksel KİMLİKTİR ve atlanır;
  örnekleme dağılımı stokla AYNI kalır.

----------------------------------------------------------------------------
KURULUM (Linux, RTX 3090 — bir kez; ayrıntı için requirements-chatterbox.txt)
----------------------------------------------------------------------------
    python3.11 -m venv .venv && source .venv/bin/activate
    pip install -U pip
    pip install -r requirements-chatterbox.txt
    # RTX 3090 (Ampere) için torch 2.6.0'ın CUDA (cu124) wheel'i uyumludur;
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
# 0.5 YAPISAL OPTİMİZASYON BAYRAKLARI (gerekçeler: README-chatterbox.md)
#     Colab scripti ile AYNI değerler — kıyas için. Kıyas birimine DOKUNMAZLAR;
#     yalnızca aynı 200 ms sesin daha hızlı üretilmesini sağlarlar.
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
# 1. CİHAZ + DTYPE (Ampere+ -> bf16, Turing/T4 -> fp16). RTX 3090 = Ampere -> bf16.
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
# 2. MODEL + REFERANS (bir kez)
# ===========================================================================
# t3_model="v3" bazı chatterbox build'lerinde YOK -> imzayı kontrol et, varsa geç.
if "t3_model" in inspect.signature(ChatterboxMultilingualTTS.from_pretrained).parameters:
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
else:
    print("[uyarı] Bu chatterbox build'i t3_model parametresini desteklemiyor; "
          "varsayılan multilingual model yükleniyor.")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
if REF_WAV and os.path.exists(REF_WAV):
    model.prepare_conditionals(REF_WAV, exaggeration=PRESET["exaggeration"])
    print(f"[bilgi] Referans '{REF_WAV}' bir kez gömüldü.")
else:
    print(f"[uyarı] '{REF_WAV}' yok; varsayılan ses (conds.pt) kullanılacak.")

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
_CFM_STEPS = OPT_CFM_STEPS if (OPT_CFM_STEPS and _STEPS_OK) else None
if OPT_FLOW_AUTOCAST and use_autocast and not _HAS_SPLIT:
    print("[uyarı] Bu build'de flow/hift ayrı çağrılamıyor; flow autocast KAPATILDI "
          "(stok fp32 yol kullanılacak).")
if OPT_CFM_STEPS and not _STEPS_OK:
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

REF_DICT = _trimmed_ref_dict(model.conds.gen, OPT_REF_TRIM_SEC)
_ref_sec = REF_DICT["prompt_token"].shape[1] / S3_TOKEN_RATE
_flow_dtype_desc = (("autocast " + ("bf16" if autocast_dtype == torch.bfloat16 else "fp16"))
                    if _FLOW_AUTOCAST else "fp32 (stok)")
OPT_DESC = f"ref {_ref_sec:.1f} sn | CFM {_CFM_STEPS or 10} adım | flow {_flow_dtype_desc}"
print(f"[bilgi] optimizasyonlar: {OPT_DESC}   (stok: 10.0 sn | 10 adım | fp32)")

_TRIM_FADE = getattr(model.s3gen, "trim_fade", None)

def vocode_chunk(new_tokens):
    """Token parçasını sese çevirir. Split yol: CFM+encoder autocast'te,
    HiFT fp32'de — işlem sırası ve trim_fade stok s3gen.inference ile BİREBİR.
    Eski build: stok s3gen.inference (fp32)."""
    if _HAS_SPLIT:
        kw = {"n_cfm_timesteps": _CFM_STEPS} if _CFM_STEPS else {}
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=_FLOW_AUTOCAST):
            mels = model.s3gen.flow_inference(new_tokens, ref_dict=REF_DICT,
                                              finalize=True, **kw)
        with torch.autocast(device_type=device, enabled=False):   # HiFT fp32 -> ses güvenli
            wav, _ = model.s3gen.hift_inference(mels.float())
        if _TRIM_FADE is not None:                                # stok inference ile birebir
            wav[:, :_TRIM_FADE.shape[0]] *= _TRIM_FADE
        return wav
    kw = {"n_cfm_timesteps": _CFM_STEPS} if _CFM_STEPS else {}
    with torch.autocast(device_type=device, enabled=False):       # stok: tamamı fp32
        wav, _ = model.s3gen.inference(speech_tokens=new_tokens, ref_dict=REF_DICT, **kw)
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
                    wav = vocode_chunk(new_tokens)
                    out = wav.squeeze(0).detach().float().cpu()              # cpu kopyası senkron
                    if stage is not None and first_piece:
                        stage["vocode"] = time.perf_counter() - t_decode_end
                    yield out
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
def speak_measure(text):
    """Bir cümleyi stream et; TTFB, toplam süre, ses uzunluğu, birleşik ses ve
    ilk parçanın faz bütçesini (prefill/decode/vocode) döndür."""
    stage = {}
    _sync(); t0 = time.perf_counter()
    ttfb = None
    pieces = []
    for wav_chunk in generate_stream(text, stage=stage):
        _sync()
        now = time.perf_counter() - t0
        if ttfb is None:
            ttfb = now                          # <-- İLK ses baytı = TTFB
        pieces.append(wav_chunk)
    total = time.perf_counter() - t0
    audio = torch.cat(pieces).unsqueeze(0) if pieces else torch.zeros(1, 1)
    audio_s = audio.shape[-1] / model.sr
    return ttfb, total, audio_s, len(pieces), audio, stage

# ===========================================================================
# 5. ISINMA (ölçüm dışı — ilk CUDA derleme/autotune ortalamayı bozmasın)
# ===========================================================================
print(f"[bilgi] Isınıyor ({WARMUP} tur)...")
for _ in range(WARMUP):
    speak_measure(BENCH_TEXT)

# ===========================================================================
# 6. TTFB ÖLÇÜMÜ (ElevenLabs/Colab ile kıyas — aynı cümle, aynı tur sayısı)
# ===========================================================================
print(f"\n[TTFB kıyası] cümle: \"{BENCH_TEXT[:48]}...\"  ({RUNS} tur)")
ttfbs, totals, audio_lens, stages, last_audio = [], [], [], [], None
for i in range(RUNS):
    ttfb, total, audio_s, nchunks, audio, stage = speak_measure(BENCH_TEXT)
    last_audio = audio
    ttfbs.append(ttfb); totals.append(total); audio_lens.append(audio_s)
    stages.append(stage)
    print(f"  tur {i+1:2}/{RUNS}: TTFB {ttfb*1000:6.0f} ms | tam {total*1000:6.0f} ms "
          f"| ses {audio_s:.2f} sn | {nchunks} parça")

bench_wav = os.path.join(OUT_DIR, "out_bench.wav")
if SAVE_WAV and last_audio is not None:
    ta.save(bench_wav, last_audio.cpu(), model.sr)

# ===========================================================================
# 7. CANLI GÖRÜŞME SİMÜLASYONU (ajan cümle cümle konuşuyor)
# ===========================================================================
print(f"\n[canlı görüşme simülasyonu] {len(CONVERSATION)} replik")
for idx, line in enumerate(CONVERSATION, 1):
    ttfb, total, audio_s, nchunks, audio, _stage = speak_measure(line)
    if SAVE_WAV:
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
print("Optimizasyonlar kaliteyi etkileyebilir -> TTFB'yi CER% ve out_bench.wav")
print("ile BİRLİKTE değerlendir (README-chatterbox.md).")

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
