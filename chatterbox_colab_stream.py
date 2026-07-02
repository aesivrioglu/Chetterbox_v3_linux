"""
============================================================================
Chatterbox Multilingual TTS (v3) — COLAB CANLI-GÖRÜŞME STREAMING scripti
============================================================================

AMAÇ
  Şirket localinde çalışacak, ElevenLabs'in stock modelinin YERİNE geçecek,
  DAHA DÜŞÜK GECİKMELİ bir TTS. Bu script Colab GPU'sunda modeli gerçek bir
  telefon/çağrı ajanı gibi PARÇA PARÇA (streaming) konuşturur ve her cümle
  için TTFB'yi (ilk ses baytı) ölçer.

NEDEN STREAMING (canlı görüşme için şart)
  Stok `model.generate()` AKIŞ YAPMAZ: önce TÜM token'ları üretir, sonra
  vocoder'ı çalıştırır, tek parça döner. Canlı görüşmede müşteri sesin
  BAŞLAMASINI bekler — tamamlanmasını değil. Bu yüzden burada T3 token'larını
  üretildikçe parça parça vocode edip yield ediyoruz: ilk parça hazır olur
  olmaz ses çalmaya başlanabilir = düşük gecikme, tıpkı ElevenLabs stream gibi.

  Teknik cevap: EVET, Chatterbox'ı ElevenLabs gibi canlı müşteriyle konuşturmak
  mümkün. Stok API streaming vermez ama T3 otoregresif üreteç olduğu için
  aşağıdaki gibi araya girip chunk'layabiliyoruz. Prodüksiyonda pürüzsüz akış
  için token örtüşmesi + crossfade önerilir (github.com/davidbrowne17/
  chatterbox-streaming). Buradaki üreteç ÖLÇÜM ve POC için birebir T3 mantığıdır.

DOĞRULUK: Aşağıdaki streaming döngüsü, kurulu paketteki
  chatterbox/models/t3/t3.py -> T3.inference() ile BİREBİR aynıdır (CFG, KV-cache,
  örnekleme sırası: rep-penalty -> temperature -> min_p -> top_p). Örnekleme
  parametreleri de mtl_tts.generate() varsayılanlarıyla aynıdır. Tek fark:
  "yeterince token biriktiyse parçayı vocode et ve yield et" adımı eklenmiştir.

----------------------------------------------------------------------------
COLAB KURULUMU (ilk hücrede bir kez çalıştır — GPU runtime seç: T4/L4/A100)
----------------------------------------------------------------------------
    !pip -q install chatterbox-tts faster-whisper "Levenshtein"
    # Referans sesi (Türkçe, 5-10 sn temiz) yükle:
    from google.colab import files; files.upload()   # -> sayfa-33.wav
----------------------------------------------------------------------------
"""
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
# 0. AYARLAR
# ===========================================================================
LANG    = "tr"
REF_WAV = "sayfa-33.wav"     # Türkçe referans ses (klonlama + aksan sabitleme)

# Ajan preset'i (mtl_tts.generate varsayılanlarıyla uyumlu; sadece exagg/cfg/temp ayarlı)
PRESET = dict(exaggeration=0.4, cfg_weight=0.3, temperature=0.6,
              repetition_penalty=1.2, min_p=0.05, top_p=1.0)

# --- ADİL KIYAS BİRİMİ: "ilk duyulabilir X ms ses" ---
# TTFB'yi ElevenLabs ile AYNI mantıkta ölçmek için ortak birim: "karşı taraf ilk
# kaç ms sesi duymaya başladı". İkisi de AYNI ses süresini hedefler; böylece biri
# 26 ms, diğeri 600 ms ses için ölçülmez. elevenlabs_ttfb.py'deki FIRST_AUDIO_MS
# ile MUTLAKA AYNI değer olmalı. (Ağ/local avantajı KORUNUR, sadece tanım eşitlenir.)
FIRST_AUDIO_MS = 200
# 25 token/sn -> 200 ms ~= 5 token. Chatterbox ilk parçayı bu kadar ses birikince verir.
# NOT: ~120 ms (3 token) altına inme; vocoder çok az token'da kararsızlaşır.
FIRST_CHUNK_TOKENS = max(1, round(FIRST_AUDIO_MS / 1000 * S3_TOKEN_RATE))
CHUNK_TOKENS       = 40      # sonraki parçalar (TTFB'yi etkilemez, akış sürekliliği için)
MAX_NEW_TOKENS     = 1000

# Ölçüm
RUNS     = 10               # elevenlabs_ttfb.py ile AYNI
WARMUP   = 2
SAVE_WAV = True
MEASURE_CER = True          # faster-whisper ile Türkçe doğruluk (kaliteyi de kıyasla)

# Canlı görüşme simülasyonu: ajanın arka arkaya söyleyeceği cümleler.
# Her biri ayrı "tur" gibi ölçülür (gerçek çağrıdaki gibi cümle cümle).
CONVERSATION = [
    "Merhaba, ben Ahmet. Ürün demosu için on dört Temmuz Pazartesi "
    "saat on beş sıfır sıfır'da size randevu oluşturdum, uygun mudur?",
    "Anladım, size uygun başka bir gün ve saat önerebilir miyim?",
    "Harika, o zaman görüşmemizi teyit ediyorum. İyi günler dilerim.",
]
# TTFB kıyas cümlesi (ElevenLabs scriptindeki TEXT ile AYNI olmalı)
BENCH_TEXT = CONVERSATION[0]

# ===========================================================================
# 1. CİHAZ + DTYPE (Ampere+ -> bf16, Turing/T4 -> fp16)
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
    gpu_name = "CPU (GPU yok — Colab'da Runtime > GPU seç!)"
use_autocast = device == "cuda"
print(f"[cihaz] {device} | {gpu_name} | dtype={autocast_dtype if use_autocast else 'fp32'}")

def _sync():
    if device == "cuda":
        torch.cuda.synchronize()

# ===========================================================================
# 2. MODEL + REFERANS (bir kez)
# ===========================================================================
model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
if REF_WAV and os.path.exists(REF_WAV):
    model.prepare_conditionals(REF_WAV, exaggeration=PRESET["exaggeration"])
    print(f"[bilgi] Referans '{REF_WAV}' bir kez gömüldü.")
else:
    print(f"[uyarı] '{REF_WAV}' yok; varsayılan ses (conds.pt) kullanılacak.")

# ===========================================================================
# 3. STREAMING ÜRETEÇ — T3.inference() döngüsü + parça parça vocoder
#    Ses parçalarını (1D CPU tensor) üretildikçe yield eder.
# ===========================================================================
def generate_stream(text, first_chunk=FIRST_CHUNK_TOKENS, chunk=CHUNK_TOKENS):
    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=autocast_dtype, enabled=use_autocast
    ):
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
        top_p_warper = TopPLogitsWarper(top_p=PRESET["top_p"])
        min_p_warper = MinPLogitsWarper(min_p=PRESET["min_p"])
        rep_proc = RepetitionPenaltyLogitsProcessor(penalty=float(PRESET["repetition_penalty"]))

        output = patched(inputs_embeds=inputs_embeds, past_key_values=None,
                         use_cache=True, output_hidden_states=True, return_dict=True)
        past = output.past_key_values

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
            logits = min_p_warper(generated_ids[:1], logits)
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
                    with torch.autocast(device_type=device, enabled=False):  # vocoder fp32 -> ses güvenli
                        wav, _ = model.s3gen.inference(
                            speech_tokens=new_tokens, ref_dict=model.conds.gen)
                    yield wav.squeeze(0).detach().float().cpu()
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
    """Bir cümleyi stream et; TTFB, toplam süre, ses uzunluğu ve birleşik sesi döndür."""
    _sync(); t0 = time.perf_counter()
    ttfb = None
    pieces = []
    for wav_chunk in generate_stream(text):
        _sync()
        now = time.perf_counter() - t0
        if ttfb is None:
            ttfb = now                          # <-- İLK ses baytı = TTFB
        pieces.append(wav_chunk)
    total = time.perf_counter() - t0
    audio = torch.cat(pieces).unsqueeze(0) if pieces else torch.zeros(1, 1)
    audio_s = audio.shape[-1] / model.sr
    return ttfb, total, audio_s, len(pieces), audio

# ===========================================================================
# 5. ISINMA (ölçüm dışı — ilk CUDA derleme/autotune ortalamayı bozmasın)
# ===========================================================================
print(f"[bilgi] Isınıyor ({WARMUP} tur)...")
for _ in range(WARMUP):
    speak_measure(BENCH_TEXT)

# ===========================================================================
# 6. TTFB ÖLÇÜMÜ (ElevenLabs ile kıyas — aynı cümle, aynı tur sayısı)
# ===========================================================================
print(f"\n[TTFB kıyası] cümle: \"{BENCH_TEXT[:48]}...\"  ({RUNS} tur)")
ttfbs, totals, last_audio = [], [], None
for i in range(RUNS):
    ttfb, total, audio_s, nchunks, audio = speak_measure(BENCH_TEXT)
    last_audio = audio
    ttfbs.append(ttfb); totals.append(total)
    print(f"  tur {i+1:2}/{RUNS}: TTFB {ttfb*1000:6.0f} ms | tam {total*1000:6.0f} ms "
          f"| ses {audio_s:.2f} sn | {nchunks} parça")

if SAVE_WAV and last_audio is not None:
    ta.save("out_bench.wav", last_audio.cpu(), model.sr)

# ===========================================================================
# 7. CANLI GÖRÜŞME SİMÜLASYONU (ajan cümle cümle konuşuyor)
# ===========================================================================
print(f"\n[canlı görüşme simülasyonu] {len(CONVERSATION)} replik")
for idx, line in enumerate(CONVERSATION, 1):
    ttfb, total, audio_s, nchunks, audio = speak_measure(line)
    if SAVE_WAV:
        ta.save(f"turn_{idx}.wav", audio.cpu(), model.sr)
    print(f"  replik {idx}: TTFB {ttfb*1000:6.0f} ms | tam {total*1000:6.0f} ms "
          f"| ses {audio_s:.2f} sn -> turn_{idx}.wav")

# ===========================================================================
# 8. RAPOR
# ===========================================================================
print("\n" + "=" * 60)
print(f"GPU              : {gpu_name}")
print(f"kıyas birimi     : ilk {FIRST_AUDIO_MS} ms ses ({FIRST_CHUNK_TOKENS} token) "
      f"— ElevenLabs ile AYNI olmalı")
print(f"TTFB   (median)  : {statistics.median(ttfbs)*1000:.0f} ms   <-- ElevenLabs'i GEÇMELİ")
print(f"TTFB   (min)     : {min(ttfbs)*1000:.0f} ms")
print(f"tam    (median)  : {statistics.median(totals)*1000:.0f} ms")
print("=" * 60)
print("KIYAS NOTU: Bu TTFB TAMAMEN yereldir (ağ yok) — saf hesaplama. ElevenLabs")
print("TTFB'si ise ağ+sunucu+model. Local avantajın tam da bu: ağ turu yok.")

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
        segments, _info = asr.transcribe("out_bench.wav", language=LANG)
        hyp = "".join(seg.text for seg in segments)
        score = cer(BENCH_TEXT, hyp) * 100
        print(f"CER%             : {score:.1f}   (düşük = Türkçe daha net/doğru)")
        print(f"ASR duydu        : {hyp.strip()}")
    except Exception as e:  # noqa: BLE001
        print(f"[uyarı] CER ölçülemedi: {e}")
