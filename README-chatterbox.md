# Streaming TTFB Kıyası — ElevenLabs vs. Chatterbox vs. XTTS v2

Amaç: ElevenLabs stock TTS'in YERİNE geçecek, şirket localinde **ağsız** çalışan,
**daha düşük TTFB'li** bir Türkçe TTS bulmak. Üç aday da **aynı birimle** —
"ilk `FIRST_AUDIO_MS=200` ms sesin dinleyicinin eline geçme süresi" — ölçülür ki
kıyas elmayla elma olsun.

**Baz çıta (ElevenLabs, ölçüldü 2026-07-02):** `eleven_flash_v2_5`, ADİL TTFB
medyanı **~182-188 ms** (ağ+sunucu+model dahil). Yerel adayların bunu geçmesi hedef.

## Dosyalar
| Dosya | Ortam | Model | Not |
|---|---|---|---|
| `elevenlabs_ttfb.py` | herhangi | ElevenLabs `eleven_flash_v2_5` | Ağ+sunucu+model dahil çıta |
| `chatterbox_colab_stream.ipynb` / `.py` | Colab GPU (T4/L4/A100) | Chatterbox v3 | Colab kurulumu + `files.upload()` |
| `chatterbox_local_stream.py` | Linux + RTX 3090, headless | Chatterbox v3 | `requirements-chatterbox.txt` |
| `xtts_local_stream.py` | Linux + RTX 3090, headless | **XTTS v2 (Coqui)** | `requirements-xtts.txt` (**AYRI venv**) |

Üç script de **aynı** `FIRST_AUDIO_MS=200`, `CONVERSATION`, `BENCH_TEXT`,
`RUNS=10`, `WARMUP=2` ve **aynı referans ses** (`sayfa-33.wav`) kullanır. Birini
değiştirirsen hepsinde değiştir, yoksa kıyas bozulur.

### Kıyas biriminin üç scriptte de aynı olması (neden adil)
"İlk 200 ms ses eline geçti" anı üç tarafta da şöyle damgalanır:
- **ElevenLabs:** gelen mp3 baytları 200 ms'ye çevrilip biriktirilir, eşiği geçince damga.
- **Chatterbox:** ilk parça ~200 ms sese denk gelecek token sayısında (5 token) yield edilir.
- **XTTS v2:** üretilen ses ÖRNEKLERİ biriktirilir, 24 kHz'de 4800 örnek (200 ms) dolunca damga.

Üçü de "dinleyici 200 ms sesi ne zaman eline aldı" sorusunu ölçer → **aynı miktar veri**.

---

## Chatterbox — v2 yapısal optimizasyonlar (neden ve ne değişti)

T4'te ilk stok ölçüm **TTFB medyanı 1051 ms** çıktı (ElevenLabs'in ~6 katı). Faz
bütçesi kabaca: **~%60 s3gen vocode, ~%25 T3 decode, ~%15 prefill**. Sebep senin
tahmininle aynı: (a) zero-shot referansın her parçada baştan işlenmesi, (b) token'ların
tek tek, ağır Python/launch overhead'iyle üretilmesi. v2 üç yapısal optimizasyon ekler;
hepsi **kıyas birimine dokunmadan** aynı 200 ms sesi daha hızlı üretir. Kapatmak için
ilgili bayrağı `None/False` yap (stok davranışa döner).

**1) `OPT_REF_TRIM_SEC` — referans prompt'u 10 sn → 3 sn.**
s3gen'in CFM'i her parçada `prompt_token + parça_token`'ı birlikte işler ve sonunda
referans kısmını **atar** (`flow.py`: `feat[:, :, mel_len1:]`). Yani 10 sn referans (~250
token / ~500 mel frame) her chunk'ta boşuna yeniden üretiliyordu. 3 sn'ye indirince CFM+encoder
dizisi ~510 → ~170 frame; ilk parça ~3× hızlanır. Konuşmacı kimliği (`embedding`, x-vector)
TAM klipten geldiği için ses rengi büyük ölçüde korunur. `model.conds.gen` bozulmaz (sığ kopya).

**2) `OPT_CFM_STEPS` — flow-matching Euler adımı 10 → 6.**
CFM decoder her parça için `n_timesteps` Euler adımı atar; her adım dahili CFG yüzünden
2 estimator geçişi (`flow_matching.py`: `x_in = zeros([2*B, ...])`). Maliyet adım sayısıyla
lineer. 6 adım ~1.7× kazandırır; kaliteyi CER + kulakla doğrula (aşırı düşükte prozodi bozulur).

**3) `OPT_FLOW_AUTOCAST` — CFM+encoder fp16/bf16, HiFT vocoder fp32.**
Stok kod tüm s3gen'i bilinçli fp32 çalıştırıyordu; **T4'te (Turing) fp32'nin tensor core'u
YOK** (8.1 TFLOPS) → en pahalı kısım en yavaş modda koşuyordu. CFM+encoder'ı autocast'e
(T4→fp16, Ampere/3090→bf16) alıp **HiFi-GAN vocoder'ı fp32'de bırakıyoruz** (ses güvenliği).
Ağırlıklar dönüştürülmez (yalnızca autocast) → sürüm-güvenli. RTX 3090'da `allow_tf32=True`
sayesinde fp32 matmul/conv'lar da tensor core kullanır.

**Ek: sürüm-güvenli dağıtım.** `t3_model`, `flow_inference`/`hift_inference` ve
`n_cfm_timesteps` bazı chatterbox build'lerinde YOK. Hepsi imza/`hasattr` ile tespit edilir;
desteklenmeyen optimizasyon uyarı basıp kendini kapatır, script stok yola düşerek yine çalışır.
Böylece Colab'ın eski PyPI build'i ile yerel `.venv` build'i aynı scripti sorunsuz koşar.

**Ek ölçüm: RTF ve faz bütçesi.** Rapor artık her turda ilk parçanın **prefill / decode /
vocode** kırılımını ve **RTF** (tam süre ÷ ses süresi) medyanını basar. Canlı akışın kopmaması
için **RTF < 1.0 şart** (T4 stok koşusunda RTF ≈ 1.6 idi → T4 akışı sürdüremiyordu).

### GPU seçimi (Colab)
| | T4 (ölçüldü) | L4 | A100 40GB | RTX 3090 (hedef) |
|---|---|---|---|---|
| Bellek BW | 320 GB/s | ~300 GB/s (!) | 1555 GB/s | 936 GB/s |
| fp32 / TF32 | 8.1 / yok | 30 / 60 | 19.5 / 156 | 36 / ~70 |
| Stok TTFB | **1051 ms** | ~400-550 | ~300-400 | ~250-350 |

- **L4 tuzağı:** Colab ücretli L4'ünün bant genişliği T4'ten bile düşük; token decode hiç
  hızlanmaz. Paran L4'e gitmesin.
- 5× GPU ≠ 5× hız: token-başı Python overhead'i ve stok kodun full-ref CFM'i her GPU'da durur.
  **Stok kodla hiçbir GPU 188 ms'yi geçemez** → çıtayı geçmek yazılım optimizasyonu ister
  (yukarıdaki 1-3). Optimizasyonlar yapılınca asıl hedef RTX 3090 zaten yeterli.

---

## XTTS v2 (alternatif aday) — neden yapısal olarak daha hızlı olmalı

Senin Chatterbox için doğru teşhisin (referansın sürekli yeniden işlenmesi + sıralı token)
XTTS v2'de **yapısal olarak** çözülü:

- **Referans BİR KEZ işlenir.** `get_conditioning_latents()` referans sesi başlangıçta
  `gpt_cond_latent` + `speaker_embedding`'e çevirir; her istekte SADECE bu hazır latent'ler
  kullanılır — referans **tekrar işlenmez**. (Chatterbox'ta her chunk'ta yeniden üretilip atılan
  maliyetin karşılığı burada YOK.) Script bunu latent önbelleğiyle (`.pt`) tekrar koşularda anında yapar.
- **Native streaming.** `inference_stream`, GPT `stream_chunk_size` token biriktirir biriktirmez
  HiFi-GAN ile decode edip parça yield eder → "ilk ses"e kadar süre kısa.

**TTFB'yi düşüren asıl kaldıraç `STREAM_CHUNK_SIZE`.** Küçük = düşük TTFB, ama <~10'da parça
sınırında ses bozulabilir. Rapor "ilk parça ~kaç ms ses" değerini basar: >200 ms ise TTFB'yi
düşürmek için bu değeri azalt. Referans optimizasyonu (`REF_GPT_COND_SEC=6`) TTFB'yi doğrudan
etkilemez (bir kezliktir) ama kurulum maliyetini düşürür.

### ⚠️ Lisans (üretim kararı için kritik)
- **XTTS v2 → Coqui Public Model License (CPML): TİCARİ KULLANIMA KAPALI.** Kıyas/POC için
  uygundur; ElevenLabs YERİNE ticari üretimde kullanılamaz.
- **Chatterbox → MIT: ticari kullanıma açık.** Üretim hedefi ticariyse asıl aday Chatterbox'tır;
  XTTS v2 yalnızca "ne kadar hızlı olabilir" referansı olarak ölçülür.

### Çalıştırma (Linux + RTX 3090, headless) — **AYRI venv**
```bash
# XTTS'in torch/transformers pin'leri chatterbox ile ÇAKIŞIR -> ayrı ortam:
python3.11 -m venv .venv-xtts && source .venv-xtts/bin/activate
pip install -U pip
pip install -r requirements-xtts.txt
# Referans sesi koy (sayfa-33.wav) ve çalıştır (ilk sefer ~1.8 GB model indirir):
REF_WAV=/yol/sayfa-33.wav OUT_DIR=out python xtts_local_stream.py
```
`COQUI_TOS_AGREED=1` script içinde ayarlı (headless'ta interaktif lisans sorusu olamaz).

### Daha da hızlandırmak istersen (opsiyonel, script dışı kaldıraçlar)
- **DeepSpeed:** XTTS GPT'yi 2-3× hızlandırır ama düşük seviye yükleme + CUDA derleyici ister
  (headless kutuda kurulum riskli). Gerekirse `Xtts.load_checkpoint(..., use_deepspeed=True)`.
- **fp16:** GPT'yi yarıya indirir ama latent dtype'larını da eşlemek gerekir; sesde bozulma
  riski → önce CER ile doğrula.

---

## Linux + RTX 3090 (Chatterbox, yerel)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements-chatterbox.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python chatterbox_local_stream.py
#   opsiyonel:  REF_WAV=/yol/ref.wav OUT_DIR=out python chatterbox_local_stream.py
```
RTX 3090 Ampere → script otomatik **bf16** autocast seçer; CFM/encoder tensor core kullanır.

## Çıktı (her iki yerel script)
- Konsol: her tur TTFB / (XTTS'te ham ilk-parça) / tam süre / ses uzunluğu, medyan+min özet,
  faz bütçesi (Chatterbox), **RTF medyanı**, (opsiyonel) **CER%** + ASR'ın duyduğu metin.
- Dosya: `out_bench*.wav`, `turn_*.wav` (`OUT_DIR` içine).

**CER notu:** faster-whisper "on dört Temmuz"u "14 Temmuz", "on beş sıfır sıfır"ı "15.00"
diye normalize eder; bu, CER%'i olduğundan yüksek gösterir (fonetik metin doğru olsa bile).
Gerçek doğruluk için sayı/saat normalizasyonu ekleyip öyle kıyasla.

Hedef özet: her iki yerel adayın da TTFB medyanı ElevenLabs'in **~182 ms**'sini geçmeli;
asıl soru RTX 3090'da ne kadar altına inildiği ve bunu yaparken **RTF < 1.0** + kabul edilebilir
CER'in korunup korunmadığı.
