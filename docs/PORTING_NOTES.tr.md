# GSQ + RCO → ROCm (gfx120x RDNA4 + gfx115x RDNA3.5 APU) port notları

Bu notlar port sürecinde tutulan laboratuvar kaydıdır ve iki ayrı depoyu
kapsar:

* `gsq-rocm` — GSQ'nun ROCm uyarlaması (upstream: IST-DASLab/GSQ)
* `rco-rocm` — RCO'nun ROCm uyarlaması (upstream: IST-DASLab/RCO)

Her iki depoda da `README.md` (kurulum + hızlı test), `docs/VERIFICATION.md`
(doğrulama kaydı) ve `src/rocm_compat.py` (ortak uyumluluk katmanı) bulunur.
Aşağıdaki `runtime/`, `final_summary.json`, `HANDOFF_NEW_SYSTEM.md` gibi
atıflar geliştirme makinesindeki geçici dosyalara işaret eder; bu depolarda
yer almazlar (ağırlık/checkpoint/log commit edilmez).

> **Durum (24 Ekim 2026):** GPU yolu **doğrulandı**. Yeni masaüstünde
> RX 9060 XT (gfx1200, ROCm 7.2.4, torch 2.10.0+rocm7.0) üzerinde GSQ
> 2/3-bit ve RCO 2/3/4-bit uçtan uca çalıştı. Yeni 128×1024 WikiText-2+C4
> kalibrasyon DB'si 42 blok olarak GPU'da tamamlandı ve hold-out'ta eski
> 4×128 README DB'sini belirgin biçimde geçti; kalibrasyon overfit'i
> doğrulandı. Checkpoint yeni DB ile yeniden üretildi: adı "3.286" ama gerçek
> bütçe param-ağırlıklı **3.533 bit**. Tam model search iki bütçede denendi,
> sabit politikayı geçemedi (reddedildi). Ayrıntılar bölüm 5.3–5.4 ve
> `final_summary.json`.
>
> **CPU referansı:** Eski makinede (AMD APU, ROCm yok) portun tamamı CPU'da
> MiniCPM5-2B ile çalıştırıldı; sonuçlar bölüm 5.1'de.
>
> **Taşıma/devir:** Eski makineden devir notu `HANDOFF_NEW_SYSTEM.md`;
> yeni durum ve sıradaki işler orada güncellendi.

---

## 1. Ortam gerçekleri (gfx120x + gfx115x)

| Konu | Not |
|---|---|
| GPU | `gfx1200` = Navi 44 sınıfı (RX 9060 XT / 9060 / 9050). `gfx1201` = Navi 48 (RX 9070 / 9070 XT). |
| ROCm sürümü | gfx1200 desteği ROCm **6.4.2** ile geldi; **ROCm 7.x** önerilir (gfx1200/gfx1201 için hipBLASLt tercihli). |
| PyTorch | CUDA wheel'i AMD GPU'yu görmez. `https://download.pytorch.org/whl/rocm7.0` (veya rocm7.1) kurulmalı. |
| Bellek | 16 GiB sınıfı. Kâğıttaki Llama-70B / Kimi-K2.5 / Qwen3-235B tarifleri bu karta sığmaz. |
| bf16 | RDNA4'te bf16 WMMA var; PyTorch ROCm gfx12'de hipBLASLt kullanır. Kararsızlıkta `TORCH_BLAS_PREFER_HIPBLASLT=0`. |
| xGMI | Tüketici RDNA4'te yok; çoklu GPU testleri PCIe üzerinden ve yavaş. |
| `flash-attn` | ROCm/RDNA4 backend'i yok. |
| Humming / Marlin | CUDA + PTX kernel'leri; HIP portu yok. |
| gfx1150/1151/1152/1153 | RDNA3.5 Ryzen APU'ları (Strix Point / Strix Halo / Krackan Point). |
| gfx1152 | Krackan Point; **APU**, ayrı VRAM yok, GPUVM/GTT ile sistem RAM'i. ROCm **>= 7.2** (7.1.x desteklenmiyor). |
| APU belleği | BIOS VRAM carve-out'u küçük tutun; GTT varsayılanı RAM'in ~%50'si. `amd-ttm --set <GB>` ile yükseltilebilir. `total_memory` carve-out'u gösterebilir. |
| PyTorch wheel | gfx120x → `rocm7.0` (7.1/7.2 de olur); gfx115x → `rocm7.2`. Kurulum script'i `rocminfo` ile seçer. |

Cihaz adı hâlâ PyTorch'ta `"cuda"`'dır; `torch.cuda.*` API'si HIP'e map edilir,
`tensor.is_cuda` HIP tensor'ları için `True` döner. Port bu yüzden çoğu yerde
`"cuda"` yazımını korur; kullanıcıya `rocm` / `hip` alias'ları eklenmiştir.

---

## 2. GSQ-ROCM'de yapılan değişiklikler

### 2.1 Yeni dosyalar

| Dosya | İş |
|---|---|
| `src/rocm_compat.py` | Platform algılama, cihaz normalizasyonu, `gcnArchName`, RNG state yardımcıları, TF32 guard, allocator/dağıtım yardımcıları, Humming kapısı. |
| `scripts/setup_env_rocm.sh` | ROCm torch wheel'i + portatif bağımlılıklar; `uv.lock`/`pyproject` CUDA pinlerini atlar. |
| `requirements-rocm.txt` | torch/flash-attn/humming/vLLM olmadan bağımlılıklar. |
| `configs/rocm/gfx1200_qwen3_0.6b_2bit.yaml` | 16 GiB için güvenli smoke config. |
| `configs/rocm/gfx1200_llama32_1b_2bit.yaml` | Alternatif küçük model. |
| `configs/rocm/gfx1152_minicpm5_2b_2bit.yaml` | MiniCPM5-2B, 1 katman, 2-bit, yerel README korpusu (CPU/APU smoke). |
| `configs/rocm/gfx1152_minicpm5_2b_3bit.yaml` | Aynı smoke test, 3-bit (2.5-3.5 bit bandı). |
| `src/data/dataset.py` `local_text` | HF dataset indirmeden yerel UTF-8 metinden kalibrasyon. |
| `README-ROCM.md` | Kurulum, çalıştırma, limitler. |

### 2.2 Gumbel quantizer'ları (asıl ROCm riski)

Etkilenenler: `gumbel_quantizer_1bit.py`, `_2bit.py`, `_ternary.py`, `_int.py`, `_24.py`.

Orijinal kod, custom autograd `backward` içinde forward'un RNG durumunu
`torch.cuda.get_rng_state` / `set_rng_state` + `torch.random.fork_rng` ile
yeniden oynatıyor. Bu yaklaşım:

* ROCm'de çalışır (HIP RNG state'i vardır) **ama** CPU'da `torch.cuda.*`
  çağrısı patlar;
* `rand_like` doğrudan bf16 üretir; bazı ROCm sürümlerinde yarım hassasiyet
  RNG kernel'i eksik olabilir;
* CUDA ve ROCm RNG dizileri aynı değildir → CUDA ile bit-birebir aynı sonuç
  beklenmemelidir (algoritma aynı, örnekleme farklı).

Port: `rocm_compat.get_rng_state/set_rng_state/fork_rng/rand_like` kullanır.
`rand_like` varsayılan olarak orijinal davranışı (hedef dtype'ta örnekleme)
korur; yarım hassasiyet RNG kernel'i eksik olan bir ROCm build'inde
`GSQ_RNG_FP32=1` ile fp32 örnekleyip cast edebilirsiniz. 24-bit varyantında
ayrıca backward'da forward state'inin geri yüklenmesi eksikti; düzeltildi.

### 2.3 GPTQ (`src/prior/gptq.py`)

* `(W - Q) @ H` işlemi fp32 W ile bf16 H'yi karıştırıyordu. Bazı BLAS
  backend'leri (özellikle hipBLASLt) karışık dtype matmul'ü reddeder.
  Kayıp artık `W.float()`, `Q.float()`, `H.float()` ile hesaplanıyor.
  fp32 H kopyası bir kez önbelleğe alınır (`H_f32`), her Gumbel adımında
  yeniden oluşturulmaz.
* `torch.cuda.empty_cache()` → `rocm.empty_cache()`.
* `torch.linalg.cholesky` / `cholesky_inverse` fp32'de rocSOLVER'a gider;
  gfx1200'de ROCm 6.4+ ile destekli. Büyük katmanlarda Hessian `[K,K]`
  fp32 olduğu için VRAM maliyeti CUDA ile aynı.

### 2.4 Model / eğitim / veri

* `src/models/base.py`: attention backend'i `sdpa` (ROCm'de çalışan tek
  makul seçenek); `GSQ_ATTN_IMPLEMENTATION=eager` ile değiştirilebilir.
  `tensor.is_cuda` filtresi `device.type != "meta"` yapıldı; `empty_cache`
  compat'e taşındı.
* `src/models/*.py`: `torch.cuda.synchronize/empty_cache` çağrıları
  `rocm.synchronize/empty_cache`'e çevrildi (ROCm'de zaten çalışıyordu;
  CPU fallback'i de bozmamak için).
* `src/trainer.py`: synchronize/empty_cache compat.
* `src/data/dataset.py`: broadcast tensörleri `"cuda"` yerine mevcut cihaza
  (yoksa CPU) taşınır; `cuda:0` zorlaması kaldırıldı. `local_text` dataset'i
  eklendi (`data.local_text_path` veya `GSQ_LOCAL_TEXT`); README/metin
  dosyasını `max_length`'lik parçalara böler, eksikse kendini tekrarlar.
* `src/evaluation/wiki_eval.py`: `cuda:0` yerine `torch.cuda.current_device()`.
* `main.py`: cihaz `GSQ_DEVICE` ile seçilebilir (`cuda/rocm/cpu`); RCCL
  `backend="nccl"` adıyla başlatılır (`GSQ_DIST_BACKEND` override); açılışta
  platform/arch özeti basılır. Tüm `torch.cuda.synchronize/empty_cache` ve
  max-memory/log çağrıları `rocm.*` yardımcılarına taşındı; CPU'da no-op.
* `main.py` `get_model_wrapper`: `minicpm` ismi `LLaMAWrapper`'a yönlenir
  (MiniCPM5-2B `LlamaForCausalLM`). `src/models/llama.py` vb. tüm
  `move_embed_to` / `move_output_heads_to` fonksiyonları `device.type !=
  "meta"` kontrolüne geçti; CPU'da embedding/head meta'da kalmıyor.
* `src/models/base.py` `_compress_weight`: paketleme şemasındaki `num_bits`
  artık `quantization.gsq_bits`'ten geliyor. Önceden 4'e sabitlenmişti; bu
  yüzden 2/3-bit çalışmalar 4-bit konteynere paketleniyor, dosya boyutu ve
  metadata gerçek bit genişliğiyle uyuşmuyordu. Karma/ternary modlar için
  tek bir tamsayı genişliği olmadığından varsayılan 4 bırakılıyor.
* `save_model.py` `inject_quantization_config` artık `gsq_bits`'i `num_bits`
  olarak yazıyor; `load_from_disc` de shard'daki packed/scale şekillerinden
  bit genişliğini ve group size'ı çıkarıyor. Böylece 3-bit checkpoint taze
  bir süreçte 4-bit sanılıp yanlış decompress edilmiyor.
* `pin_memory()` çağrıları PPL değerlendirmesinde duruyor; CPU'da hâlâ
  desteklenmiyor, bu yüzden CPU config'lerinde `ppl_eval_every_n_layers: 0`
  ve `eval_baseline: false` kullanılıyor.

### 2.5 Humming / vLLM

* `src/quantization/humming_pack.py` ve `convert_to_humming.py`: ROCm'de
  net bir hata ile erken durur:
  `"Humming kernels are CUDA-only and cannot run on ROCm/gfx1200"`.
  `GSQ_ALLOW_HUMMING=1` yalnızca kernel'lerin HIP portu varsa işe yarar.
* `scripts/serve_model.sh`: `sm_90` (Hopper) kontrolü ROCm'de anlamsızdı;
  `gcnArchName` kontrolü ve "Marlin/Humming WNA16 yok" uyarısı eklendi.
* vLLM: CUDA wheel'i kurulmaz. ROCm vLLM wheel'i compressed-tensors'ı
  listeler, ancak WNA16 fused-MoE/Marlin çekirdekleri CUDA'ya özgüdür.
  Pratikte iki yol: (a) ROCm vLLM dequantize edebiliyorsa serve et,
  (b) `transformers` ile bf16'ya dequantize edip serve et.

### 2.6 Script'ler

* `scripts/_common.sh`: `GSQ_BACKEND=cuda|rocm` algılama; ROCm'de
  `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8` (expandable
  segments kapalı), CUDA'da eski davranış; `detect_num_gpus` artık
  `rocm-smi` ve torch fallback'i biliyor.
* `scripts/setup_env.sh`: ROCm tespitinde `setup_env_rocm.sh`'e devreder.
* `scripts/setup_env_rocm.sh`: `rocminfo` ile gfx'i algılar; `gfx115x` için
  varsayılan `ROCM_WHEEL=rocm7.2`, gfx120x için `rocm7.0`; `ROCM_WHEEL`
  kullanıcı override'ı her zaman kazanır. Doğrulama gfx1150-1153'ü kabul eder
  ve APU/GTT uyarısı basar.
* `scripts/verify_setup.sh` / `verify_smoke_test.py`: HIP sürümü, `gcnArchName`,
  RCCL backend ve flash-attn'in ROCm'de atlanması.

---

## 3. RCO-ROCM'de yapılan değişiklikler

### 3.1 Yeni dosyalar

| Dosya | İş |
|---|---|
| `src/rocm_compat.py` | Cihaz/arch yardımcıları, `control_device()`, TF32 guard, koşullu dist init. |
| `scripts/setup_env_rocm.sh` | ROCm torch + `requirements-rocm.txt`. |
| `scripts/run_quantize_gfx1200_smoke.sh` | Qwen3-0.6B, 32×512 token, 2/3/4-bit hızlı test. |
| `scripts/run_quantize_minicpm5_cpu_smoke.sh` | MiniCPM5-2B, 1 blok, 4×128 token, 2/3/4-bit CPU/APU smoke. |
| `requirements-rocm.txt` | `torch==2.11.0+cu126` yerine portatif set. |
| `README-ROCM.md` | Kurulum ve limitler. |

### 3.2 Çöken/yanlış davranan noktalar

| Yer | Sorun | Çözüm |
|---|---|---|
| `run_quantize.py` | `dist.init_process_group("nccl")` **koşulsuz** çağrılıyordu. Düz `python run_quantize.py` → `environment variable RANK expected`. | `rocm_compat.init_distributed_if_needed()`: yalnızca `WORLD_SIZE>1` ise RCCL başlatır. |
| `run_quantize.py` | `dist.barrier()` init yokken de çağrılıyordu. | `is_dist_available_and_initialized()` guard'ı. |
| `run_quantize.py` | `device = f"cuda:{rank}"`; GPU yoksa/CPU'da patlar. | `resolve_compute_device()`; GPU varsa `cuda:rank`, yoksa CPU. |
| `src/search/quant.py` | `CONTROL_DEVICE = torch.device('cuda:0')` sabit. | `rocm.control_device()`: mevcut GPU, `RCO_CONTROL_DEVICE` ile override. |
| `src/search/prune.py` | `ctrl = torch.device('cuda:0')` sabit. | Aynı. |
| `src/quant/fast_obq.py` | `torch.backends.cuda.matmul.allow_tf32 = False` ve `cudnn.allow_tf32`; RDNA4'te TF32 yok, bazı build'lerde attribute yok. | `rocm.disable_tf32()` (guard'lı). |
| `run_quantize.py` CLI | `flash_attention_2` seçeneği ROCm'de `from_pretrained` içinde patlar. | `rocm.attn_implementation()` → uyarı + `sdpa`. |
| `src/metrics.py`, `src/common.py`, `src/search/*` | `torch.cuda.empty_cache/synchronize` CPU'da patlar. | compat yardımcıları; CPU'da no-op. |
| `scripts/run_quantize.sh` | `NPROC=8` varsayılanı tek GPU'da 8 süreç → her biri tam model → OOM. | GPU sayısından otomatik (`auto`), varsayılan 1. |
| `src/models.py` | Tek-GPU modunda `model.to("cuda")`. | `rocm.resolve_compute_device()`; GPU yoksa CPU. |
| `rco_search_quant.py`, `rco_search_prune.py` | GPU yokken `balanced` device map. | `n_gpus==0` ise tek/CPU moda düşer; arch bilgisi loglanır. |
| `run_quantize.py` | `AutoTokenizer(use_fast=False)` MiniCPM5 tokenizer'ında yavaş sınıf gerektiriyordu. | `use_fast=True` (tokenizer.json yeterli). |
| `run_quantize.py` | `--calibration_samples` / `--calibration_seq_length` yanlış argüman adları script'lerde kullanılıyordu. | `--calibration_tokens` / `--calibration_sequence_length` olarak düzeltildi. |
| `run_quantize.py` | Tüm 42 katmanı quantize etmek CPU'da saatler sürer. | `--max_blocks N` eklendi (ilk N blok; smoke/partial DB). |
| `src/quant/quantizer.py` | Kesilen DB build'i baştan başlıyordu; tamamlanmış bloklar tekrar GPTQ'lanıyordu. | Resume: tamamlanmış blok dizinleri otomatik atlanır; atlanan blok calibration-bitwidth `QLinear` ile geri yüklenip aktivasyonlar ondan geçirilir. Eksik dosyalı yarım blok atlanmaz. |
| `src/data.py` | Sadece `.pt` dosyası ve HF dataset'leri vardı. | `.txt/.md` dosyaları tokenize edilir; `get_data` yol desteği. |
| `rco_search_quant.py` | `--calibration-data` sadece sabit dataset adlarını kabul ediyordu; dosya yolu cache adında `/` üretiyordu. | Serbest metin yolu kabul edilir; cache adı `calibration_tag` ile sanitize edilir. |
| `run_build_checkpoint.py` | `group_default: ["Linear"]` önce geldiği için compressed-tensors her katmanı ona bağlıyor, 4-bit grup için `Could not match ...` uyarıları basıyordu. | Özel bit genişliği grupları `group_default`'tan önce yazılır; 3/4-bit eşleşmesi net. |

### 3.3 RCO'da değişmeyen riskler

* `src/quant/linalg_utils.inv_sym` → `torch.cholesky_inverse(torch.linalg.cholesky(X))`.
  rocSOLVER'da fp32 için destekli; yine de gfx1200'de ilk denemede
  `cholesky_inverse` çağrısını izole test etmek gerekir.
* `FastOBQ.step` içindeki `H[zero_cols, :] = 0` gibi gelişmiş indeksleme
  ATen seviyesinde; ROCm'de aynı semantik. Özel kernel yok.
* `H` her zaman fp32; `qweight` uint8; matmul'ler aynı dtype. Karışık dtype
  matmul yok (GSQ'da vardı, düzeltildi).
* MoE `unfuse_moe_experts` + prune yolu 16 GiB'de pratik değil; algoritma
  portu tamam ama donanım sınırı.

---

## 4. Kurulum (masaüstünde)

```bash
# GSQ
cd gsq-rocm
bash scripts/setup_env_rocm.sh
source .venv-rocm/bin/activate

# RCO
cd ../rco-rocm
bash scripts/setup_env_rocm.sh
source .venv-rocm/bin/activate
```

Kurulum script'i `rocminfo` ile hedefi algılar: `gfx115x` → `rocm7.2`,
`gfx120x` → `rocm7.0`. `ROCM_WHEEL=...`, `PYTHON_VERSION=3.12`,
`VENV_PATH=...` ile override edilebilir. ROCm sürümünüzle eşleşen wheel
etiketini kullanın (`rocm6.4`, `rocm7.0`, `rocm7.1`, `rocm7.2`). gfx1152
(Krackan Point) için **ROCm >= 7.2** gerekir; 7.1.x desteklenmez.

CPU smoke testi için ROCm venv'i gerekmez; ayrı bir CPU venv'i yeterlidir:

```bash
uv venv --python 3.12 .venv-cpu
uv pip install --python .venv-cpu/bin/python \
    --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv-cpu/bin/python \
    transformers==5.7.0 accelerate==1.13.0 datasets==4.8.5 \
    safetensors numpy compressed-tensors==0.15.0.1 lion-pytorch \
    wandb tqdm pyyaml python-dotenv tiktoken matplotlib
```

---

## 5. Doğrulama

### 5.1 CPU doğrulaması (bu makinede tamamlandı)

Bu makinede AMD GPU/ROCm yok; CPU-only torch ile portun tamamı denendi.
Ortam: Python 3.12, `torch==2.14.1+cpu`, `transformers==5.7.0`,
`accelerate==1.13.0`, 16 vCPU, 9 GiB RAM, 15 GiB swap. Model:
`MiniCPM5-2B` (Llama, 42 katman, bf16 ~4.7 GiB).

| Test | Komut (özet) | Sonuç |
|---|---|---|
| GSQ quantizer | `GSQ_DEVICE=cpu python scripts/verify_rocm_quantizer.py` | **ALL OK** (determinism, RNG replay, gradyanlar) |
| GSQ 2-bit, 1 katman | `... main.py --config ..._2bit.yaml --max-layers 1` | 70.0 s; GPTQ ort. loss 526.93; Gumbel val hard 2.64e-04; MLP shard 9.6 MiB |
| GSQ 3-bit, 1 katman | `... main.py --config ..._3bit.yaml --max-layers 1` | 77.8 s; GPTQ ort. loss 131.49; Gumbel val hard 6.99e-05; MLP shard 15 MiB |
| RCO OBQ | `RCO_DEVICE=cpu python scripts/verify_rocm_obq.py` | **ALL OK** (2/4/8-bit, fp32 Hessian, Cholesky) |
| RCO DB 2/3/4-bit, 1 blok | `run_quantize.py ... --max_blocks 1` | 14.9 s; 7 projeksiyon; 410 MB |
| RCO tam DB, 42 blok | `run_quantize.py ... --bitwidth_options 2 3 4` | 704.8 s; 294 projeksiyon; 882 weight + 882 qparams; ~17 GB |
| RCO search, hedef 3.0 | `rco_search_quant.py ... --bitwidths 2,3,4 --target-avg-bits 3.0` | 2b:3 / 3b:1 / 4b:3 = **3.000** ort.; NLL 1.9864; KL 0.010351 |
| RCO search, hedef 4.0 (referans) | aynı, 2/4/8 DB | NLL 1.9813; KL 0.009175 |

Tam 42 blokluk DB üzerinde, ağırlıkları tek tek değiştirip aynı 4×128
token'lık yerel korpusta tam model NLL'i ölçüldü:

| Atama | Ort. bit | NLL |
|---|---|---|
| bf16 referans | 16 | 1.9646 |
| uniform 2-bit | 2.000 | 14.2992 |
| uniform 3-bit | 3.000 | 2.0650 |
| uniform 4-bit | 4.000 | 1.9685 |
| 3-bit + up/down 4-bit | **3.286** | **1.9469** |
| 3-bit + up/down/o 4-bit | 3.429 | 1.9429 |

Seçilen model: **3.286 ortalama bit** (84 projeksiyon 4-bit, 210 projeksiyon
3-bit, 2-bit yok). Q4'ten küçük, bu korpusta uniform Q3 ve Q4'ten daha düşük
NLL. `runtime/minicpm5_2b_q3p286_model` altında compressed-tensors paketli
checkpoint (1.9 GiB) üretildi ve HF `AutoModelForCausalLM` ile CPU'da yüklenip
forward çalıştırıldı. Block-0 RCO desenini tüm katmanlara kopyalamak kötü
sonuç verdi (NLL > 3), bu yüzden tam model için 2-bit kullanılmadı.

Yorum: Tek blok araması 3.0'da 4.0'a çok yakın NLL/KL veriyor; tam modelde
ise 3.286'lık 3/4-bit karışımı uniform Q3/Q4'ü bu küçük korpusta geçiyor.
Sonuçlar 4×128 token'lık README korpusuna aşırı uyumlu olabilir; genel bir
benchmark değildir. Tam model RCO *araması* 9 GiB RAM'e sığmıyor (iki bf16
delta kopyası ~9.4 GB); bu yüzden tam model dağılımı NLL ile sabit seçildi.

CPU testinde bulunup düzeltilenler:

* `torch.cuda.synchronize()` CPU torch'ta `AssertionError` atıyordu → tüm
  çağrılar `rocm.synchronize()` (CPU'da no-op).
* `move_embed_to()` / `move_output_heads_to()` CPU'da embedding'i meta'ya
  atıyordu → `device.type != "meta"` kontrolü.
* MiniCPM ismi `get_model_wrapper`'da eşleşmiyordu → `minicpm` → `LLaMAWrapper`.
* RCO script'leri yanlış CLI adları (`--calibration_samples`,
  `--calibration_seq_length`) kullanıyordu → düzeltildi.
* MiniCPM tokenizer'ı `use_fast=False` ile yüklenemiyordu → `use_fast=True`.
* RCO search, dosya yolu verildiğinde cache adında `/` oluşturuyordu →
  `calibration_tag` sanitize edildi.
* RCO search `--layer-dir` olarak `<save_dir>/<model>/<calib_bw>bit` bekler
  (ör. `runtime/minicpm5_2b_q3_db/MiniCPM5-2B/4bit`), üst dizini değil.
* GSQ checkpoint paketleme `num_bits=4`'e sabitlenmişti; 2/3-bit değerler
  4-bit konteynere yazılıyordu. `quantization.gsq_bits`'ten okunacak şekilde
  düzeltildi; 2-bit shard 9.6 MiB, 3-bit shard 15 MiB (1 katman MLP).

#### 5.1.1 Yeni 128×1024 kalibrasyon DB'si — yarıda durduruldu (22 Ekim 2026)

Kullanıcı isteğiyle uzun DB build'i makine uyumadan/taşınmadan önce temiz
olarak durduruldu. Kullanılan kalibrasyon seti:
`runtime/calib_wikitext2train_c4_128x1024.pt` — 64 WikiText-2 train + 64 C4
chunk, 128×1024 token, `(128, 1, 1024)`.

| Durum | Değer |
|---|---|
| Tamamlanan blok | **1 / 42** (yalnızca `model.layers.0`) |
| Block 0 çıktısı | 7 projeksiyon × 3/4-bit = 28 dosya (`.pth` + `_qparams.pt`), **273 MiB** |
| Block 1 | İşlem sırasında SIGTERM; yazılmış dosya **0** (yarım blok yok) |
| DB dizini | `RCO-ROCM/runtime/minicpm5_2b_q3_wiki_c4_full/MiniCPM5-2B/4bit` |
| Build süresi (block 0) | 134.0 s (1 blokluk ölçüm); tam 42 blok tahmini ~94 dk |
| Resume testi | Aynı dizinle `--max_blocks 1`: **34.96 s**, `Skipping completed block 0/42` |

**Resume davranışı:** `run_quantize.py` / `src/quant/quantizer.py` artık her
blok için tüm istenen bit genişliklerinin hem `N.pth` hem `N_qparams.pt`
dosyalarını kontrol eder. Hepsi varsa ve boş değilse blok atlanır; atlanan
bloğun calibration-bitwidth ağırlıkları `QLinear` olarak geri yüklenir ve
aktivasyonlar o bloktan geçirilir. Böylece sonraki blokların Hessian'ı,
kesintisiz bir koşudakiyle aynı quantize edilmiş aktivasyonlarla toplanır.
Yarım kalmış blok (eksik dosya) atlanmaz; yeniden quantize edilir.

GPU'da hâlâ doğrulanmayanlar: HIP RNG, bf16 WMMA/hipBLASLt, rocSOLVER
Cholesky, RCCL, gerçek bellek/GTT davranışı. Bu yüzden aşağıdaki masaüstü
sırası geçerliliğini koruyor.

### 5.2 Masaüstü / GPU doğrulama sırası

1. **Sürücü/ROCm**
   ```bash
   rocminfo | grep -m1 gfx
   python -c "import torch; print(torch.__version__, torch.version.hip); \
              print(torch.cuda.get_device_properties(0))"
   ```
   Beklenen: `gfx1200` (veya `gfx1201`), `torch.version.hip` dolu.

2. **Kurulum testi** (`setup_env_rocm.sh` sonu) şunları çalıştırır:
   * bf16 512×512 matmul,
   * fp32 `cholesky` + `cholesky_inverse`.
   İkisi de geçmezse quantization'a geçmeyin.

3. **GSQ, riskli yol izole**
   ```bash
   cd GSQ-ROCM
   python scripts/verify_rocm_quantizer.py
   ```
   RNG replay, bf16 softmax ve custom autograd backward'ı model olmadan test eder.

4. **GSQ, 2 katman**
   ```bash
   cd GSQ-ROCM
   export GSQ_RUNTIME="$PWD/runtime"
   python main.py --config configs/rocm/gfx1200_qwen3_0.6b_2bit.yaml --max-layers 2
   ```
   Bakılacaklar: GPTQ Hessian toplama, Gumbel forward/backward, loss'ın
   `NaN/Inf` olmaması, katman checkpoint'inin yazılması.

5. **GSQ, tam küçük model**
   ```bash
   bash scripts/run.sh --config configs/rocm/gfx1200_qwen3_0.6b_2bit.yaml
   python save_model.py --config configs/rocm/gfx1200_qwen3_0.6b_2bit.yaml
   ```
   `save_model.py` compressed-tensors paketlemesi CPU'da çalışır; ROCm'e özel
   kernel kullanmaz.

6. **RCO, GPTQ yolu izole**
   ```bash
   cd RCO-ROCM
   python scripts/verify_rocm_obq.py
   ```

7. **RCO, layer DB**
   ```bash
   cd RCO-ROCM
   bash scripts/run_quantize_gfx1200_smoke.sh
   ```

8. **RCO, arama**
   ```bash
   python rco_search_quant.py \
       --model Qwen/Qwen3-0.6B \
       --layer-dir runtime/qwen3_0.6b_db/Qwen3-0.6B/4bit \
       --bitwidths 2,3,4 --target-avg-bits 3.0 \
       --n-steps 20 --batch-size 1 --batches-per-step 1 \
       --device-map single --save-json
   ```

9. **RNG/backward kontrolü** (opsiyonel ama önerilir): aynı seed ile 2-bit
   quantizer'ı iki kez forward/backward edin; gradient'lerin sonlu ve
   tekrarlanabilir olduğunu doğrulayın. CUDA ile **birebir aynı** değerler
   beklenmemeli.

10. **Bellek**: `rocm-smi --showmeminfo vram` veya
   `torch.cuda.max_memory_allocated()` ile izleyin. 16 GiB'de
   `device_microbatch_size: 1`, `max_length <= 1024`, `calibration_samples`
   küçük tutulmalı.

### 5.3 Masaüstü GPU doğrulaması — tamamlandı (24 Ekim 2026)

Donanım: Ryzen 9 7950X (32 iş parçacığı), 61 GiB RAM, **RX 9060 XT
(gfx1200, 15.9 GiB)**, ROCm **7.2.4**, `torch 2.10.0+rocm7.0`,
`transformers 4.57.1`, `compressed-tensors 0.19.0`.

| Kapı | Sonuç |
|---|---|
| `rocminfo` / `torch.version.hip` | `gfx1200` / `7.0.51831`; `torch.cuda.is_available()=True` |
| bf16 matmul 512×512 | geçti |
| fp32 `cholesky` + `cholesky_inverse` 256×256 | geçti |
| GSQ Gumbel quantizer (`verify_rocm_quantizer.py`) | **ALL OK** (determinism, RNG replay, gradyanlar) |
| RCO OBQ (`verify_rocm_obq.py`) | **ALL OK** (2/4/8-bit, fp32 Hessian, rocSOLVER) |
| GSQ 1 katman 2-bit | 11.8 s; GPTQ ort. 528.05 (CPU 526.93); val hard 2.60e-4 (CPU 2.64e-4) |
| GSQ 1 katman 3-bit | 11.9 s; GPTQ ort. 131.67 (CPU 131.49); val hard 6.91e-5 (CPU 6.99e-5) |
| RCO block 0 | 16.3 s (CPU 134.0 s); 7 projeksiyon × 3/4-bit |
| RCO tam DB (yeni kalibrasyon) | 721.7 s; 882 `.pth` + 882 `_qparams.pt`; ~17 GiB |
| bf16 GPU vs fp32 CPU matmul | relL2 2.9e-3 (bf16 yuvarlaması); fp32 GPU vs CPU 8.5e-7 |

**CPU↔GPU kod eşitliği:** Aynı block-0 için qweight kodlarının
%87.6–99.9'u birebir; ancak aynı 128×1024 aktivasyonları üzerinde GPTQ
rekonstrüksiyon hatası GPU/CPU oranı **0.998–1.005** (7 projeksiyonun tümü).
Yani fark, Hessian toplama sırasının fp32 yuvarlamasından kaynaklanan
eşdeğer çözümlerdir; GPU yolu daha kötü değildir.

**Bulunan ROCm hatası ve düzeltmesi:** İlk tam GPU DB build'i
`model.layers.7.self_attn.q_proj`'da
`linalg.cholesky: input is not positive-definite` ile çöktü. Hessian
sonlu ve fp64'te PD (en küçük özdeğer **+2.49**), fakat fp32 asimetrisi
**4.9e-4**; rocSOLVER bu matrisi reddediyor. Çözüm:

* RCO `src/quant/linalg_utils.inv_sym`: `(X + Xᵀ)/2` simetrileştirme,
  gerekirse fp64 Cholesky fallback'i.
* GSQ `src/prior/gptq.py`: aynı davranışı veren `_safe_cholesky()`.
* Taşınabilir yama: `patches/gsq_safe_cholesky.patch`.

Düzeltmeden sonra 42/42 blok hatasız tamamlandı. Ayrıca
`scripts/eval_holdout_db.py`, model GPU'ya taşındığında hold-out
tensörünü de cihaza taşıyacak şekilde düzeltildi.

**Yeni DB hold-out sonucu** (16×512 WikiText-2 test, 8176 token):

| Atama | Nominal bit (eşit ağırlık) | Param-ağırlıklı bit | Eski DB NLL (CPU) | Yeni DB NLL (GPU) | Eski PPL | Yeni PPL |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 16 | 16 | 3.041096 | 3.041096 | 20.928 | 20.928 |
| uniform 3-bit | 3.000 | 3.000 | 4.667319 | **3.311155** | 106.412 | **27.417** |
| uniform 4-bit | 4.000 | 4.000 | 3.222114 | **3.085127** | 25.081 | **21.870** |
| "3.286" (up/down 4-bit) | 3.286 | **3.533** | 3.701566 | **3.159491** | 40.511 | **23.559** |
| "3.429" (up/down/o 4-bit) | 3.429 | **3.622** | 3.500978 | **3.153620** | 33.148 | **23.421** |

> **Önemli düzeltme:** "3.286" ve "3.429" adları 294 projeksiyon üzerinden
> **eşit ağırlıklı** ortalamadır. Gerçek bütçe, parametre ağırlıklı
> ortalamadır: up/down 4-bit politikası **3.533**, up/down/o 4-bit
> politikası **3.622** bittir. down_proj+up_proj, quantize edilen
> parametrelerin ~%52'sidir ve 4 bittedir. Checkpoint adı `q3p286`
> tarihseldir; dosya boyutu (1.9 GiB) 3.533 bit ile tutarlıdır.

**Karar kuralı uygulandı:** Yeni DB her atamada iyileştiriyor →
kök neden **kalibrasyon overfit'i** (4×128 README korpusu). Yeni DB
"3.286" NLL'i hâlâ bf16'nın üstünde (3.1595 > 3.0411), yani eval leakage
yok. Eski DB'nin GPU eval'i eski CPU tablosunu ~0.006 NLL içinde
doğruladı (`runtime/minicpm5_2b_q3_full/holdout_nll_gpu.json`).

**Yeni checkpoint:** `runtime/minicpm5_2b_q3p286_wiki_c4_model`
(nominal 3.286 / gerçek **3.533** bit, compressed-tensors, 1.9 GiB).
Hold-out NLL **3.159491** — DB ölçümüyle birebir aynı.
`AutoModelForCausalLM` ile CUDA ve CPU'da yüklenip
"The capital of France is Paris ..." üretti.

**Tam WikiText-2 test PPL (orijinal vs checkpoint'ler):** Aynı dosya
(`runtime/wikitext2_test_raw.txt`) üzerinden, üst üste binmeyen 1024-token
pencereleri (277 pencere, 283.371 skorlu token). `scripts/eval_ppl_compare.py`
ile tüm modeller aynı chunk'ları gördü:

| Model (kalibrasyon DB) | Param-ağırlıklı bit | NLL | PPL | Orijinale göre |
|---|---:|---:|---:|---:|
| Orijinal bf16 | 16 | 2.758176 | 15.771 | — |
| up/down 4-bit (128×1024) | 3.533 | 2.868276 | 17.607 | +11.64% |
| attention+up/down 4-bit (128×1024) | 3.733 | 2.820663 | 16.788 | +6.45% |
| attention+up/down 4-bit (256×1024) | 3.733 | 2.817908 | **16.742** | **+6.16%** |
| uniform 4-bit (128×1024) | 4.000 | 2.800656 | 16.455 | +4.34% |

> **Protokol uyarısı:** Daha önce raporlanan **23.56**, tam test PPL'i
> değil, 16×512 hold-out PPL'idir; o protokolde bf16 de 20.93'tür. Aynı
> protokolde kayıp %12.6'dır. 23.56'yı 15.771 ile karşılaştırmak %49 gibi
> yanıltıcı bir kayıp üretir; gerçek tam-test kaybı %11.6'dır.

**Politika taraması (16×512 hold-out, yeni 256 DB):** 2-bit hiç kullanılmıyor
(en iyi atama zaten yalnızca 3/4-bit). Attention projeksiyonlarını 4-bit
yapmak belirgin kazanç sağlıyor:

| Politika | Bit | NLL | PPL |
|---|---:|---:|---:|
| yalnız up/down 4-bit | 3.533 | 3.157534 | 23.513 |
| +k/v 4-bit | 3.556 | 3.129159 | 22.855 |
| +o 4-bit | 3.622 | 3.140900 | 23.125 |
| tüm attention + up/down 4-bit | 3.733 | **3.109589** | **22.412** |
| uniform 3-bit | 3.000 | 3.280822 | 26.598 |
| uniform 4-bit | 4.000 | 3.076321 | 21.679 |

128→256 örnek DB kazancı küçük ama tutarlı (uniform-3 3.3112→3.2808,
attn4 3.1164→3.1096, uniform-4 3.0851→3.0763). Lineer referans:
`minicpm5_2b_attn4_256x1024_model` (1.9 GiB, 3.733 bit, 16.742 PPL);
boyut serbestse `minicpm5_2b_uniform4_wiki_c4_model` (2.0 GiB, 16.455).
**Nihai seçim için §5.5'teki duvar quantization'a bakın:**
`runtime/final_model` = attn4 + int6 duvarlar, 1.281 GiB, 16.796 PPL.

### 5.4 Tam model RCO search (24 Ekim 2026)

Yeni DB ve 61 GiB RAM ile tam model search artık çalıştırılabildi. Güvenlik
için `--cpu-deltas --gradient-checkpointing --vram-fraction 0.8` kullanıldı
(16 GiB ekran kartı; capsiz VRAM tüketimi masaüstünü çökertiyordu). 16×512
token, 100 adım, 4 batch/adım:

| Hedef | Gerçek bit | Search-set NLL | Hold-out NLL | Hold-out PPL | Süre |
|---|---:|---:|---:|---:|---:|
| 3.286 | 3.284 | 3.4347 | 3.355186 | 28.651 | 633 s |
| 3.533 | 3.530 | 3.2756 | 3.200587 | 24.547 | 643 s |
| **Sabit up/down** | **3.533** | — | **3.159491** | **23.559** | — |

**Sonuç:** Aranan atamalar hold-out'ta sabit politikadan **daha kötü**.
3.284 bit araması, 3.000 bit uniform-3'ten bile kötü (3.3552 > 3.3112).
3.530 bit araması sabit politikadan 0.041 NLL geride. Search sonucu
**reddedildi**; mevcut `q3p286` checkpoint korundu. (Search-set NLL'in
hold-out'tan iyi olması, 16×512 kalibrasyon setine overfit olduğunu
gösteriyor.)

**GSQ adil testi** (`configs/rocm/gfx1200_minicpm5_2b_3bit_fair.yaml`,
3-bit, 2 katman, 4 epoch, 28.8 s):

| Katman | Başlangıç val-hard | Bitiş val-hard | Değişim |
|---|---:|---:|---:|
| layers.0 | 7.13e-05 | 6.45e-05 | **−9.5%** |
| layers.1 | 2.78e-03 | 2.48e-03 | **−10.8%** |

Yani GSQ, adil 4-epoch koşusunda her iki katmanda val-hard MSE'yi ~%10
düşürüyor. Bu test CPU-only makinede hiç yapılmamıştı.

---

### 5.5 Duvar quantization (24 Ekim 2026)

`embed_tokens` ve `lm_head` bf16 duvarları 2 × 267M parametre = 1.07 GiB.
`tie_word_embeddings=False` doğrulandı (iki tensör farklı), yani ikisi de
ayrı quantize edilmeli. Normlar bf16 bırakıldı: compressed-tensors'ta RMSNorm
şeması yok ve toplam boyut <1 MiB.

Not: Q8_0/Q6_K/IQ4_NL adları GGUF'a aittir. Burada HF/safetensors karşılığı
grup-bazlı simetrik int8/int6/int4 (group 128) üretildi; gerçek k-quant
linearları da yeniden quantize eder ve RCO atamasını kaybeder.

| Duvar varyantı | Dosya | Tam-test PPL | attn4'e göre |
|---|---:|---:|---:|
| bf16 (attn4 referans) | 1.896 GiB | 16.742 | — |
| int8-g128 ("Q8_0") | 1.406 GiB | 16.742 | +0.00% |
| **int6-g128 ("Q6_K")** | **1.281 GiB** | **16.796** | **+0.32%** |
| int4-g128 ("IQ4_NL") | 1.157 GiB | 17.484 | +4.43% (red) |
| uniform-4 (tavan) | 1.958 GiB | 16.455 | — |

Seçim kuralı (+%2) ile **int6-g128 final**: 1.896 → **1.281 GiB**,
PPL 16.742 → 16.796. int4 %2'yi aştığı için reddedildi. `runtime/final_model`
bu checkpoint'e symlink'tir.

**Stratejik bulgu:** RCO'nun değeri PPL şampiyonluğu değil, boyut/kalite
trade-off'unu yönetmek: attn4 + duvar int8, uniform-4'e yakın kalitede
(16.742 vs 16.455) ~%28 daha küçük dosya veriyor (1.406 vs 1.958 GiB).
int6 ile fark ~%35'e çıkar (16.796 vs 16.455).

GSQ: 4-epoch MSE iyileşmesi pozitif AMA MSE→PPL köprüsü yok; gerçek korpus +
PPL eval hattı kurulursa 30 epoch tekrar denenir. Şimdilik rafa kaldırıldı.

### 5.6 RCO GGUF export + llama.cpp doğrulaması (25 Ekim 2026)

Ayrı GGUF hattı kuruldu; RCO linearları **yeniden quantize edilmedi**.
`scripts/export_rco_gguf_minicpm.py` HF compressed-tensors checkpoint'ini
okuyup 294 RCO tensörünü aynı 3/4-bit kodlarla, duvarları bf16, normları f32
olarak yazıyor. Donör `minicpm5-2b-f16.gguf` sadece metadata/şablon için
kullanıldı; tensörleri kopyalanmadı.

Kritik detay: HF→GGUF dönüşümünde **q/k RoPE permütasyonu** uygulanmalı
(`LlamaModel.permute`; q için 16, k için 2 head). İlk denemede unutulduğu
için PPL 112.2 çıktı; düzeltmeden sonra 15.09.

| Artefakt | Boyut | PPL (16×512, son-256) |
|---|---:|---:|
| Donör f16 (orijinal model) | 4.69 GiB | 13.8451 |
| F16 kontrol (attn4 HF → GGUF, aynı hat) | 4.69 GiB | **15.0641** |
| RCO attn4, bf16 duvar | 1.921 GiB | **15.0875** |
| **RCO attn4, Q6_K duvar** | **1.328 GiB** | **15.1354** |

RCO (bf16 duvar), aynı hattan geçen F16 kontrole göre **+0.155%**; Q6_K
duvarlı final **+0.47%** — ikisi de %0.5 kuralının içinde. HF'nin kendi
PPL'i (attn4 16.3268) llama.cpp ile mutlak olarak karşılaştırılamaz
(framework/protokol farkı ~%11); bu yüzden referans, aynı HF checkpoint'inden
üretilen F16 kontrol GGUF.

Duvar quantization `llama-quantize` ile yapıldı; `--tensor-type-file` 294 RCO
tensörünü kendi tipine (`rco3`/`rco4`) mapliyor, böylece `cur_type ==
new_type` olup **ham kopya** geçiyor; sadece `token_embd.weight` ve
`output.weight` Q6_K'ya iniyor. 294 RCO tensörünün byte'ları girdi GGUF ile
birebir aynı (sha256). 379 tensör C dequant ile HF'ye karşı `max_abs=0`.

Fork'ta tek küçük düzeltme: `ggml_validate_row_data` RCO2/3/4 tiplerini
tanımıyordu ve `llama-quantize` kopya yolunda hata veriyordu; RCO bloklarının
`d`/`z` alanları için bf16 finite kontrolü eklendi (yeni kernel değil).

GPU: `llama-perplexity -ngl 99` ile 42/42 katman ROCm0'a offload; PPL CPU ile
birebir aynı (15.1354), çökme yok.

Komutlar:
```bash
python scripts/export_rco_gguf_minicpm.py \
  --checkpoint runtime/minicpm5_2b_attn4_256x1024_model \
  --donor runtime/gguf/minicpm5-2b-f16.gguf \
  --output runtime/gguf/minicpm5-2b-rco-attn4.gguf
llama-quantize --tensor-type-file runtime/gguf/rco_types.txt \
  runtime/gguf/minicpm5-2b-rco-attn4.gguf \
  runtime/gguf/minicpm5-2b-rco-attn4-q6walls.gguf Q8_0
llama-perplexity -m runtime/gguf/minicpm5-2b-rco-attn4-q6walls.gguf \
  -f runtime/wikitext2_test_raw.txt -c 512 --chunks 16 -ngl 99
```

## 6. Bilinen sınırlar / yapılmayanlar

* **GPU doğrulandı, RCCL hariç**: gfx1200 + ROCm 7.2.4 üzerinde HIP RNG,
  bf16/hipBLAS, rocSOLVER Cholesky ve tek-GPU uçtan uca akış geçti. RCCL /
  çoklu-GPU ve ROCm vLLM servisi **sonraki iş**; şimdi denenmedi.
* **Son parity merdiveni (24 Ekim)**: (a) aynı seed CPU/GPU RNG **birebir
  eşit değil** (farklı torch RNG algoritmaları; korelasyon ~0) — bu literal
  test tasarım gereği geçmez; her backend'in kendi replay'i birebir. (b) tek
  katman GPTQ/OBQ + rocSOLVER potrf **PASS** (`verify_rocm_obq.py` 2/4/8-bit).
  (c) bf16 matmul 2048³ vs CPU fp32: max abs 0.500, relL2 1.66e-3 **PASS**.
  (d) llama-bench + PPL **PASS (25 Ekim)**: ayrı GGUF hattı kuruldu; RCO
  linearları kendi kodlarıyla korundu, duvarlar Q6_K'ya indi. CPU ve
  `-ngl 99` GPU PPL birebir 15.1354; F16 kontrole göre +%0.47. Eski "GGUF
  yok / ölçülemez" notu geçersiz (bkz. 5.6).
* **Duvar quantization GGUF adlarıyla karıştırılmamalı**: `Q8_0/Q6_K/IQ4_NL`
  GGUF k-quant'larıdır. HF finalinde karşılıkları int8/int6/int4-g128'dir.
  GGUF hattında (5.6) duvarlar gerçek Q6_K, RCO linearları ise kendi
  `rco3/rco4` tiplerinde kaldı; normlar f32. HF finalinde normlar bf16
  (RMSNorm şeması yok, <1 MiB).
* **Tam model PPL yapıldı**: RCO 42 blokluk yeni DB, 6 atama için 16×512
  hold-out NLL/PPL ile ölçüldü; 3.533 bit (nominal 3.286) paketli checkpoint
  üretildi ve HF ile CUDA/CPU'da yüklendi. Tam model RCO *search*'ü 3.286 ve
  3.533 hedefleriyle çalıştırıldı; ikisi de sabit politikadan kötü çıktı ve
  reddedildi (bkz. 5.4).
* **Yeni 128×1024 DB tamamlandı**: GPU'da 42/42 blok, 882 `.pth` + 882
  `_qparams.pt`, ~17 GiB, 721.7 s. Hold-out karşılaştırması yapıldı;
  kalibrasyon overfit'i doğrulandı (bkz. 5.3).
* **gfx1152 APU belleği**: ayrı VRAM yok; GTT/TTM ve BIOS carve-out ayarı
  kullanıcıya bağlı. `--max-blocks` / `--max-layers` olmadan büyük model
  denemeyin.
* **16 GiB ekran kartında tam model search**: `--cpu-deltas` tek başına
  yetmiyor; autograd, katman deltalarını backward'a kadar tuttuğu için
  3×5 GB birikiyor. `--gradient-checkpointing` (script artık `model.train()`
  çağırıyor; HF checkpointing eval modunda etkisiz) + `--vram-fraction 0.8`
  kullanın. Cap olmadan 15.9 GiB'i doldurmak X11 kompozitörünü aç bıraktı ve
  masaüstü oturumunu çökertti (`amdgpu ... Failed to pin framebuffer:
  -ENOMEM`). 0.8 cap ile tüm optimize/reload/eval/kaydet hattı ~13 GB'da
  tamamlanıyor; gözlenen tepe VRAM ~7.3 GB. 100 adımlık tam search iki
  hedef için de çalıştırıldı (~633/643 s); sonuçlar bölüm 5.4'te.
* **CPU'da PPL kapalı**: `pin_memory()` CPU-only torch'ta desteklenmiyor;
  CPU config'lerinde PPL değerlendirmesi kapatıldı.
* **Bit-birebir CUDA/CPU eşdeğerliği yok**: RNG (Philox vs HIP), reduction
  sırası, hipBLASLt/rocBLAS ve fp32 Hessian toplama farkları nedeniyle
  sayısal sonuçlar sapabilir. RCO'da CPU ve GPU qweight kodları birebir
  değildir, ancak aynı aktivasyonlarda GPTQ rekonstrüksiyon hatası
  eşdeğerdir (GPU/CPU 0.998–1.005). Algoritma ve veri akışı aynı.
* **Humming / Marlin port edilmedi**: yalnızca net hata mesajı + erken çıkış.
* **vLLM servis**: ROCm wheel'ine bağlı; WNA16 fused-MoE Marlin yok.
  `serve_model.sh` uyarı verip devam eder.
* **Çoklu GPU / MoE dist**: RCCL `nccl` adıyla çalışır; tüketici RDNA4'te
  xGMI yok. `all_to_all_single` RCCL'de var; yine de çoklu kart test
  edilmedi. Askıda kalırsa `NCCL_P2P_DISABLE=1`.
* **CPU fallback kısmi**: GSQ model wrapper'larındaki `torch.cuda` çağrıları
  compat'e taşındı, ancak CPU'da büyük model eğitimi zaten pratik değil.
* **`uv.lock` / `pyproject.toml`**: GSQ tarafında CUDA pinleri duruyor;
  ROCm kurulumu bunları bilinçli olarak kullanmıyor. `uv sync` çalıştırmayın.

---

## 7. Hızlı sorun giderme

| Belirti | Olası neden / çözüm |
|---|---|
| `no kernel image is available` | Wheel gfx1200/gfx1152 içermiyor. ROCm 7.x (gfx1152 için 7.2+) PyTorch wheel kullanın. |
| gfx1152'de OOM / yavaşlık | APU GTT sınırı; BIOS carve-out'u küçültün, `amd-ttm --set <GB>`, `--max-blocks`/`--max-layers` kullanın. |
| `HSA_OVERRIDE_GFX_VERSION` | gfx1152 için son çare `11.5.2`; native ROCm 7.2+ tercih edin. |
| `hipBLASLt ... unsupported` / bozuk bf16 GEMM | `export TORCH_BLAS_PREFER_HIPBLASLT=0` |
| `linalg.cholesky: input is not positive-definite` (Hessian) | FP32 asimetri; `inv_sym` / `_safe_cholesky` zaten simetrileştirip fp64 fallback yapıyor. Daha eski bir ağaçta `patches/gsq_safe_cholesky.patch` uygulayın. |
| GPU görünmüyor | `rocminfo`; `render`/`video` grupları; son çare `HSA_OVERRIDE_GFX_VERSION=12.0.0`. |
| Allocator çökmesi | `expandable_segments` kapalı kalsın (script varsayılanı). |
| Çoklu GPU askı | `NCCL_P2P_DISABLE=1`, `NCCL_DEBUG=INFO`. |
| vLLM checkpoint reddi | WNA16/Marlin CUDA'ya özgü; dequantize edip serve edin. |
| `flash_attn` import hatası | ROCm'de desteklenmiyor; `--attn_implementation sdpa`. |

---

## 8. Dosya envanteri (değişen/eklenen)

GSQ-ROCM:
`src/rocm_compat.py`, `src/quantization/gumbel_quantizer_{1bit,2bit,ternary,int,24}.py`,
`src/prior/gptq.py` (`_safe_cholesky`), `src/trainer.py`, `src/models/base.py`,
`src/models/*.py`, `src/data/dataset.py`, `src/evaluation/wiki_eval.py`,
`src/quantization/humming_pack.py`, `main.py`, `convert_to_humming.py`,
`scripts/_common.sh`, `scripts/setup_env.sh`, `scripts/setup_env_rocm.sh`,
`scripts/verify_setup.sh`, `scripts/verify_smoke_test.py`, `scripts/serve_model.sh`,
`requirements-rocm.txt`, `configs/rocm/gfx1152_minicpm5_2b_{2,3}bit.yaml`,
`configs/rocm/gfx1200_minicpm5_2b_3bit_fair.yaml`, `configs/rocm/*.yaml`,
`scripts/verify_rocm_quantizer.py`, `README-ROCM.md`.

RCO-ROCM:
`src/rocm_compat.py`, `src/quant/fast_obq.py`, `src/quant/quantizer.py`,
`src/quant/linalg_utils.py` (`inv_sym` simetri + fp64 fallback),
`src/search/quant.py`, `src/search/prune.py`, `src/metrics.py`, `src/common.py`,
`src/models.py`, `run_quantize.py`, `rco_search_quant.py`, `rco_search_prune.py`,
`scripts/run_quantize.sh`, `scripts/run_search_quant.sh`,
`scripts/run_search_prune.sh`, `scripts/setup_env_rocm.sh`,
`scripts/run_quantize_gfx1200_smoke.sh`,
`scripts/run_quantize_minicpm5_cpu_smoke.sh`, `scripts/verify_rocm_obq.py`,
`scripts/eval_holdout_db.py` (data→device fix, custom assignment),
`scripts/eval_ppl_compare.py` (çoklu checkpoint PPL), `scripts/eval_policy_sweep.py`,
`scripts/quantize_walls.py` (duvar int8/int6/int4), `src/data.py`,
`requirements-rocm.txt`, `README-ROCM.md`.

Kök: `HANDOFF_NEW_SYSTEM.md` (devir notu), `final_summary.json` (makine-okunur
özet), `patches/gsq_num_bits_loader.patch` (num_bits/loader fix'i),
`patches/gsq_safe_cholesky.patch` (Hessian simetri/fp64 Cholesky fix'i).

Search/eval çıktıları:
`RCO-ROCM/runtime/minicpm5_2b_q3_wiki_c4_gpu_full/holdout_nll_search.json`
(3.284 hedefi), `.../holdout_nll_search3533.json` (3.530 hedefi),
`.../MiniCPM5-2B/4bit/projected_gumbel_kl_bw2-3-4_target3.28{6,533}_...json`
(assignments). `scripts/eval_holdout_db.py` artık 6. argüman olarak bir search
JSON'u alıp o atamayı da ölçüyor ve `avg_bits`'i param-ağırlıklı hesaplıyor.
Tam test PPL: `RCO-ROCM/runtime/ppl_compare_walls.json`; parity:
`RCO-ROCM/runtime/parity_final.json`. Final model:
`RCO-ROCM/runtime/final_model` (attn4 + int6 duvarlar).
