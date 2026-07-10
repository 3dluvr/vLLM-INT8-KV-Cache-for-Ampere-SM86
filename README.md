# INT8 KV Cache Support for Ampere (SM80) GPUs


## Summary

Ampere GPUs (SM80) lack native FP8 support, making FP8 KV cache unusable. This patch adds **per-layer INT8 KV cache quantization** support as a practical alternative, reducing memory bandwidth by 2x compared to BF16/F16 with acceptable accuracy loss.

The patch also implements optional FP8-E4M3 emulation for V Cache for extended value range beyond INT8 +/-127.

It was primarily tested with Gemma-4-31B (cyankiwi/gemma-4-31B-it-AWQ-4bit), while limited testing was also done with Qwen3.6-27B (Minachist/Qwen3.6-27B-Mixed-AutoRound), YMMV.


## Motivation

When you desperately want more context than otherwise possible, and are prepared to sacrifice performance for it.


## Inspiration

The original patch for vLLM 0.17.0 was done by Yeb Havinga, at https://github.com/yhavinga/qwen-3.5-27b-dual3090 and https://github.com/yhavinga/vllm-gemma-int8-kv-rtx3090.

To read some excellent background research and documentation on what, why and how, definitely visit the above repositories. This patch is a practical continuation of that work, bringing it to newer vLLM version (0.24.0) while improving a few things here and there.


## Usage

### For vLLM 0.24.0 ONLY

You will need to clone the release/v0.24.0 as this patch was based on that branch. Adjust MAX_JOBS to match your hardware configuration.

```bash
git clone https://github.com/vllm-project/vllm.git -b release/v0.24.0
cd vllm
patch -p1 < /path/to/int8_kv_cache.patch
export MAX_JOBS=48
export CMAKE_BUILD_TYPE=Release
pip install -e . --no-deps
```
Remove any lingering cache:
```bash
rm -rf ~/.cache/vllm/torch_compile_cache/
rm -rf ~/.triton/cache/
```
### Generating Per-Layer Scales

```bash
export VLLM_INT8_V_FP8_EMUL=0
export VLLM_KV_SCALES_RECORD=1
# force it for hybrid models
export VLLM_FORCE_CALC_KV_SCALES=1
  vllm serve <model> \
    --kv-cache-dtype int8 \
    --calculate-kv-scales \
    --served-model-name your_model_name \
    --host 127.0.0.1 \
    --port 8080 \
    --tensor-parallel-size 2 \
    ...
```
During vLLM loading you should see:

```
(EngineCore pid=9180) INFO 07-09 19:55:42 [kv_cache_utils.py:2146] GPU KV cache size: 534,750 tokens
(EngineCore pid=9180) INFO 07-09 19:55:42 [kv_cache_utils.py:2147] Maximum concurrency for 262,144 tokens per request: 2.04x
```

Once it's loaded, run:

```bash
python scripts/calibrate_per_layer_scales.py --url http://127.0.0.1:8080 --model <your_model_name>
```

After the script is done, press CTRL+C once in the terminal window where your vLLM is running, and you should see something like this during shutdown:

```
...
(APIServer pid=9028) INFO:     Shutting down
(APIServer pid=9028) INFO:     Waiting for application shutdown.
(APIServer pid=9028) INFO:     Application shutdown complete.
(Worker_TP1 pid=9195) INFO 07-09 20:13:47 [per_layer_scales.py:214] [Rank 1] Saved per-layer KV scales to: /tmp/vllm_kv_scales_tp1.json
(Worker_TP0 pid=9194) INFO 07-09 20:13:47 [per_layer_scales.py:214] [Rank 0] Saved per-layer KV scales to: /tmp/vllm_kv_scales_tp0.json
(Worker_TP1 pid=9195) INFO 07-09 20:13:47 [per_layer_scales.py:219] [Rank 1] K absmax range: 0.79 - 1.93
(Worker_TP0 pid=9194) INFO 07-09 20:13:47 [per_layer_scales.py:219] [Rank 0] K absmax range: 0.86 - 1.97
(Worker_TP1 pid=9195) INFO 07-09 20:13:47 [per_layer_scales.py:220] [Rank 1] V absmax range: 15.69 - 22.50
(Worker_TP0 pid=9194) INFO 07-09 20:13:47 [per_layer_scales.py:220] [Rank 0] V absmax range: 15.62 - 22.38
(Worker_TP1 pid=9195) INFO 07-09 20:13:47 [per_layer_scales.py:222] [Rank 1] V variance ratio: 1.4x
(Worker_TP0 pid=9194) INFO 07-09 20:13:47 [per_layer_scales.py:222] [Rank 0] V variance ratio: 1.4x
...
```

If you don't see the above, the exit hook did not run the method to save the scales, and you will need to repeat the process (restart vLLM, re-run the calibration script). Make sure you only run the script once, and you only press CTRL+C once.

### Using the Calibrated Scales

It is important that the scales are correctly named (look at the Env variable for VLLM_KV_SCALES_FILE as the root name). It is imperative you do not mix the scales up (tp0->tp1, tp1->tp0) because they need to apply to their respective TP Workers.

```bash
mv /tmp/vllm_kv_scales_tp0.json /path/to/your_model_name_scales_tp0.json
mv /tmp/vllm_kv_scales_tp1.json /path/to/your_model_name_scales_tp1.json

export VLLM_KV_SCALES_FILE=/path/to/your_model_name_scales.json
  vllm serve <model> \
    --kv-cache-dtype int8
    ...
```

### Enabling FP8-E4M3 Emulation for V Cache

If you wish to give FP8-E4M3 emulation a go (it provides a more nuanced V Cache), enable it above when calibrating the scales (change to ```export VLLM_INT8_V_FP8_EMUL=1```), then set it when calling vLLM with calibrated scales:

```bash
export VLLM_INT8_V_FP8_EMUL=1 \
export VLLM_KV_SCALES_FILE=/path/to/your_model_name_scales.json \
  vllm serve <model> \
    --kv-cache-dtype int8 \
    ...
```

Make sure you remove the other calibration Env variables because they are only used during calibration, as well as the ```--calculate-kv-scales``` vLLM argument.

## LICENSE

Yeb Havinga did not specify license on his repos. I presumed MIT and copied his calibration script and data text file here for ease of use.

Otherwise, MIT.
