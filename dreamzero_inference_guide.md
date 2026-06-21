# DreamZero 推理全流程

## 一、服务器环境

- **服务器**: NSCC 广州超算，4× A800 80GB GPU，CUDA 12.8
- **限制**: 计算节点无法访问外网，无法从 HuggingFace/GitHub 下载

## 二、vLLM 0.22.0 离线编译

```bash
# 所有依赖包预下载到 ~/.bihu/pytorch-ng-19195249/
# cutlass-v4.4.2.tar.gz, triton-v3.5.1.tar.gz, vllm-flash-attn-v0.22.tar.gz

# 解压
tar -xzf cutlass-v4.4.2.tar.gz
tar -xzf triton-v3.5.1.tar.gz
tar -xzf vllm-flash-attn-v0.22.tar.gz

# 设置离线源码路径
export VLLM_CUTLASS_SRC_DIR=/XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/cutlass-v4.4.2
export TRITON_KERNELS_SRC_DIR=/XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/triton-v3.5.1
export VLLM_FLASH_ATTN_SRC_DIR=/XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/vllm-flash-attn-v0.22

# MAX_JOBS=8 编译，限制 CUDA 架构跳过 SM100/120 避免 OOM
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0"


# 安装
cd vllm-0.22.0
pip install -e . --no-build-isolation
```
汇总
```bash
cd /XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/vllm-0.22.0
rm -rf build/
MAX_JOBS=8 \
  TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" \
  VLLM_CUTLASS_SRC_DIR=/XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/cutlass-v4.4.2 \
  TRITON_KERNELS_SRC_DIR=/XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/triton-v3.5.1/python/triton_kernels/triton_kernels \
  VLLM_FLASH_ATTN_SRC_DIR=/XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/vllm-flash-attn-v0.22 \
  pip install -e . --no-build-isolation > build.log 2>&1
```
### 遇到的问题

| 问题 | 解决 |
|------|------|
| cmake/utils.cmake 缺失（tar 解压不完整） | `rm -rf` 后重新解压 |
| Ninja OOM（CUTLASS SM100/120 编译） | `TORCH_CUDA_ARCH_LIST` 跳过 SM100/120 |
| `version.py` 语法错误（文件内容只有 "0.22.0"） | 重写 `__version__` / `__version_tuple__` |

## 三、vllm-omni 安装

```bash
cd vllm-omni
pip install -e .
```

## 四、模型文件准备

**模型**: `GEAR-Dreams/DreamZero-DROID`（14B World Action Model）

```bash
# 模型目录结构
/XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/model/
├── model-00001~00010-of-00010.safetensors   # 10个分片，共46GB
├── model.safetensors.index.json
├── config.json
├── tokenizer/             # 从 google/umt5-xxl 提取
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── spiece.model
│   ├── special_tokens_map.json
│   └── generation_config.json
├── scheduler.pt
└── experiment_cfg/
```

### 遇到的问题

| 问题 | 解决 |
|------|------|
| `model-00005` 损坏（1.2GB vs 应为 4.99GB） | 从 HuggingFace 重新下载后验证 195 keys，size 完全匹配 |
| Tokenizer 无法从 HF 下载（无网络） | 离线下载 umt5 tokenizer 放到 `model/tokenizer/` |
| Pipeline 硬编码 `google/umt5-xxl` | Patch `pipeline_dreamzero.py:140` 改为读本地 tokenizer 路径 |

**Tokenizer patch** (`vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py`):

```python
# 原代码
tokenizer_source = od_config.model_paths.get("tokenizer", "google/umt5-xxl")
# 改为
tokenizer_source = od_config.model_paths.get("tokenizer") or os.path.join(model_path, "tokenizer")
```

## 五、示例视频资源

从 `YangshenDeng/vllm-omni-dreamzero-assets` 下载示例视频（3个 MP4），解压到：

```
vllm-omni/outputs/dreamzero/assets/
├── exterior_image_1_left.mp4
├── exterior_image_1_right.mp4
└── ...
```

## 六、运行推理

```bash
cd /XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/vllm-omni

python examples/offline_inference/dreamzero/export_prediction_video.py \
  --deploy-config vllm_omni/deploy/dreamzero_tp1_cfg2.yaml
```
实际
```bash
cd /XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/vllm-omni/examples/offline_inference/dreamzero/
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python export_prediction_video.py \
  --model /XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/model \
  --deploy-config /XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/vllm-omni/vllm_omni/deploy/dreamzero_tp1_cfg2.yaml \
  --video-dir /XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/dreamzero_assets
```

**结果**:

- 模型加载: 10/10 shards, 2146 parameters, 42.79 GiB GPU, ~265s
- 推理耗时: ~23s
- 输出: `outputs/dreamzero/generated_predictions/dreamzero_prediction.mp4`

## 七、输出文件

```
/XYFS01/nsccgz_ywang_wzh/.bihu/pytorch-ng-19195249/vllm-omni/outputs/dreamzero/generated_predictions/dreamzero_prediction.mp4
```
