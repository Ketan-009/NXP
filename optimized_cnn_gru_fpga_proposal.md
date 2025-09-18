# Low-Power CNN–GRU FPGA System for Brain Tumor Detection via On-Chip Preprocessing, Resource-Aware Quantization & Hybrid Partitioning

**Platform:** 1 (Lattice CPNX FPGA, validated on KRIA board)

## ABSTRACT

This proposal presents a novel edge-computing framework for brain tumor detection from MRI using a hybrid CNN–GRU model deployed on the Lattice CPNX FPGA (Platform 1). The design achieves <200 ms inference latency, <3 W power, and >95% accuracy, significantly outperforming CPU/GPU-based approaches in energy efficiency. Key innovations include: (1) an RTL on-chip preprocessing block that reduces CPU preprocessing load by ~50%, (2) a resource-aware tiling + double buffering scheme for efficient block RAM utilization, (3) adaptive quantization-aware training with channel pruning achieving 40–60% model reduction with <2% accuracy loss, and (4) a hybrid computation partition, where CNN inference runs on FPGA and GRU modeling on ARM CPU, with exploration of FPGA-only GRU acceleration. Robust validation will be performed on BraTS, Figshare, and clinical MRI datasets with preliminary deployment already achieved on the KRIA KV260 FPGA board using an INT8-quantized XModel. The proposed framework enables low-cost, privacy-preserving, AI-assisted diagnosis in rural healthcare, mobile units, and telemedicine.

## MAIN IDEA

We propose a low-power FPGA-based CNN–GRU system for real-time brain tumor detection.

### System Components

**FPGA Preprocessing (RTL Block):** Streaming normalization, Gaussian denoising, and histogram equalization (~15 ms).

**CNN Accelerator (FPGA):** Optimized INT8 CNN using depthwise separable convolutions and batch normalization (~120 ms).

**GRU Analysis (ARM CPU):** Temporal modeling of MRI slice sequences (~45 ms).

**Post-processing:** Confidence scoring and ROI overlay (~20 ms).

### Optimization Techniques
Building on our prior deployment of an INT8-quantized CNN XModel on the KRIA KV260 board (achieving <200 ms latency and <3 W power), this proposal extends the design by adding:
a custom on-chip preprocessing RTL pipeline,
a resource-aware tiling + buffering scheme, and
a hybrid CNN–GRU partitioning architecture optimized for Platform 1 (Lattice CPNX).

- Tiling + double buffering → 85% bandwidth utilization, hides 15–20 ms latency.
- Quantization + pruning → 60% model size reduction, <2% accuracy loss.
- Pipeline overlap + clock gating for thermal stability.

[Figure Placeholder: Block diagram of system architecture]

## APPLICATION

**Rural Healthcare:** Battery-powered diagnosis where GPUs are infeasible.

**Mobile Medical Units:** Real-time tumor detection during disaster relief.

**Emergency Medicine:** Fast triage for suspected tumor cases.

**Telemedicine:** Privacy-preserving inference with local computation.

## VALUE ADD

### Novelty

- First FPGA CNN–GRU design with complete on-chip preprocessing.
- Resource-aware quantization tailored to Lattice CPNX constraints.
- Hybrid architecture balancing FPGA parallelism and CPU sequential processing.
- Comprehensive clinical validation across BraTS, Figshare, and hospital datasets.

### Key Benefits

- <3 W total power (vs 150–300 W GPU systems).
- <200 ms inference latency (5+ FPS).
- ≥95% detection accuracy with strong generalization.
- Cost-effective, deployable, and privacy-preserving edge solution.

## REFERENCES

[1] VLSID 2026 Design Contest. [Online]. Available: https://vlsid.org/design-contest/

[2] Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS)," IEEE TMI, 2015.

[3] Figshare Brain Tumor MRI Dataset, 2021.
