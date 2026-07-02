# UVM Enabled PyTorch

- Motivation

UVM (Unified Virtual Memory) is an NVIDIA GPU memory management mechanism that has been available since the Volta architecture, yet it remains underutilized in many practical applications. Currently, PyTorch relies heavily on cudaMemcpy for data management, which can be a bottleneck.

To address this, I have implemented a UVM-integrated PyTorch framework to enable more efficient memory management.

```
@article{park2026pytorch_uvm,
  title={An Empirical Study of LLM Serving in Confidential GPUs},
  author={Park, Eunseong and Xiong, Wenjie},
  journal={ISPASS 2026},
  year={2026}
}
```



- Check this out

For scripts related to LLM inference, please refer to the following repository: Confidential-GPU-LLM-Serving-Performance-Profiling [link] https://github.com/bearhw/Confidential-GPU-LLM-Serving-Performance-Profiling

This repository contains a comprehensive set of LLM inference scripts. You may find them useful as a reference when running your experiments or evaluating performance.


