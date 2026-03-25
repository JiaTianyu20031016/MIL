1. 检查buffer baseline的实现，并跑一个版本的模型，with last step scalar = 3
2. 跑一个 DPO baseline， with beta=0.05，global batch size 保持为 384
3. 借鉴 ImplicitPRM 论文的代码，完成best of N实验
4. 探索 PUL