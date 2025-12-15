# 基于非配准跨域频空先验迁移的SAR图像分割
## SIDAR_GAN的详细网络结构
<img width="978" height="528" alt="image" src="https://github.com/user-attachments/assets/9ba28ac9-a437-4cca-8f26-a5f28dc30450" />
该框架通过融合频域与空间结构解耦、方向性建模与跨模态语义引导三种策略，旨在提升网络在复杂散射背景下的结构建模能力与关键目标感知能力，在强干扰环境下增强关键目标区域感知。 
## 频-空扰度解析模块图
<img width="795" height="714" alt="image" src="https://github.com/user-attachments/assets/0a63f1e0-41b6-4277-b013-e1343bde9280" />

## 双域结构对齐模块图
<img width="429" height="264" alt="image" src="https://github.com/user-attachments/assets/ddb11319-a3e1-47da-9105-330b159ba982" />
# SAR数据集

SARBuD与HRSID两个数据集

# 实现细节
为保证各方法比较的可重复性与公平性，所有网络均在统一的数据预处理与训练配置下进行评估。输入图像先经过一致的归一化与尺寸调整，随后施加随机裁剪、随机强度扰动以及多种仿射变换（如旋转、平移、缩放等）以增强数据多样性。初始学习率设为0.0001，采用AdamW优化器，网络训练了100轮次，训练时的批大小统一为2。全部实验在同一工作站上完成，该平台配备2块 NVIDIA RTX 3090 GPU、AMD Ryzen 9 5950X 处理器及128 GB内存，软件环境为 Python 3.10.0、PyTorch 2.0.0 与 CUDA 11.8。

# 性能比较
<img width="746" height="1054" alt="image" src="https://github.com/user-attachments/assets/3bebe64b-a4c9-4531-8b9d-814f79650a1b" />

<img width="871" height="737" alt="image" src="https://github.com/user-attachments/assets/80c5543b-4129-4820-aab7-b2489e8eda6a" />
