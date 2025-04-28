# 中英文神经机器翻译系统

基于minGPT的中文到英文(Chinese-to-English)神经机器翻译系统，使用小型GPT模型实现序列到序列的翻译任务。

## 功能概述

`chinese_english_translator.py`实现了一个完整的中英文翻译流水线，包括：

- 加载中英文平行语料库并进行预处理
- 训练一个小型GPT模型(gpt-nano)进行翻译
- 实现训练过程可视化(损失曲线)
- 翻译结果表格展示与评估
- 模型注意力机制可视化

## 环境要求

本项目依赖于minGPT框架，需要以下环境：

- Python 3.6+
- PyTorch 1.7+
- matplotlib
- seaborn
- pandas
- minGPT库（需要先克隆并安装）

## 安装指南

1. 克隆minGPT仓库：
```bash
git clone https://github.com/karpathy/minGPT.git
cd minGPT
```

2. 安装minGPT及依赖：
```bash
pip install -e .
pip install matplotlib seaborn pandas
```

3. 将`chinese_english_translator.py`放在minGPT目录下

4. 确保数据文件`cmn.txt`位于正确位置（默认在当前目录或`cmn-eng/`目录下）

## 使用方法

直接运行脚本启动训练和评估过程：

```bash
python chinese_english_translator.py
```

可选参数可以通过修改代码中的配置对象设置，如：
- 模型大小：更改`config.model.model_type`（默认为'gpt-nano'）
- 训练轮数：更改`max_epochs`和`max_iters`变量
- 学习率：更改`config.trainer.learning_rate`

## 数据格式

输入数据文件（cmn.txt）格式为制表符分隔的英文和中文对应句子：
```
English sentence    中文句子    其他信息(可选)
```

## 数据处理

系统对数据的处理流程如下：

1. **数据加载与划分**：
   - 从cmn.txt读取中英文对照语料
   - 按9:1比例自动划分为训练集和验证集
   - 支持从命令行参数指定其他数据路径

2. **字符级处理**：
   - 系统采用字符级别处理，不进行额外分词
   - 对所有出现的字符创建词汇表（vocabulary）
   - 为每个字符分配唯一的整数ID

3. **特殊标记处理**：
   - `<PAD>`：用于序列填充，使批处理中所有序列长度相同
   - `< SOS >`：序列起始标记，表示翻译开始
   - `<SEP>`：分隔符，用于分隔中文输入和英文输出
   - `<EOS>`：序列结束标记，表示翻译结束

4. **序列构建**：
   - 每个翻译对被构建为特定格式：`< SOS > 中文文本 <SEP> 英文文本 <EOS>`
   - 将字符序列转换为对应整数索引序列
   - 截断超过最大长度(block_size)的序列

5. **训练目标生成**：
   - 输入(X)为：整个序列去掉最后一个字符
   - 目标(Y)为：整个序列去掉第一个字符
   - 使用掩码机制，只计算`<SEP>`后部分（英文部分）的损失
   - 在批处理中对序列进行填充（padding）处理

6. **批处理处理**：
   - 使用自定义`collate_batch`函数处理不同长度序列
   - 使用`<PAD>`标记填充每个批次中的序列到相同长度
   - 为损失计算创建掩码，忽略填充部分

这种数据处理方式使模型能够学习根据中文输入生成对应的英文翻译，同时处理了变长序列和批处理的技术挑战。

## 输出结果

运行后，程序将在`./out/chinese_english_translator/`目录下生成以下内容：

- `model.pt`：训练好的模型权重
- `loss_curve.png`：训练损失曲线图
- `translation_results.csv`：翻译结果表格（CSV格式）
- `translation_results.png`：翻译结果表格（图片格式）
- `attention_maps/`：注意力可视化图像（如果成功）

## 技术实现

本项目使用了以下技术：

- **模型架构**：基于GPT（生成预训练Transformer）的自回归语言模型
- **特殊标记**：使用`< SOS >`、`<SEP>`、`<EOS>`标记来分隔输入和输出序列
- **批处理**：自定义批处理逻辑以处理不同长度的序列
- **掩码机制**：在损失计算中使用掩码，只关注从中文生成英文的部分

## 可能的改进

- 使用更大的模型如`gpt-mini`或`gpt-micro`提高翻译质量
- 添加中文分词预处理
- 实现基于Beam Search的解码策略
- 增加BLEU等评估指标
- 修复注意力可视化功能
- 添加对中文字体的支持以优化可视化效果

## 许可证

本项目遵循与原始minGPT库相同的MIT许可。 
