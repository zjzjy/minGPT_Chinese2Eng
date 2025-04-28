"""
使用minGPT实现一个中文到英文的神经机器翻译系统
包含训练和评估过程、损失曲线可视化、结果表格展示和注意力可视化
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.font_manager import FontProperties
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():
    """
    获取配置信息
    """
    C = CN()

    # 系统设置
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chinese_english_translator'

    # 数据设置
    C.data = TranslationDataset.get_default_config()

    # 模型设置
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'  # 使用最小的GPT模型

    # 训练设置
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4
    C.trainer.max_iters = 2000

    return C

# -----------------------------------------------------------------------------

class TranslationDataset(Dataset):
    """
    处理中英文翻译数据的数据集类
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 256  # 序列最大长度
        C.data_path = 'cmn.txt'
        return C

    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.block_size = config.block_size
        
        # 特殊标记定义
        self.special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<SEP>']
        
        # 加载翻译数据
        self.english_texts = []
        self.chinese_texts = []
        
        print(f"正在加载数据集: {config.data_path}")
        with open(config.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    eng = parts[0].strip()
                    cmn = parts[1].strip()
                    self.english_texts.append(eng)
                    self.chinese_texts.append(cmn)
        
        data_size = len(self.english_texts)
        print(f'加载了 {data_size} 条中英文翻译对')
        
        # 先创建字符集
        chars = set()
        for text in self.english_texts + self.chinese_texts:
            chars.update(text)
        
        # 将字符排序形成列表
        self.chars = sorted(list(chars))
        
        # 将特殊标记添加到词汇表的开头
        self.vocab = self.special_tokens + self.chars
        self.vocab_size = len(self.vocab)
        print(f'词汇表大小: {self.vocab_size} 个字符')
        
        # 创建字符到索引的映射
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}
        
        # 获取PAD的索引用于填充
        self.pad_idx = self.stoi['<PAD>']
        
        # 分割训练集和验证集 (9:1)
        if split == 'train':
            self.indices = list(range(int(data_size * 0.9)))  # 使用90%的数据作为训练集
        else:
            self.indices = list(range(int(data_size * 0.9), data_size))  # 使用10%的数据作为验证集

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 获取实际索引
        real_idx = self.indices[idx]
        
        # 准备中文(源)和英文(目标)文本
        chinese_text = self.chinese_texts[real_idx]
        english_text = self.english_texts[real_idx]
        
        # 将文本转换为序列形式
        # 我们将分别处理中文和英文文本，然后加入特殊标记
        
        # 准备token序列
        tokens = []
        tokens.append('<SOS>')  # 起始标记
        
        # 添加中文字符
        for char in chinese_text:
            tokens.append(char)
        
        tokens.append('<SEP>')  # 分隔标记
        
        # 添加英文字符
        for char in english_text:
            tokens.append(char)
        
        tokens.append('<EOS>')  # 结束标记
        
        # 将token转换为索引
        input_ids = [self.stoi[token] for token in tokens]
        
        # 确保长度不超过block_size
        if len(input_ids) > self.block_size:
            input_ids = input_ids[:self.block_size]
        
        # 创建输入(x)和目标(y)张量，根据GPT的设计，目标是输入向右移动一位
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(input_ids[1:], dtype=torch.long)
        
        # 找到<SEP>在tokens中的位置，这之前的部分是中文输入，我们不计算损失
        sep_idx = tokens.index('<SEP>') if '<SEP>' in tokens else -1
        if sep_idx != -1 and sep_idx < len(x):
            # 创建一个掩码，只在<SEP>之后的位置计算损失
            mask = torch.ones_like(y, dtype=torch.long) * -1
            mask[sep_idx:] = y[sep_idx:]
            y = mask
        
        return x, y

# 自定义collate_fn,用于处理不同长度的序列
def collate_batch(batch):
    # batch是一个列表，包含多个元组(x, y)
    # 找到最长的序列长度
    max_x_len = max([x.size(0) for x, _ in batch])
    max_y_len = max([y.size(0) for _, y in batch])
    
    # 创建填充后的批次
    x_batch = []
    y_batch = []
    
    # 获取PAD索引值
    pad_idx = 0  # 默认值
    if hasattr(batch[0][0], 'dataset') and hasattr(batch[0][0].dataset, 'pad_idx'):
        pad_idx = int(batch[0][0].dataset.pad_idx)  # 转换为整数标量
    
    for x, y in batch:
        # 填充x到最长长度
        x_padded = torch.cat([
            x,
            torch.full((max_x_len - x.size(0),), pad_idx, dtype=torch.long)
        ], dim=0)
        x_batch.append(x_padded)
        
        # 填充y到最长长度，使用-1保持损失计算一致性
        y_padded = torch.cat([
            y,
            torch.full((max_y_len - y.size(0),), -1, dtype=torch.long)
        ], dim=0)
        y_batch.append(y_padded)
    
    # 堆叠为批次
    return torch.stack(x_batch), torch.stack(y_batch)

# -----------------------------------------------------------------------------

def translate(model, dataset, chinese_text, device='cpu', max_len=100):
    """
    使用训练好的模型进行翻译
    """
    model.eval()
    
    # 准备输入序列
    tokens = ['<SOS>']
    for char in chinese_text:
        tokens.append(char)
    tokens.append('<SEP>')
    
    # 转换为索引
    input_ids = [dataset.stoi[token] for token in tokens]
    x = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # 生成翻译
    with torch.no_grad():
        output_ids = model.generate(x, max_new_tokens=max_len, temperature=0.7, do_sample=True)[0]
    
    # 将输出ids转换回token
    output_tokens = [dataset.itos[int(i)] for i in output_ids]
    
    # 找到<SEP>和<EOS>的位置
    try:
        sep_idx = output_tokens.index('<SEP>')
    except ValueError:
        sep_idx = -1
    
    try:
        eos_idx = output_tokens.index('<EOS>')
    except ValueError:
        eos_idx = len(output_tokens)
    
    # 提取翻译结果 (位于<SEP>和<EOS>之间的文本)
    if sep_idx != -1 and sep_idx < eos_idx:
        translated_tokens = output_tokens[sep_idx+1:eos_idx]
        translated = ''.join(translated_tokens)
    else:
        # 如果没有找到标记，返回所有输出
        translated = ''.join([t for t in output_tokens if t not in dataset.special_tokens])
    
    return translated

# -----------------------------------------------------------------------------

def plot_loss_curve(losses, save_path):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def create_result_table(examples, ground_truth, translations, save_path):
    """创建翻译结果表格并保存为CSV和图片"""
    # 创建DataFrame
    df = pd.DataFrame({
        'Chinese Input': examples,
        'Model Translation': translations,
        'Reference Translation': ground_truth
    })
    
    # 保存为CSV
    df.to_csv(save_path + '.csv', index=False, encoding='utf-8')
    
    # 创建表格图像
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    # 创建表格
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.3, 0.3]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title('Translation Results Comparison', fontsize=16)
    plt.tight_layout()
    
    # 保存表格为图片
    plt.savefig(save_path + '.png', bbox_inches='tight', dpi=200)
    plt.close()
    
    return df

def visualize_attention(input_text, attention_maps, dataset, save_dir):
    """可视化注意力权重"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 处理输入文本，提取标记
    tokens = ['<SOS>']
    for char in input_text:
        tokens.append(char)
    tokens.append('<SEP>')
    
    # 绘制每一层的注意力图
    for layer_idx, attn_weights in attention_maps:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 取第一个头部的注意力
        head_idx = 0  # 使用第一个注意力头
        
        # 安全地获取注意力矩阵维度
        if len(attn_weights.shape) < 3:
            print(f"跳过层 {layer_idx+1}，注意力权重维度不匹配: {attn_weights.shape}")
            continue
            
        # 获取注意力矩阵，限制为输入序列的长度
        if len(attn_weights.shape) == 4:  # 标准形状 [batch, head, seq_len, seq_len]
            seq_len = min(len(tokens), attn_weights.shape[2])
            attn_map = attn_weights[0, head_idx, :seq_len, :seq_len]
        elif len(attn_weights.shape) == 3:  # 可能的形状 [batch, seq_len, seq_len]
            seq_len = min(len(tokens), attn_weights.shape[1])
            attn_map = attn_weights[0, :seq_len, :seq_len]
        else:
            print(f"无法处理层 {layer_idx+1} 的注意力权重，形状为: {attn_weights.shape}")
            continue
        
        # 创建热力图
        try:
            sns.heatmap(
                attn_map,
                annot=False,
                cmap='viridis',
                xticklabels=tokens[:seq_len],
                yticklabels=tokens[:seq_len],
                ax=ax
            )
            
            plt.title(f"Layer {layer_idx+1} - Attention Head {head_idx+1}")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/attention_layer{layer_idx+1}_head{head_idx+1}.png")
        except Exception as e:
            print(f"绘制层 {layer_idx+1} 的注意力图时出错: {e}")
        finally:
            plt.close()

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # 获取配置并设置随机种子
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    set_seed(config.system.seed)
    
    # 创建输出目录
    os.makedirs(config.system.work_dir, exist_ok=True)
    
    # 构建训练和验证数据集
    train_dataset = TranslationDataset(config.data, split='train')
    val_dataset = TranslationDataset(config.data, split='val')
    
    # 设置模型参数
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    
    # 构建自定义的DataLoader，使用我们的collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.num_workers,
        collate_fn=collate_batch,
        pin_memory=True
    )
    
    # 构建训练器，使用自定义DataLoader
    # config.trainer.max_iters = None  # 注释掉这行
    trainer = Trainer(config.trainer, model, train_dataset)
    
    # 保存训练损失
    losses = []
    
    # 定义训练循环
    device = trainer.device
    model.to(device)
    optimizer = model.configure_optimizers(config.trainer)
    model.train()
    
    print("开始训练...")
    
    max_epochs = 20
    max_iters = config.trainer.max_iters if config.trainer.max_iters is not None else 2000
    iter_num = 0
    
    for epoch in range(max_epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            # 将数据移到设备上
            x = x.to(device)
            y = y.to(device)
            
            # 前向传播
            logits, loss = model(x, y)
            
            # 反向传播和优化
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.grad_norm_clip)
            optimizer.step()
            
            # 记录损失
            losses.append(loss.item())
            
            if iter_num % 100 == 0:
                print(f"epoch {epoch}, iter {iter_num}: train loss {loss.item():.5f}")
            
            if iter_num % 500 == 0 or iter_num == max_iters - 1:
                model.eval()
                with torch.no_grad():
                    # 验证一些示例
                    examples = [
                        "你好。",
                        "我已经起来了。",
                        "谢谢你。"
                    ]
                    
                    print("\n--- 验证翻译效果 ---")
                    for ex in examples:
                        translation = translate(model, train_dataset, ex, device=device)
                        print(f"中文: {ex}")
                        print(f"翻译: {translation}\n")
                    
                    # 保存模型
                    print("保存模型")
                    ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                    torch.save(model.state_dict(), ckpt_path)
                
                model.train()
            
            iter_num += 1
            if iter_num >= max_iters:
                break
        
        if iter_num >= max_iters:
            break
    
    print("训练完成!")
    
    # 绘制损失曲线
    plot_loss_curve(losses, os.path.join(config.system.work_dir, "loss_curve.png"))
    
    # 训练完成后，测试一系列示例并创建表格
    print("\n--- 训练完成，创建结果表格 ---")
    
    # 测试样例 (至少10个)
    test_examples = [
        "你好。",
        "我已经起来了。",
        "我不干了。",
        "我知道。",
        "等一下！",
        "我试试。",
        "他跑了。",
        "开始！",
        "我赢了。",
        "跳进来。",
        "住手！",
        "我沒事。"
    ]
    
    # 获取原始的英文翻译（基本事实）
    ground_truth = []
    for example in test_examples:
        # 在训练集中查找匹配的翻译
        found = False
        for i in range(len(train_dataset.chinese_texts)):
            if train_dataset.chinese_texts[i] == example:
                ground_truth.append(train_dataset.english_texts[i])
                found = True
                break
        if not found:
            ground_truth.append("Unknown")  # 如果找不到对应的翻译
    
    # 模型翻译结果
    translations = []
    
    # 进行翻译并收集结果
    for ex in test_examples:
        translated = translate(model, train_dataset, ex, device=device)
        translations.append(translated)
        print(f"中文: {ex}")
        print(f"翻译: {translated}\n")
    
    # 创建结果表格
    table_df = create_result_table(
        test_examples, 
        ground_truth, 
        translations, 
        os.path.join(config.system.work_dir, "translation_results")
    )
    
    print(f"\n所有结果已保存到 {config.system.work_dir} 目录") 