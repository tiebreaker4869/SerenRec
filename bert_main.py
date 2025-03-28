import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict
from tqdm import tqdm
import argparse
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score

# 设置随机种子确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 数据预处理函数 - 适用于已处理的数据
def preprocess_data(data_file, min_seq_len=2):
    """
    预处理数据集
    """
    # 读取数据
    df = pd.read_csv(data_file)
    print(f"数据集大小: {len(df)}")
    
    # 按用户ID和时间戳排序
    df = df.sort_values(by=['user_id', 'timestamp'])
    
    # 构建用户交互序列
    user_seq = defaultdict(list)
    for row in df.itertuples():
        user_seq[row.user_id].append(row.item_id)
    
    # 过滤掉交互少于min_seq_len的用户
    filtered_users = [user for user, items in user_seq.items() if len(items) >= min_seq_len]
    print(f"有效用户数量: {len(filtered_users)}")
    
    # 获取最大的用户ID和物品ID以确定范围
    max_user_id = max(filtered_users)
    all_items = set()
    for user in filtered_users:
        all_items.update(user_seq[user])
    max_item_id = max(all_items)
    
    # 构建序列数据
    seqs = {}
    for user in filtered_users:
        seqs[user] = user_seq[user]
    
    # 计算一些统计信息
    num_users = max_user_id + 1  # 用户ID范围是[0, max_user_id]
    num_items = max_item_id + 1  # 物品ID范围是[0, max_item_id]，0用于padding
    
    # 如果物品ID从1开始，则调整num_items
    if min(all_items) > 0:
        num_items += 1
    
    sequence_stats = [len(seq) for seq in seqs.values()]
    avg_seq_len = np.mean(sequence_stats)
    max_seq_len = np.max(sequence_stats)
    
    print(f"用户数量: {len(filtered_users)}, 物品数量: {len(all_items)}")
    print(f"最大用户ID: {max_user_id}, 最大物品ID: {max_item_id}")
    print(f"平均序列长度: {avg_seq_len:.2f}, 最大序列长度: {max_seq_len}")
    print(f"数据集密度: {len(df) / (len(filtered_users) * len(all_items)):.6f}")
    
    return seqs, num_users, num_items, filtered_users, all_items

# 按用户划分训练、验证和测试数据集
def split_users_train_val_test(seqs, user_threshold=0.8, max_len=50):
    """
    按用户ID划分训练集、验证集和测试集
    user_threshold: 用于训练的用户比例，其余用于测试
    """
    # 获取所有用户ID并随机打乱
    all_users = list(seqs.keys())
    np.random.shuffle(all_users)
    
    # 根据用户比例划分
    train_users_count = int(len(all_users) * user_threshold)
    train_users = all_users[:train_users_count]
    test_users = all_users[train_users_count:]
    
    # 根据用户ID划分数据
    train_seqs = {}
    val_seqs = {}  # 验证集从训练用户中取部分数据
    test_seqs = {}
    
    # 处理训练用户
    for user in train_users:
        seq = seqs[user]
        if len(seq) <= 1:  # 至少需要2个交互
            continue
        
        # 使用最后一个交互作为验证，其余用于训练
        val_item = seq[-1]
        train_seq = seq[:-1]
        
        # 如果序列过长，截断
        if len(train_seq) > max_len:
            train_seq = train_seq[-max_len:]
        
        train_seqs[user] = train_seq
        val_seqs[user] = (train_seq, val_item)
    
    # 处理测试用户
    for user in test_users:
        seq = seqs[user]
        if len(seq) <= 1:  # 至少需要2个交互
            continue
        
        # 使用最后一个交互作为测试目标，其余作为输入序列
        test_item = seq[-1]
        test_input = seq[:-1]
        
        # 如果序列过长，截断
        if len(test_input) > max_len:
            test_input = test_input[-max_len:]
        
        test_seqs[user] = (test_input, test_item)
    
    # 打印统计信息
    print(f"总用户数: {len(all_users)}")
    print(f"训练集用户数: {len(train_seqs)}")
    print(f"验证集用户数: {len(val_seqs)}")
    print(f"测试集用户数: {len(test_seqs)}")
    
    return train_seqs, val_seqs, test_seqs

# BERT4Rec数据集类定义
class BERT4RecDataset(Dataset):
    def __init__(self, user_seqs, num_items, max_len, mask_prob=0.2, is_training=True):
        self.user_seqs = user_seqs
        self.users = list(self.user_seqs.keys())
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.is_training = is_training
        self.mask_token = num_items  # 使用num_items作为[MASK]的token ID
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        user = self.users[index]
        
        # 处理训练数据（BERT4Rec使用随机掩码的方式进行训练）
        if self.is_training:
            seq = self.user_seqs[user].copy()
            
            # 序列长度小于2不能进行训练
            if len(seq) <= 1:
                return None
                
            # 如果序列过长则截断
            if len(seq) > self.max_len:
                seq = seq[-self.max_len:]
                
            # 创建序列副本用于保存原始物品ID（作为标签）
            tokens = seq.copy()
            labels = [-100] * len(tokens)  # 初始化标签，-100表示不计算损失的位置
            
            # 获取要被掩码的索引位置（随机选择部分位置进行掩码）
            mask_indices = []
            for i in range(len(tokens)):
                prob = random.random()
                if prob < self.mask_prob:
                    mask_indices.append(i)
                    
            # 对选中的位置应用掩码
            for i in mask_indices:
                labels[i] = tokens[i]  # 保存原始标签用于计算损失
                tokens[i] = self.mask_token  # 替换为[MASK]标记
            
            # 填充序列到固定长度
            pad_len = self.max_len - len(tokens)
            tokens = [0] * pad_len + tokens
            labels = [-100] * pad_len + labels
            
            # 创建注意力掩码（padding位置为0，其他位置为1）
            attention_mask = [0] * pad_len + [1] * len(seq)
            
            return {
                'input_ids': torch.LongTensor(tokens),
                'attention_mask': torch.LongTensor(attention_mask),
                'labels': torch.LongTensor(labels),
                'user': torch.LongTensor([user])
            }
            
        # 处理验证或测试数据
        else:
            # 获取输入序列和目标物品
            input_seq, target_item = self.user_seqs[user]
            
            # 如果输入序列过长则截断
            if len(input_seq) > self.max_len - 1:  # 预留一个位置给[MASK]
                input_seq = input_seq[-(self.max_len-1):]
            
            # 添加[MASK]标记在序列末尾，用于预测下一个物品
            tokens = list(input_seq) + [self.mask_token]
            
            # 填充序列到固定长度
            pad_len = self.max_len - len(tokens)
            input_ids = [0] * pad_len + tokens
            
            # 创建注意力掩码
            attention_mask = [0] * pad_len + [1] * len(tokens)
            
            return {
                'input_ids': torch.LongTensor(input_ids),
                'attention_mask': torch.LongTensor(attention_mask),
                'target_item': torch.LongTensor([target_item]),
                'user': torch.LongTensor([user])
            }
    
# DataLoader的collate_fn
def collate_fn(batch):
    # 过滤None值
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    
    result = {}
    for key in batch[0].keys():
        if batch[0][key] is not None:
            result[key] = torch.stack([b[key] for b in batch])
    
    return result

# BERT4Rec模型架构定义
class BERT4Rec(nn.Module):
    def __init__(self, num_items, hidden_size, num_layers, num_heads, dropout, max_len):
        super(BERT4Rec, self).__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.mask_token = num_items  # 使用num_items作为[MASK]的token ID
        
        # 物品嵌入层（额外添加一个嵌入用于[MASK]标记）
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        
        # 位置编码
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        
        # LayerNorm和Dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # BERT Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 预测层
        self.prediction_layer = nn.Linear(hidden_size, num_items)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 批大小和序列长度
        batch_size, seq_len = input_ids.size()
        
        # 获取物品嵌入
        item_embeddings = self.item_embeddings(input_ids)
        
        # 添加位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(positions)
        
        # 结合物品嵌入和位置编码
        embeddings = item_embeddings + position_embeddings
        embeddings = self.dropout(self.layer_norm(embeddings))
        
        # 转化为Transformer期望的注意力掩码格式
        if attention_mask is not None:
            # 这里attention_mask中的0表示padding位置，1表示有效位置
            # 需要将其转换为transformer要求的格式：True表示掩盖，False表示保留
            extended_attention_mask = (attention_mask == 0)
        else:
            extended_attention_mask = None
        
        # 通过Transformer Encoder层
        sequence_output = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=extended_attention_mask
        )
        
        # 预测物品
        prediction_scores = self.prediction_layer(sequence_output)
        
        # 如果提供了标签，计算损失
        loss = None
        if labels is not None:
            # 将预测分数整形为[batch_size * seq_len, num_items]
            flat_prediction_scores = prediction_scores.view(-1, self.num_items)
            # 将标签整形为[batch_size * seq_len]
            flat_labels = labels.view(-1)
            # 使用交叉熵损失，忽略标签为-100的位置
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(flat_prediction_scores, flat_labels)
            
        return {
            'loss': loss,
            'logits': prediction_scores,
            'hidden_states': sequence_output
        }
    
    def predict(self, input_ids, attention_mask=None, target_item=None):
        """
        计算序列对目标物品的预测分数
        """
        # 获取模型输出
        outputs = self.forward(input_ids, attention_mask)
        sequence_output = outputs['logits']
        
        # 获取[MASK]位置的输出
        # 首先找到每个序列中[MASK]的位置
        mask_positions = (input_ids == self.mask_token).nonzero(as_tuple=True)
        batch_indices = mask_positions[0]
        seq_indices = mask_positions[1]
        
        # 获取每个序列中[MASK]位置的预测分数
        mask_output = sequence_output[batch_indices, seq_indices, :]
        
        if target_item is not None:
            # 计算目标物品的预测分数
            target_scores = torch.gather(mask_output, 1, target_item)
            return target_scores
        else:
            # 返回所有物品的预测分数
            return mask_output
        
# 训练函数
def train(model, train_loader, optimizer, device, args):
    model.train()
    total_loss = 0
    train_steps = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        if batch is None:
            continue
            
        # 将数据移至设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        train_steps += 1
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / train_steps

# 评估函数
def evaluate(model, data_loader, device, k_list=[1, 5, 10], name=""):
    model.eval()
    
    # 初始化性能指标
    metrics = {
        'HR': {k: 0 for k in k_list},
        'NDCG': {k: 0 for k in k_list},
        'MRR': 0
    }
    
    valid_user_count = 0
    
    # 数据预处理
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {name}"):
            if batch is None:
                continue
                
            # 将数据移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_item = batch['target_item'].to(device)
            
            # 获取预测分数
            mask_output = model.predict(input_ids, attention_mask)
            
            # 对已有的序列中的物品进行掩码（不推荐已经看过的）
            # 获取除了padding和mask以外的物品ID
            for i, seq in enumerate(input_ids):
                for item in seq:
                    if item != 0 and item != model.mask_token:  # 排除padding和mask
                        mask_output[i, item] = -np.inf
            
            # 获取每个用户的top-k推荐
            max_k = max(k_list)
            _, topk_indices = torch.topk(mask_output, k=max_k)
            topk_indices = topk_indices.cpu().numpy()
            
            # 更新指标
            valid_user_count += input_ids.size(0)
            
            for i, target in enumerate(target_item):
                target_item_i = target.item()
                rank = 0
                
                # 查找目标物品在排名中的位置
                if target_item_i in topk_indices[i]:
                    rank = np.where(topk_indices[i] == target_item_i)[0][0] + 1
                    
                    # 更新Hit Ratio
                    for k in k_list:
                        if rank <= k:
                            metrics['HR'][k] += 1
                    
                    # 更新NDCG
                    for k in k_list:
                        if rank <= k:
                            metrics['NDCG'][k] += 1 / np.log2(rank + 1)
                    
                    # 更新MRR
                    metrics['MRR'] += 1 / rank
    
    # 计算平均指标
    for k in k_list:
        metrics['HR'][k] /= valid_user_count
        metrics['NDCG'][k] /= valid_user_count
    metrics['MRR'] /= valid_user_count
    
    return metrics

# 主函数
def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据预处理
    seqs, num_users, num_items, filtered_users, all_items = preprocess_data(
        args.data_file, 
        args.min_seq_len
    )
    
    # 按用户划分训练集、验证集和测试集
    print("按用户ID划分训练集、验证集和测试集...")
    train_seqs, val_seqs, test_seqs = split_users_train_val_test(
        seqs, 
        user_threshold=args.user_threshold, 
        max_len=args.max_len
    )
    
    # 创建输出目录
    if args.save_model and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # 保存划分方式和数据统计信息
    if args.save_model:
        with open(os.path.join(args.output_dir, 'data_stats.txt'), 'w') as f:
            f.write(f"数据集: {args.data_file}\n")
            f.write(f"用户数量: {len(filtered_users)}, 物品数量: {len(all_items)}\n")
            f.write(f"训练集用户数: {len(train_seqs)}\n")
            f.write(f"验证集用户数: {len(val_seqs)}\n")
            f.write(f"测试集用户数: {len(test_seqs)}\n")
            f.write(f"训练集用户比例: {args.user_threshold}\n")
            f.write(f"划分方式: 按用户划分 (随机选择{args.user_threshold*100}%的用户用于训练)\n")
    
    # 创建数据集
    train_dataset = BERT4RecDataset(
        user_seqs=train_seqs,
        num_items=num_items,
        max_len=args.max_len,
        mask_prob=args.mask_prob,
        is_training=True
    )
    
    val_dataset = BERT4RecDataset(
        user_seqs=val_seqs,
        num_items=num_items,
        max_len=args.max_len,
        is_training=False
    )
    
    test_dataset = BERT4RecDataset(
        user_seqs=test_seqs,
        num_items=num_items,
        max_len=args.max_len,
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # 初始化模型
    model = BERT4Rec(
        num_items=num_items,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_len=args.max_len
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 训练循环
    best_ndcg10 = 0
    best_epoch = 0
    k_list = [1, 5, 10]  # 评估的k值列表
    
    # 记录训练过程中的指标
    train_losses = []
    val_metrics_history = []
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # 训练
        train_loss = train(model, train_loader, optimizer, device, args)
        train_losses.append(train_loss)
        
        # 在验证集上评估
        val_metrics = evaluate(model, val_loader, device, k_list, "validation")
        val_metrics_history.append(val_metrics)
        
        # 打印验证集指标
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}")
        print(f"验证集 - HR@1: {val_metrics['HR'][1]:.4f}, HR@5: {val_metrics['HR'][5]:.4f}, HR@10: {val_metrics['HR'][10]:.4f}")
        print(f"验证集 - NDCG@5: {val_metrics['NDCG'][5]:.4f}, NDCG@10: {val_metrics['NDCG'][10]:.4f}, MRR: {val_metrics['MRR']:.4f}")
        
        # 保存最佳模型（使用NDCG@10作为主要指标）
        if val_metrics['NDCG'][10] > best_ndcg10:
            best_ndcg10 = val_metrics['NDCG'][10]
            best_epoch = epoch
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
    
    # 训练时间统计
    total_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {total_time/60:.2f} 分钟")
    print(f"最佳模型（Epoch {best_epoch+1}）的验证集NDCG@10: {best_ndcg10:.4f}")
    
    # 加载最佳模型并在测试集上评估
    if args.save_model:
        print("\n加载最佳模型并在测试集上评估...")
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    
    test_metrics = evaluate(model, test_loader, device, k_list, "test")
    print("\n测试集评估结果:")
    print(f"HR@1: {test_metrics['HR'][1]:.4f}, HR@5: {test_metrics['HR'][5]:.4f}, HR@10: {test_metrics['HR'][10]:.4f}")
    print(f"NDCG@5: {test_metrics['NDCG'][5]:.4f}, NDCG@10: {test_metrics['NDCG'][10]:.4f}, MRR: {test_metrics['MRR']:.4f}")
    
    # 保存测试结果
    if args.save_model:
        results = {
            'best_epoch': best_epoch + 1,
            'train_time': total_time,
            'validation': {
                'HR': val_metrics_history[best_epoch]['HR'],
                'NDCG': val_metrics_history[best_epoch]['NDCG'],
                'MRR': val_metrics_history[best_epoch]['MRR']
            },
            'test': {
                'HR': test_metrics['HR'],
                'NDCG': test_metrics['NDCG'],
                'MRR': test_metrics['MRR']
            }
        }
        
        # 保存结果为文本文件
        with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
            f.write(f"最佳模型（Epoch {best_epoch+1}）\n")
            f.write(f"训练耗时: {total_time/60:.2f} 分钟\n\n")
            
            f.write("验证集结果:\n")
            for k in k_list:
                f.write(f"HR@{k}: {val_metrics_history[best_epoch]['HR'][k]:.4f}\n")
            for k in k_list[1:]:  # 跳过k=1的NDCG
                f.write(f"NDCG@{k}: {val_metrics_history[best_epoch]['NDCG'][k]:.4f}\n")
            f.write(f"MRR: {val_metrics_history[best_epoch]['MRR']:.4f}\n\n")
            
            f.write("测试集结果:\n")
            for k in k_list:
                f.write(f"HR@{k}: {test_metrics['HR'][k]:.4f}\n")
            for k in k_list[1:]:  # 跳过k=1的NDCG
                f.write(f"NDCG@{k}: {test_metrics['NDCG'][k]:.4f}\n")
            f.write(f"MRR: {test_metrics['MRR']:.4f}\n")
        
        # 保存训练历史为pickle文件以便后续分析
        history = {
            'train_losses': train_losses,
            'val_metrics': val_metrics_history
        }
        with open(os.path.join(args.output_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        # 绘制训练损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'train_loss.png'))
        
        # 绘制验证指标曲线
        plt.figure(figsize=(12, 8))
        
        # HR曲线
        plt.subplot(2, 2, 1)
        for k in k_list:
            plt.plot([metrics['HR'][k] for metrics in val_metrics_history], label=f'HR@{k}')
        plt.xlabel('Epoch')
        plt.ylabel('Hit Ratio')
        plt.title('HR Over Epochs')
        plt.legend()
        
        # NDCG曲线
        plt.subplot(2, 2, 2)
        for k in k_list[1:]:  # 跳过k=1的NDCG
            plt.plot([metrics['NDCG'][k] for metrics in val_metrics_history], label=f'NDCG@{k}')
        plt.xlabel('Epoch')
        plt.ylabel('NDCG')
        plt.title('NDCG Over Epochs')
        plt.legend()
        
        # MRR曲线
        plt.subplot(2, 2, 3)
        plt.plot([metrics['MRR'] for metrics in val_metrics_history], label='MRR')
        plt.xlabel('Epoch')
        plt.ylabel('MRR')
        plt.title('MRR Over Epochs')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'validation_metrics.png'))
    
    return model, test_metrics

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="BERT4Rec model")
    
    # 数据参数
    parser.add_argument('--data_file', type=str, default='data.csv', help='数据文件路径')
    parser.add_argument('--min_seq_len', type=int, default=3, help='最小序列长度')
    parser.add_argument('--user_threshold', type=float, default=0.8, help='训练集用户比例')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=128, help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=2, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--max_len', type=int, default=50, help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--mask_prob', type=float, default=0.2, help='掩码概率')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作线程数')
    parser.add_argument('--save_model', action='store_true', help='是否保存模型')
    parser.add_argument('--output_dir', type=str, default='output_bert4rec', help='输出目录')
    
    return parser.parse_args()

# 执行主函数
if __name__ == "__main__":
    args = parse_args()
    
    # 创建输出目录
    if args.save_model and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args)