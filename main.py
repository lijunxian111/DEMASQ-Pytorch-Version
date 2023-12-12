# -*- coding: utf-8 -*-
from train import train_toy_model, train_DEM_model
import argparse


parser = argparse.ArgumentParser(description='argparse learning')  # 创建解析器
parser.add_argument('--mod', type=str, default='DEMASQ', help='model for training')
parser.add_argument('--epochs', type=int, default=1000, help='epochs for training')  # 添加参数
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')
parser.add_argument('--input_dim', type=int, default=768, help='feature dim of text embeddings')
parser.add_argument('--save_path', type=str, default='save_models', help='the directory for saving models')

args = parser.parse_args()  # 解析参数
print(args)

if __name__ == "__main__":
    if args.mod == 'toy':
        train_toy_model()
    elif args.mod == 'DEMASQ':
        train_DEM_model(args.epochs, args.learning_rate, args.input_dim, args.save_path)