import itertools
import os
import os.path as osp
import pickle
import urllib
from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
%matplotlib inline
import networkx as nx

Data = namedtuple('Data', ['x', 'y', 'adjacency',
                           'train_mask', 'val_mask', 'test_mask'])


def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


class CoraData(object):
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="E:/Task2-3/data/cora", rebuild=False):
        """Cora数据，包括数据下载，处理，加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将下载、进行处理，并缓存到磁盘

        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
            * y: 节点的标签，总共包括7个类别，类型为 np.ndarray
            * adjacency: 邻接矩阵，维度为 2708 * 2708，类型为 scipy.sparse.coo.coo_matrix
            * train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False

        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: ../data/cora
                缓存数据路径: {data_root}/ch5_cached.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据

        """
        self.data_root = data_root
        save_file = osp.join(self.data_root, "ch5_cached.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)#将self.data保存到中#self.data也就是self._data
            print("Cached file: {}".format(save_file))
    
    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        引用自：https://github.com/rusty1s/pytorch_geometric
        """
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            osp.join(self.data_root, name)) for name in self.filenames]#x没有读
        train_index = np.arange(y.shape[0])#y.shape[0]是y的行数
        val_index = np.arange(y.shape[0], y.shape[0] + 500)#validation set 500个样本
        sorted_test_index = sorted(test_index) 

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)#
        val_mask = np.zeros(num_nodes, dtype=np.bool)#
        test_mask = np.zeros(num_nodes, dtype=np.bool)#数组中全为False
        train_mask[train_index] = True#
        val_mask[val_index] = True#
        test_mask[test_index] = True#index对应的位置标记为True
        adjacency1 = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())#
        print("Number of validation nodes: ", val_mask.sum())#
        print("Number of test nodes: ", test_mask.sum())#统计三种nodes的数量，列表中为true的值才算进sum里 
        
        return Data(x=x, y=y, adjacency=adjacency,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表创建邻接矩阵"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # 去除重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)), 
                                   (edge_index[:, 0], edge_index[:, 1])),
                    shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency
    

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    @staticmethod
    def normalization(adjacency):
        """计算 L=D^-0.5 * (A+I) * D^-0.5"""
        adjacency += sp.eye(adjacency.shape[0])    # 增加自连接
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
    
        Args: 
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'

class GcnNet(nn.Module):
    """
    NUM_HID表示隐藏层的层数
    """
    def __init__(self, input_dim=1433):
        super(GcnNet, self).__init__()
        self.midlayer = nn.ModuleList()

        if NUM_HID==1:
            self.gcn = GraphConvolution(input_dim,7)
            self.midlayer.append(self.gcn)
        else:
            self.gcn = GraphConvolution(input_dim,128)
            self.midlayer.append(self.gcn)
            for i in range(NUM_HID - 2):
                self.gcn = GraphConvolution(128,128)
                self.midlayer.append(self.gcn)
            self.gcn = GraphConvolution(128,7)
            self.midlayer.append(self.gcn)
            
    def forward(self, adjacency, feature):
        
        if NUM_HID==1:
            h = F.relu(self.midlayer[0](adjacency,feature))
            logits = h
            return logits
        else:
            h = F.relu(self.midlayer[0](adjacency,feature))
            for i in range(NUM_HID-1):
                h = self.midlayer[i+1](adjacency,h)
            logits = h
        return logits
                
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(DEVICE)

def randomedge_sampler(percent,ori_adj):
        """
        Randomly drop edge and preserve percent% edges.
        """
        "Opt here"

        nnz = ori_adj.nnz#ori_adj类型sp.coo_matrix
        perm = np.random.permutation(nnz)#在nnz基础上生成一个随机序列=‘’
        preserve_nnz = int(nnz*percent)
        perm = perm[:preserve_nnz]
        r_adj = sp.coo_matrix((ori_adj.data[perm],
                                (ori_adj.row[perm],
                                ori_adj.col[perm])),
                                shape=ori_adj.shape)
        r_adj = _preprocess_adj(r_adj)
        return r_adj
    
def _preprocess_adj(adj):
        adj = adj + sp.eye(adj.shape[0])#加单位矩阵
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        r_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()#计算(D+I)^-1/2*(A+I)*(D+I)^-1/2
        r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()#转为sparse tensor
        r_adj = r_adj.cuda()
        return r_adj


def ori_adjacency(graph):
        G = nx.from_dict_of_lists(graph)#Returns a graph from a dictionary of lists.
        adj = nx.adjacency_matrix(G)#Returns adjacency matrix of G
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)#get a symmetric matrix
        adj = sp.coo_matrix(adj)
        return adj


# 超参数定义
LEARNING_RATE = 0.1
WEIGHT_DACAY = 5e-4
EPOCHS = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLER_PERCENT = 0.7
#NUM_HID为隐藏层层数，根据需要修改
NUM_HID = 1
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)     # 为CPU设置随机种子
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 加载数据，并转换为torch.Tensor
#dataset = CoraData().data
dataset = CoraData()
with open(os.path.join(dataset.data_root, "{}".format(dataset.filenames[6])), 'rb') as f:
    graph = pickle.load(f,encoding='latin1')
dataset = CoraData().data
node_feature = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
tensor_x = tensor_from_numpy(node_feature, DEVICE)
tensor_y = tensor_from_numpy(dataset.y, DEVICE)
tensor_train_mask = tensor_from_numpy(dataset.train_mask, DEVICE)
tensor_val_mask = tensor_from_numpy(dataset.val_mask, DEVICE)
tensor_test_mask = tensor_from_numpy(dataset.test_mask, DEVICE)
num_nodes, input_dim = node_feature.shape

# 模型定义：Model, Loss, Optimizer
model = GcnNet(input_dim).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), 
                       lr=LEARNING_RATE, 
                       weight_decay=WEIGHT_DACAY)
ori_adj = ori_adjacency(graph)

# 训练主体函数
def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    #ori_adj = ori_adjacency(graph)
    #print(type(ori_adj))
    global dropedge
    for epoch in range(EPOCHS):
        #DropEdge
        dropedge_adj = randomedge_sampler(SAMPLER_PERCENT,ori_adj)
        
        logits = model(dropedge_adj, tensor_x)
        train_mask_logits = logits[tensor_train_mask]  # 只选择训练节点进行监督
        loss = criterion(train_mask_logits, train_y)    # 计算损失值
        optimizer.zero_grad()
        loss.backward()     # 反向传播计算参数的梯度
        optimizer.step()    # 使用优化方法进行梯度更新
    
        train_acc, _, _ = test(tensor_train_mask)     # 计算当前模型训练集上的准确率
        val_acc, _, _ = test(tensor_val_mask)     # 计算当前模型在验证集上的准确率
        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))
    print(f"current hidden layer: {NUM_HID}")
    print(f"validation accuracy: {val_acc_history[-1]}")
    
    return loss_history, val_acc_history

# 测试函数
def test(mask):
    model.eval()
    with torch.no_grad():
        #dropedge_adj = randomedge_sampler(SAMPLER_PERCENT,ori_adj)
        ori_adjacency = sparse_mx_to_torch_sparse_tensor(ori_adj)
        logits = model(ori_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()
 
 def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')
    
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')
    
    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()

loss, val_acc = train()
test_acc, test_logits, test_label = test(tensor_test_mask)
#print("Test accuarcy: ", test_acc.item())？
plot_loss_with_acc(loss, val_acc)

# 绘制测试数据的TSNE降维图
from sklearn.manifold import TSNE
tsne = TSNE()
out = tsne.fit_transform(test_logits)
fig = plt.figure()
for i in range(7):
    indices = test_label == i
    x, y = out[indices].T
    plt.scatter(x, y, label=str(i))
plt.legend()