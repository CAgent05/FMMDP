import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import torch
import pandas as pd
import random

import torch
import gymnasium as gym
from stable_baselines3 import SAC, PPO, DQN
from gymnasium.wrappers import GrayscaleObservation

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def loadsparse(filename):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    a = sp.csr_matrix(a)
    return a


def loadsparse2(fname):
    df = pd.read_csv(fname, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    row = np.max(a[:, 0])
    column = np.max(a[:, 1])
    s = sp.csr_matrix((a[:, 2], (a[:, 0],a[:, 1])), shape=(row.astype('int64') + 1, column.astype('int64') + 1))
    return s


def loaddata(filename):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    return a


# def unison_shuffled_copies(a, b):
#     assert a.shape[0]==b.shape[0]
#     p=np.random.permutation(a.shape[0])
#     return a[p], b[p]

# def load_raw_ts(path, dataset, tensor_format=True):
#     path = path + dataset + "/"
#     # x_train = np.load(path + 'X_train.npy')
#     # y_train = np.load(path + 'y_train.npy')
#     # x_test = np.load(path + 'X_test.npy')
#     # y_test = np.load(path + 'y_test.npy')
#     x_train = torch.load(path + 'X_train.pt')
#     y_train = torch.load(path + 'y_train.pt')
#     x_test = torch.load(path + 'X_valid.pt')
#     y_test = torch.load(path + 'y_valid.pt')


#     x_train, y_train = unison_shuffled_copies(x_train, y_train)
#     # x_test, y_test = unison_shuffled_copies(x_test, y_test)

#     x_train[np.isnan(x_train)] = 0
#     # y_train[np.isnan(y_train)] = 0
#     x_test[np.isnan(x_test)] = 0
#     # y_test[np.isnan(y_test)] = 0

#     ts = np.concatenate((x_train, x_test), axis=0)
#     # ts = np.transpose(ts, axes=(0, 2, 1))
#     labels = np.concatenate((y_train, y_test), axis=0)
#     nclass = int(np.amax(labels)) + 1

#     # total data size: 934
#     train_size = y_train.shape[0]
#     # train_size = 10
#     total_size = labels.shape[0]
#     idx_train = range(train_size)
#     idx_val = range(train_size, total_size)
#     idx_test = range(train_size, total_size)

#     if tensor_format:
#         # features = torch.FloatTensor(np.array(features))
#         ts = torch.FloatTensor(np.array(ts))
#         labels = torch.LongTensor(labels)

#         idx_train = torch.LongTensor(idx_train)
#         idx_val = torch.LongTensor(idx_val)
#         idx_test = torch.LongTensor(idx_test)

#     return ts, labels, idx_train, idx_val, idx_test, nclass

def unison_shuffled_copies(a, b):
    """
    将 a、b 按相同随机顺序打乱。
    兼容 NumPy ndarray 与 torch.Tensor。
    """
    assert a.shape[0] == b.shape[0], "a 和 b 行数必须一致"
    device = a.device
    # 如果 b 不在同一设备，就先搬过去
    if b.device != device:
        b = b.to(device)

    if isinstance(a, torch.Tensor):
        # 如果 a 在 GPU，perm 也会自动放到同一设备
        perm = torch.randperm(a.shape[0], device=a.device)
        return a[perm], b[perm]
    else:  # NumPy
        perm = np.random.permutation(a.shape[0])
        return a[perm], b[perm]


def load_raw_ts(path, dataset, tensor_format=True):
    """
    加载时间序列数据并返回:
        ts, labels, idx_train, idx_val, idx_test, nclass
    - 若 tensor_format=True  → 全部返回 torch.Tensor
    - 若 tensor_format=False → 全部返回 np.ndarray，与旧代码接口保持一致
    """
    path = path + dataset + "/"

    # 读取 .pt 文件（Tensor），如果想兼容 .npy，可自行添加分支
    x_train = torch.load(path + 'X_train.pt').float()  # 确保是 float 类型
    y_train = torch.load(path + 'y_train.pt').float() 
    # x_train  = torch.load(path + 'X_valid.pt')
    # y_train  = torch.load(path + 'y_valid.pt')
    x_test  = torch.load(path + 'X_valid.pt').float() 
    y_test  = torch.load(path + 'y_valid.pt').float() 

    x_train = x_train.squeeze(1)
    x_test  = x_test.squeeze(1)

    print("x_train.device:", x_train.device)
    print("y_train.device:", y_train.device)
    print("x_test.device:", x_test.device)
    print("y_test.device:", y_test.device)

    print("x_train.shape:", x_train.shape)
    print("x_test.shape:", x_test.shape)

    device = x_train.device  # 获取数据所在设备
    if y_train.device != device:
        y_train = y_train.to(device)
    if y_test.device != device:
        y_test = y_test.to(device)

    # # 如果指定 device（如 'cuda:0'），统一搬到该设备
    # if device is not None:
    #     x_train = x_train.to(device)
    #     y_train = y_train.to(device)
    #     x_test  = x_test.to(device)
    #     y_test  = y_test.to(device)

    #  训练集随机打乱（保持数据-标签对齐）
    x_train, y_train = unison_shuffled_copies(x_train, y_train)

    #  NaN 置零
    x_train = x_train.detach()
    x_train = torch.nan_to_num(x_train, nan=0.0)
    
    x_test = x_test.detach()
    x_test = torch.nan_to_num(x_test, nan=0.0)

    #  合并训练 + 验证
    ts     = torch.cat((x_train, x_test), dim=0)
    labels = torch.cat((y_train, y_test), dim=0).long()

    #  类别数
    nclass = int(labels.max().item() + 1)

    #  构造索引
    train_size = y_train.shape[0]
    total_size = labels.shape[0]

    idx_train = torch.arange(0, train_size,   dtype=torch.long, device=labels.device)
    idx_val   = torch.arange(train_size, total_size, dtype=torch.long, device=labels.device)
    idx_test  = idx_val.clone()  # 这里 val、test 相同

    # #  若用户需要 NumPy 格式，全部搬回 CPU 后转 np.ndarray
    # if not tensor_format:
    #     ts        = ts.cpu().numpy()
    #     labels    = labels.cpu().numpy()
    #     idx_train = idx_train.cpu().numpy()
    #     idx_val   = idx_val.cpu().numpy()
    #     idx_test  = idx_test.cpu().numpy()

    return ts, labels, idx_train, idx_val, idx_test, nclass

def load_muse(data_path="./data/", dataset="ECG", sparse=False, tensor_format=True, shuffle=False):

    if sparse:
        path = data_path + "muse_sparse/" + dataset + "/"
    else:
        path = data_path + "muse/" + dataset + "/"
    file_header = dataset + "_"

    # load feature
    if sparse:
        train_features = loadsparse2(path + file_header + "train.csv")
        test_features = loadsparse2(path + file_header + "test.csv")

    else:
        train_features = loadsparse(path + file_header + "train.csv")
        test_features = loadsparse(path + file_header + "test.csv")


    # crop the features
    mf = np.min((test_features.shape[1], train_features.shape[1]))
    train_features = train_features[:, 0: mf]
    test_features = test_features[:, 0: mf]

    print("Train Set:", train_features.shape, ",", "Test Set:", test_features.shape)

    if shuffle:
        # shuttle train features
        non_test_size = train_features.shape[0]
        idx_non_test = random.sample(range(non_test_size), non_test_size)
        train_features = train_features[idx_non_test, ]

    features = sp.vstack([train_features, test_features])
    features = normalize(features)

    train_labels = loaddata(path + file_header + "train_label.csv")
    if shuffle:
        train_labels = train_labels[idx_non_test, ]  # shuffle labels

    test_labels = loaddata(path + file_header + "test_label.csv")
    labels = np.concatenate((train_labels, test_labels), axis=0)

    nclass = np.amax(labels) + 1

    non_test_size = train_labels.shape[0]
    # val_size = int(non_test_size * val_ratio)
    # train_size = non_test_size - val_size
    total_size = features.shape[0]
    idx_train = range(non_test_size)
    idx_val = range(non_test_size, total_size)
    idx_test = range(non_test_size, total_size)

    if tensor_format:
        features = torch.FloatTensor(np.array(features.toarray()))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return features, labels, idx_train, idx_val, idx_test, nclass


def normalize(mx):
    """Row-normalize sparse matrix"""
    # rowsum = np.array(mx.sum(1))
    # r_inv = np.power(rowsum, -1).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    # mx = r_mat_inv.dot(mx)
    row_sums = mx.sum(axis=1)
    mx = mx.astype('float32')
    row_sums_inverse = 1 / row_sums
    f = mx.multiply(row_sums_inverse)
    return sp.csr_matrix(f).astype('float32')


def convert2sparse(features):
    aaa = sp.coo_matrix(features)
    value = aaa.data
    column_index = aaa.col
    row_pointers = aaa.row
    a = np.array(column_index)
    b = np.array(row_pointers)
    a = np.reshape(a, (a.shape[0],1))
    b = np.reshape(b, (b.shape[0],1))
    s = np.concatenate((a, b), axis=1)
    t = torch.sparse.FloatTensor(torch.LongTensor(s.T), torch.FloatTensor(value))
    return t


def accuracy(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

    return accuracy_score

def random_hash(features,K):
    idx=np.array(range(features.shape[1]));
    np.random.shuffle(idx)
    feat=features[:,idx]
    for i in range(features.shape[0]):
        f=np.array(feat[0].toarray())
        f.reshape


    tmp=torch.FloatTensor(features[:,idx[0:K]].toarray())
    return tmp


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def output_conv_size(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2 * padding) / stride) + 1

    return output

def dump_embedding(proto_embed, sample_embed, labels, dump_file='./plot/embeddings.txt'):
    proto_embed = proto_embed.cpu().detach().numpy()
    sample_embed = sample_embed.cpu().detach().numpy()
    embed = np.concatenate((proto_embed, sample_embed), axis=0)

    nclass = proto_embed.shape[0]
    labels = np.concatenate((np.asarray([i for i in range(nclass)]),
                             labels.squeeze().cpu().detach().numpy()), axis=0)

    with open(dump_file, 'w') as f:
        for i in range(len(embed)):
            label = str(labels[i])
            line = label + "," + ",".join(["%.4f" % j for j in embed[i].tolist()])
            f.write(line + '\n')

def config_dataset(dataset):
    if dataset == "ArticularyWordRecognition":
        train_len = 275
        test_len = 300
        num_nodes = 9
        feature_dim = 144
        nclass = 25
    elif dataset == "AtrialFibrillation":
        train_len = 15
        test_len = 15
        num_nodes = 2
        feature_dim = 640
        nclass = 3
    elif dataset == "CharacterTrajectories":
        train_len = 1422
        test_len = 1436
        num_nodes = 3
        feature_dim = 182
        nclass = 20
    elif dataset == "FaceDetection":
        train_len = 5890
        test_len = 3524
        num_nodes = 144
        feature_dim = 62
        nclass = 2
    elif dataset == "FingerMovements":
        train_len = 316
        test_len = 100
        num_nodes = 28
        feature_dim = 50
        nclass = 2
    elif dataset == "HandMovementDirection":
        train_len = 160
        test_len = 74
        num_nodes = 10
        feature_dim = 400
        nclass = 4
    elif dataset == "Handwriting":
        train_len = 150
        test_len = 850
        num_nodes = 3
        feature_dim = 152
        nclass = 26
    elif dataset == "Heartbeat":
        train_len = 204
        test_len = 205
        num_nodes = 61
        feature_dim = 405
        nclass = 2
    elif dataset == "Libras":
        train_len = 180
        test_len = 180
        num_nodes = 2
        feature_dim = 45
        nclass = 15
    elif dataset == "LSST":
        train_len = 2459
        test_len = 2466
        num_nodes = 6
        feature_dim = 36
        nclass = 14
    elif dataset == "MotorImagery":
        train_len = 278
        test_len = 100
        num_nodes = 64
        feature_dim = 3000
        nclass = 2
    elif dataset == "NATOPS":
        train_len = 180
        test_len = 180
        num_nodes = 24
        feature_dim = 51
        nclass = 6
    elif dataset == "PEMS-SF":
        train_len = 267
        test_len = 173
        num_nodes = 963
        feature_dim = 144
        nclass = 7
    elif dataset == "PenDigits":
        train_len = 7494
        test_len = 3498
        num_nodes = 2
        feature_dim = 8
        nclass = 10
    elif dataset == "SelfRegulationSCP2":
        train_len = 200
        test_len = 180
        num_nodes = 7
        feature_dim = 1152
        nclass = 2
    elif dataset == "SpokenArabicDigits":
        train_len = 6599
        test_len = 2199
        num_nodes = 13
        feature_dim = 93
        nclass = 10
    elif dataset == "StandWalkJump":
        train_len = 12
        test_len = 15
        num_nodes = 4
        feature_dim = 2500
        nclass = 3
    elif dataset == "BipedalWalkerHCSA_20":
        train_len = 2000
        test_len = 1000
        num_nodes = 28
        feature_dim = 20
        nclass = 2
    elif dataset == "HopperSA_20":
        train_len = 2000
        test_len = 1000
        num_nodes = 14
        feature_dim = 20
        nclass = 2
    elif dataset == "Walker2dSA_20":
        train_len = 2000
        test_len = 1000
        num_nodes = 23
        feature_dim = 20
        nclass = 2
    elif dataset == "HumanoidSA_20":
        train_len = 2000
        test_len = 1000
        num_nodes = 62
        feature_dim = 20
        nclass = 2
    elif dataset == "InvertedDoublePendulumSA_20":
        train_len = 2000
        test_len = 1000
        num_nodes = 12
        feature_dim = 20
        nclass = 2
    elif dataset == "CarRacingSA_20":
        train_len = 2938
        test_len = 1260
        num_nodes = 35
        feature_dim = 20
        nclass = 2
    else:
        raise Exception("Only support these datasets...") 

    return train_len, test_len, num_nodes, feature_dim, nclass

def corr_matrix(train_len, test_len, num_nodes, use_cuda, dataset_path, dataset):
    A = np.ones((num_nodes, num_nodes), np.int8)
    A = A / np.sum(A, 0)
    A_new = np.zeros((train_len, num_nodes, num_nodes), dtype=np.float32)
    for i in range(train_len):
        A_new[i, :, :] = A
    train_A = torch.from_numpy(A_new)
    A_train_tensor = torch.load(dataset_path + dataset + '/X_train.pt')
    A_train_tensor = A_train_tensor.squeeze(1) 
    for i in range(train_len):
        # A = np.load(dataset_path+dataset+'/X_train.npy')[i]
        A = A_train_tensor[i].detach().cpu().numpy()
        d = {}
        for i in range(A.shape[0]):
            d[i] = A[i]

        df = pd.DataFrame(d)
        df_corr = df.corr()
        df_corr = df.corr().fillna(0.0)
        # train_A[i] = torch.from_numpy(df_corr.to_numpy() / np.sum(df_corr.to_numpy(), 0))
        train_A[i] = torch.from_numpy(df_corr.to_numpy() / (np.sum(df_corr.to_numpy(), 0) + 1e-8))  

        if use_cuda==1:
            train_A[i] = train_A[i].cuda()
    # breakpoint()  # 调试用
    A = np.ones((num_nodes, num_nodes), np.int8)
    A = A / np.sum(A, 0)
    A_new = np.zeros((test_len, num_nodes,num_nodes), dtype=np.float32)
    for i in range(test_len):
        A_new[i, :, :] = A
    test_A = torch.from_numpy(A_new)
    A_test_tensor = torch.load(dataset_path + dataset + '/X_valid.pt')
    A_test_tensor = A_test_tensor.squeeze(1)
    for i in range(test_len):
        # A = np.load(dataset_path+dataset+'/X_test.npy')[i]
        A = A_test_tensor[i].detach().cpu().numpy()
        d = {}
        for i in range(A.shape[0]):
            d[i] = A[i]

        df = pd.DataFrame(d)
        df_corr = df.corr()
        df_corr = df.corr().fillna(0.0)
        # test_A[i] = torch.from_numpy(df_corr.to_numpy() / np.sum(df_corr.to_numpy(), 0))
        test_A[i] = torch.from_numpy(df_corr.to_numpy() / (np.sum(df_corr.to_numpy(), 0) + 1e-8))
        if use_cuda==1:
            test_A[i] = test_A[i].cuda()
    # train_A[train_A>0.1]=1
    # test_A[test_A > 0.1] = 1
    # np.save("./train_A.npy",train_A)
    # np.save("./test_A.npy", test_A)
    A_dir = "./model/MTPool/" + dataset + "_A.pt"

    torch.save(train_A, A_dir)

    assert not torch.isnan(test_A).any(),  "test_A  里还有 NaN!"
    assert not torch.isnan(train_A).any(), "train_A 里还有 NaN!"
    

    return train_A, test_A

def prepare_agent(env_name, input_tag=False):
    if env_name == 'BipedalWalkerHC':
        env = gym.make('BipedalWalker-v3',
                       hardcore=True,
                       render_mode='rgb_array')
        model = SAC.load('./gymmodel/BipedalWalkerHC.zip')
        if input_tag == "SAR":
            num_nodes = 29
        elif input_tag == "SA":
            num_nodes = 28
        else:
            num_nodes = 24
        
        alg_tag = 'SAC'

    elif env_name == 'Walker2d':
        env = gym.make('Walker2d-v4')
        model = SAC.load('./gymmodel/Walker2d.zip')
        if input_tag == "SAR":
            num_nodes = 24
        elif input_tag == "SA":
            num_nodes = 23
        else:
            num_nodes = 17
        
        alg_tag = 'SAC'
        
    elif env_name == 'InvertedDoublePendulum':
        env = gym.make('InvertedDoublePendulum-v4')
        model = PPO.load('./gymmodel/InvertedDoublePendulum.zip')
        if input_tag == "SAR":
            num_nodes = 13
        elif input_tag == "SA":
            num_nodes = 12
        else:
            num_nodes = 11
        
        alg_tag = 'PPO'
    
    elif env_name == 'Hopper':
        env = gym.make('Hopper-v4')
        model = SAC.load('./gymmodel/Hopper.zip')
        if input_tag == "SAR":
            num_nodes = 15
        elif input_tag == "SA":
            num_nodes = 14
        else:
            num_nodes = 11
    
        alg_tag = 'SAC'
    
    elif env_name == 'Humanoid':
        env = gym.make('Humanoid-v4')
        model = SAC.load('./gymmodel/Humanoid.zip')
        if input_tag == "SAR":
            num_nodes = 63
        elif input_tag == "SA":
            num_nodes = 62
        else:
            num_nodes = 45
        
        alg_tag = 'SAC'    
    elif env_name == 'CarRacing':
        env = gym.make("CarRacing-v3",
                    render_mode="rgb_array",)
        env = GrayscaleObservation(env, keep_dim=True)
        
        model = PPO.load("./gymmodel/CarRacing.zip")

        if input_tag == "SAR":
            num_nodes = 36
        elif input_tag == "SA":
            num_nodes = 35
        else:
            num_nodes = 32
        
        alg_tag = 'PPO'
        
    return env, model, num_nodes, alg_tag
