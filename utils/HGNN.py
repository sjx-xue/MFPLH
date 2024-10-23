import numpy as np


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H ******SJX 超图的顶点-边矩阵 ********
    :param variable_weight: whether the weight of hyperedge is variable ******SJX 超边的权重是否可变 *********
    :return: G
    """
    # H = H.numpy()
    # print("H的形状：", H.shape)

    n_edge = H.shape[1]
    # print("n_edge的值：", n_edge)

    W = np.ones(n_edge)
    # print("W的形状：", W.shape)
    W = np.expand_dims(W, axis=1)
    # print("W的形状：", W.shape)

    DV = np.sum(H, axis=1)
    # print("DV的形状：", DV.shape)

    DE = np.sum(H, axis=0)
    DE = DE.astype(float)
    # print("DE的形状：", DE.shape)

    # 检查 DE 中是否存在零元素
    zero_mask = (DE == 0)
    DE[zero_mask] = 1e-10  # 将零元素替换为一个很小的值，避免除以零

    invDE = np.mat(np.diag(np.power(DE, -1)))
    # print("invDE的形状：", invDE.shape)

    DV = np.where(DV == 0, 0.000001, DV)
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    # print("DV2的形状：", DV2.shape)

    W = W.squeeze()
    W = np.mat(np.diag(W))
    # print("W的形状：", W.shape)

    H = np.mat(H)
    HT = H.T
    # print("HT的形状：", HT.shape)

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        # print("G的形状：", G.shape)
        return G


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H ******SJX 超图的顶点-边矩阵 *********
    :param variable_weight: whether the weight of hyperedge is variable ******SJX 是否超边的权重可变 ******
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


# **************************** SJX ******************************************
# 定义了一个名为 construct_H_with_KNN_from_distance 的函数，用于根据节点之间的距离矩阵构建超图的顶点-边矩阵
# 函数首先获取节点数量 n_obj，然后根据节点数量初始化超图的边数 n_edge
# 接着，创建一个全零矩阵 H，大小为 n_obj × n_edge，用于存储顶点-边矩阵
# 然后，对于每个节点，找到其与其他节点的距离，并找到其最近的 k_neig 个邻居节点。根据这些邻居节点的距离，计算顶点-边矩阵中对应的值
# 如果 is_probH 为 True，则根据距离计算概率值；否则，直接将值设置为 1.0
# 最后，返回构建好的顶点-边矩阵 H
# **************************** SJX ******************************************
def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix *****SJX 节点之间的距离矩阵 ********
    :param k_neig: K nearest neighbor *******SJX K最近邻的数量 *********
    :param is_probH: prob Vertex-Edge matrix or binary *******SJX 是否生成概率的顶点-边矩阵 *****
    :param m_prob: prob ********SJX 用于概率矩阵生成的参数 *********
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


# **************************** SJX ******************************************
# 定义了一个名为 construct_H_with_KNN 的函数，用于从原始节点特征矩阵构建多尺度超图的顶点-边矩阵（Vertex-Edge matrix）
# 函数首先确保输入的特征矩阵 X 是二维的，如果不是，则将其 reshape 成二维
# 然后，根据输入的邻居扩展数量 K_neigs，计算节点之间的欧式距离矩阵 dis_mat
# 接下来，根据每个邻居扩展数量，调用 construct_H_with_KNN_from_distance 函数构建相应的顶点-边矩阵 H_tmp
# 如果 split_diff_scale 为 False，则将所有不同尺度的超边组合成一个超图，使用 hyperedge_concat 函数进行拼接；
# 如果为 True，则将不同尺度的超边分别保存在列表 H 中
# 最后，返回构建好的超图顶点-边矩阵 H
# **************************** SJX ******************************************
def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number ****SJX 原始节点特征矩阵，大小为 N_object x feature_number ****
    :param K_neigs: the number of neighbor expansion ****SJX 邻居扩展的数量，可以是一个整数或者整数列表 ****
    :param split_diff_scale: whether split hyperedge group at different neighbor scale *****SJX 是否在不同的邻居尺度上拆分超边组 ****
    :param is_probH: prob Vertex-Edge matrix or binary ****SJX 是否生成概率的顶点-边矩阵 *****
    :param m_prob: prob *******SJX 用于概率矩阵生成的参数 ********
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        # X = X.reshape(-1, X.shape[-1])
        X = X.reshape(X.shape[0], -1)

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    # dis_mat = cosine_similarity(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H
