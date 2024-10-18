import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from scipy.special import comb

def generate_compl_labels(labels):
    # args, labels: ordinary labels
    K = torch.max(labels)+1
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, K), len(labels), 0)
    mask = np.ones((len(labels), K), dtype=bool)
    mask[range(len(labels)), labels.numpy()] = False
    candidates_ = candidates[mask].reshape(len(labels), K-1)  # this is the candidates without true class
    idx = np.random.randint(0, K-1, len(labels))
    complementary_labels = candidates_[np.arange(len(labels)), np.array(idx)]
    return complementary_labels

def class_prior(complementary_labels):
    return np.bincount(complementary_labels) / len(complementary_labels)

def prepare_train_loaders(full_train_loader, batch_size, ordinary_train_dataset):
    for i, (data, labels) in enumerate(full_train_loader):
            K = torch.max(labels)+1 # K is number of classes, full_train_loader is full batch
    complementary_labels = generate_compl_labels(labels)
    ccp = class_prior(complementary_labels)
    complementary_dataset = torch.utils.data.TensorDataset(data, torch.from_numpy(complementary_labels).float())
    ordinary_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True)
    complementary_train_loader = torch.utils.data.DataLoader(dataset=complementary_dataset, batch_size=batch_size, shuffle=True)
    return ordinary_train_loader, complementary_train_loader, ccp


def assump_free_loss(f, K, labels, ccp):
    """Assumption free loss (based on Thm 1) is equivalent to non_negative_loss if the max operator's threshold is negative inf."""
    return non_negative_loss(f=f, K=K, labels=labels, ccp=ccp, beta=np.inf)

def non_negative_loss(f, K, labels, ccp, beta):
    ccp = torch.from_numpy(ccp).float().to(device)
    neglog = -F.log_softmax(f, dim=1)
    loss_vector = torch.zeros(K, requires_grad=True).to(device)
    temp_loss_vector = torch.zeros(K).to(device)
    for k in range(K):
        idx = labels == k
        if torch.sum(idx).item() > 0:
            idxs = idx.bool().view(-1, 1).repeat(1, K)
            #idxs = idx.byte().view(-1,1).repeat(1,K)
            neglog_k = torch.masked_select(neglog, idxs).view(-1,K)
            temp_loss_vector[k] = -(K-1) * ccp[k] * torch.mean(neglog_k, dim=0)[k]  # average of k-th class loss for k-th comp class samples
            loss_vector = loss_vector + torch.mul(ccp[k], torch.mean(neglog_k, dim=0))  # only k-th in the summation of the second term inside max
    loss_vector = loss_vector + temp_loss_vector
    count = np.bincount(labels.data.cpu()).astype('float')
    while len(count) < K:
        count = np.append(count, 0) # when largest label is below K, bincount will not take care of them
    loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(K, requires_grad=True).view(-1,1).to(device)-beta), 1)
    max_loss_vector, _ = torch.max(loss_vector_with_zeros, dim=1)
    final_loss = torch.sum(max_loss_vector)
    return final_loss, torch.mul(torch.from_numpy(count).float().to(device), loss_vector)

def forward_loss(f, K, labels):
    Q = torch.ones(K,K) * 1/(K-1)
    Q = Q.to(device)
    for k in range(K):
        Q[k,k] = 0
    q = torch.mm(F.softmax(f, 1), Q)
    return F.nll_loss(q.log(), labels.long())

def pc_loss(f, K, labels):
    sigmoid = nn.Sigmoid()
    fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
    loss_matrix = sigmoid( -1. * (f - fbar)) # multiply -1 for "complementary"
    M1, M2 = K*(K-1)/2, K-1
    pc_loss = torch.sum(loss_matrix)*(K-1)/len(labels) - M1 + M2
    return pc_loss

def accuracy_check(loader, model):
    sm = F.softmax
    total, num_samples = 0, 0
    for images, labels in loader:
        labels, images = labels.to(device), images.to(device)
        outputs = model(images)
        sm_outputs = sm(outputs, dim=1)
        _, predicted = torch.max(sm_outputs.data, 1)
        total += (predicted == labels).sum().item()
        num_samples += labels.size(0)
    return 100 * total / num_samples

def chosen_loss_c(f, K, labels, ccp, meta_method):
    class_loss_torch = None
    if meta_method=='free' or meta_method=='ga':
        final_loss, class_loss_torch = assump_free_loss(f=f, K=K, labels=labels, ccp=ccp)
    elif meta_method=='nn':
        final_loss, class_loss_torch = non_negative_loss(f=f, K=K, labels=labels, beta=0, ccp=ccp)
    elif meta_method=='forward':
        final_loss = forward_loss(f=f, K=K, labels=labels)
    elif meta_method=='pc':
        final_loss = pc_loss(f=f, K=K, labels=labels)
    return final_loss, class_loss_torch


# multi_complementary label
def generate_multi_comp_labels(labels, s, num_classes):
    k = torch.tensor(num_classes)
    n = labels.shape[0]
    index_ins = torch.arange(n)  # torch type
    realY = torch.zeros(n, k)
    realY[index_ins, labels] = 1
    partialY = torch.ones(n, k)

    labels_hat = labels.clone().numpy()
    candidates = np.repeat(np.arange(k).reshape(1, k), len(labels_hat), 0)  # candidate labels without true class
    mask = np.ones((len(labels_hat), k), dtype=bool)
    for i in range(s):
        mask[np.arange(n), labels_hat] = False
        candidates_ = candidates[mask].reshape(n, k - 1 - i)
        idx = np.random.randint(0, k - 1 - i, n)
        comp_labels = candidates_[np.arange(n), np.array(idx)]
        partialY[index_ins, torch.from_numpy(comp_labels)] = 0
        if i == 0:
            complementary_labels = torch.from_numpy(comp_labels)
        else:
            complementary_labels = torch.cat((complementary_labels, torch.from_numpy(comp_labels)), dim=0)
        labels_hat = comp_labels
    return partialY

def generate_uniform_comp_labels(labels, num_classes):
    K = torch.tensor(num_classes)
    n = labels.shape[0]
    number = torch.tensor([comb(K, i + 1) for i in range(K - 1)])  # 0 to K-2, convert list to tensor
    frequency_dis = number / number.sum()
    prob_dis = torch.zeros(K - 1)  # tensor of K-1
    for i in range(K - 1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i] + prob_dis[i - 1]

    random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float()  # tensor: n
    mask_n = torch.ones(n)  # n is the number of train_data
    partialY = torch.ones(n, K)
    temp_num_comp_train_labels = 0  # save temp number of comp train_labels

    for j in range(n):  # for each instance
        if j % 1000 == 0:
            print("current index:", j)
        for jj in range(K - 1):  # 0 to K-2
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_num_comp_train_labels = jj + 1  # decide the number of complementary train_labels
                mask_n[j] = 0

        candidates = torch.from_numpy(np.random.permutation(K.item()))  # because K is tensor type
        candidates = candidates[candidates != labels[j]]
        temp_comp_train_labels = candidates[:temp_num_comp_train_labels]

        for kk in range(len(temp_comp_train_labels)):
            partialY[j, temp_comp_train_labels[kk]] = 0  # fulfill the partial label matrix
    return partialY

def ce_loss(outputs, Y):
    logsm = nn.LogSoftmax(dim=1)
    logsm_outputs = logsm(outputs)
    final_outputs = logsm_outputs * Y
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss

def unbiased_estimator(loss_fn, outputs, partialY, device):
    n, k = partialY.shape[0], partialY.shape[1]
    comp_num = k - partialY.sum(dim=1)
    temp_loss = torch.zeros(n, k).to(device)
    for i in range(k):
        tempY = torch.zeros(n, k).to(device)
        tempY[:, i] = 1.0
        temp_loss[:, i] = loss_fn(outputs, tempY)

    candidate_loss = (temp_loss * partialY).sum(dim=1)
    noncandidate_loss = (temp_loss * (1 - partialY)).sum(dim=1)
    total_loss = candidate_loss - (k - comp_num - 1.0) / comp_num * noncandidate_loss
    average_loss = total_loss.mean()
    return average_loss


def log_loss(outputs, partialY):
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float()  # n

    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY

    average_loss = - ((k - 1) / (k - can_num) * torch.log(final_outputs.sum(dim=1)+1e-8)).mean()
    return average_loss


def exp_loss(outputs, partialY):
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float()  # n

    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY

    average_loss = ((k - 1) / (k - can_num) * torch.exp(-final_outputs.sum(dim=1))).mean()
    return average_loss