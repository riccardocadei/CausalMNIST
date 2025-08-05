import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import recall_score, precision_score
from models import compute_effect
from itertools import compress
from torch.utils.tensorboard import SummaryWriter
import time
import pandas as pd

def training(model,
             dataset, 
             train_ratio=0.9,
             epochs=6,
             lr=0.001,
             batch_size=64,
             k_inv=0.1,
             method='ERM',
             verbose=True,
             log_dir='runs',
             eval=True,
             gpu=True,
             cfl=0):
    # TODO: update description
    '''
    Train the model on the CausalMNIST dataset.
    
    Args:
        finetuning: bool
        force_generation: bool
        subsampling: str
        normalize: bool

    '''
    target = 3 # W:0, U:1, T:2, Y:3
    best_val_bacc = 0
    best_model = model
    use_gpu = torch.cuda.is_available()
    if gpu:
        device = torch.device("cuda" if use_gpu else "cpu")
    else:
        device = torch.device("cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_gpu else {}
    model = model.to(device)
    model.device = device
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=lr)
    n_tr = int(train_ratio*len(dataset))
    A = time.time()
    train = dataset.data_label_tuples[:n_tr]
    train_loader = torch.utils.data.DataLoader(train, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        **kwargs)
    unsupervised_loader = torch.utils.data.DataLoader(dataset.data_label_tuples, 
                                        batch_size=int(batch_size/train_ratio), 
                                        shuffle=True, 
                                        **kwargs)
    B = time.time()
    if method in ["IRM", "vREx"]:
        train_envs = []
        for w in dataset.W.unique():
            for u in dataset.U.unique():
                for t in dataset.T.unique():
                    mask = (dataset.W == w) & (dataset.U == u) & (dataset.T == t)
                    if sum(mask) < batch_size:
                        continue
                    train_env = (dataset.X[mask], dataset.Y[mask])
                    train_envs.append(train_env)
        if verbose: print("Num Env.", len(train_envs))
    if method=="IRM":
        scale = torch.tensor(1.).to(model.device).requires_grad_()
    C = time.time()
    if verbose: print(f"Data loading: {B-A:.2f}s")
    if verbose: print(f"Data loading Multi-Env: {C-B:.2f}s")
    writer = SummaryWriter(log_dir=f"{log_dir}/{method if 'ERM' in method else f'{method}_{k_inv}'}")  # Specify log directory
    for epoch in range(epochs):
        D = time.time()
        model.train()
        for batch_idx, ((image, variables), (image_, variables_))  in enumerate(zip(train_loader, unsupervised_loader)):
            I = time.time()
            X, y, t = image.to(device).float(), variables[target].to(device).long(), variables[2].to(device).long()
            optimizer.zero_grad()
            loss = 0
            if method in ["ERM", "DERM"]:
                y_pred = model(X)
                if method=="ERM":
                    loss = torch.nn.CrossEntropyLoss()(y_pred, y)
                if method=="DERM":
                    loss = torch.nn.CrossEntropyLoss(reduction='none')(y_pred, y)
                    yvar = variables[4].to(device).float()
                    oprob = variables[5].to(device).float()
                    weight = yvar/oprob
                    loss = (weight*loss).sum()
            if cfl:
                # epsilon = 0.01
                # supervised
                # t_pred = model(X, target="t")            
                # t_pred_prob = torch.softmax(t_pred, dim=1)[:,1]
                # mask = (t_pred_prob > 1-epsilon) | (t_pred_prob < epsilon)
                # loss_cfl = torch.nn.CrossEntropyLoss()(t_pred[mask], t[mask])
                # loss_cfl = torch.nn.CrossEntropyLoss()(t_pred, t)
                # loss = loss + cfl * loss_cfl

                # unsupervised
                X_, t_ = image_.to(device).float(), variables_[2].to(device).long()
                t_pred_ = model(X_, target="t")
                # t_pred_prob_ = torch.softmax(t_pred_, dim=1)[:,1]
                # mask_ = (t_pred_prob_ > 1-epsilon) | (t_pred_prob_ < epsilon)
                # loss_cfl = torch.nn.CrossEntropyLoss()(t_pred_[mask_], t_[mask_])
                loss_cfl = torch.nn.CrossEntropyLoss()(t_pred_, t_)
                loss = loss + cfl * loss_cfl                    
            F = time.time()
            if method in ["IRM", "vREx"]:
                losses = []
                for train_env in train_envs:
                    idx = torch.randperm(len(train_env[0]))[:batch_size]
                    X, y = train_env[0][idx].to(device).float(), train_env[1][idx].to(device).long()
                    y_pred = model(X)
                    if method=="IRM":
                        y_pred = y_pred * scale
                    loss_e = torch.nn.CrossEntropyLoss()(y_pred, y)
                    if method=="IRM":
                        grad = torch.autograd.grad(loss_e, [scale], create_graph=True)[0]
                        inv_constr = torch.sum(grad ** 2)
                        loss_e = loss_e + inv_constr * k_inv
                    if cfl:
                        raise NotImplementedError("CFL not implemented for IRM and vREx")
                    losses.append(loss_e)
                loss = torch.sum(torch.stack(losses))
                if method=="vREx":
                    loss = loss + k_inv * torch.var(torch.stack(losses))
            G = time.time()
            loss.backward()
            optimizer.step()
            H = time.time()
            if batch_idx % 100 == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(X), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
        E = time.time()
        if verbose: print(f"Epoch {epoch} Train: {E-D:.2f}s")
        if verbose: print(f"Epoch {epoch} Train Multi-Env Inference: {G-F:.2f}s")
        if verbose: print(f"Epoch {epoch} Train Single Inference: {F-I:.2f}s")
        if verbose: print(f"Epoch {epoch} Train Multi-Env Backward: {H-G:.2f}s")
        if eval:
            model.eval()
            with torch.no_grad():
                #dataset.Y_hat = model(dataset.X.to(device)).max(axis=1)[1].cpu().numpy()
                dataset.Y_hat = model.cond_exp(dataset.X.to(device)).detach().cpu().numpy()
                # p0 = dataset.Y_hat[:n_tr][dataset.Y[:n_tr]==0].mean()
                # p1 = dataset.Y_hat[:n_tr][dataset.Y[:n_tr]==1].mean()
                # dataset.Y_hat = (dataset.Y_hat - p0) / (p1 - p0)
                tr_acc = accuracy_score(dataset.Y[:n_tr], dataset.Y_hat[:n_tr].round())
                tr_bal_acc = balanced_accuracy_score(dataset.Y[:n_tr], dataset.Y_hat[:n_tr].round())
                val_acc = accuracy_score(dataset.Y[n_tr:], dataset.Y_hat[n_tr:].round())
                val_bal_acc = balanced_accuracy_score(dataset.Y[n_tr:], dataset.Y_hat[n_tr:].round())
                tr_recall = recall_score(dataset.Y[:n_tr], dataset.Y_hat[:n_tr].round(), average='macro')
                tr_precision = precision_score(dataset.Y[:n_tr], dataset.Y_hat[:n_tr].round(), average='macro')
                if verbose: print(f'Train Recall: {tr_recall:.3f}, Train Precision: {tr_precision:.3f}')
                val_recall = recall_score(dataset.Y[n_tr:], dataset.Y_hat[n_tr:].round(), average='macro')
                val_precision = precision_score(dataset.Y[n_tr:], dataset.Y_hat[n_tr:].round(), average='macro')
                if verbose: print(f'Val Recall: {val_recall:.3f}, Val Precision: {val_precision:.3f}')

                if val_bal_acc > best_val_bacc:
                    best_val_bacc = val_bal_acc
                    best_model = model
                    best_model.best_epoch = epoch
                if verbose: print(f'Train Accuracy: {tr_acc:.3f}, Train Bal. Accuracy: {tr_bal_acc:.3f}')
                if verbose: print(f'Val Accuracy: {val_acc:.3f}, Val Bal. Accuracy: {val_bal_acc:.3f}')
                AD_ = compute_effect(dataset, method="AD", pred=True)
                AIPW_ = compute_effect(dataset, method="AIPW", pred=True, total=True)
                AD = compute_effect(dataset, method="AD", pred=False)
                AIPW = compute_effect(dataset, method="AIPW", pred=False, total=True)
                if verbose: print(f'AD (pred): {AD_:.3f}, AIPW (pred): {AIPW_:.3f}, AD: {AD:.3f}, AIPW: {AIPW:.3f}')
                writer.add_scalar('Train - Accuracy', tr_acc, epoch)
                writer.add_scalar('Train - Bal. Accuracy', tr_bal_acc, epoch)
                writer.add_scalar('Val - Accuracy', val_acc, epoch)
                writer.add_scalar('Val - Bal. Accuracy', val_bal_acc, epoch)
                writer.add_scalar('AD (pred)', AD_, epoch)
                writer.add_scalar('AIPW (pred)', AIPW_, epoch)
                writer.add_scalar('AD', AD, epoch)
                writer.add_scalar('AIPW', AIPW, epoch)
                

    writer.close()
    return best_model   



