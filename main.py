# Importing the libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import AMG
from config import get_args
from data_loader import get_loader
args = get_args()

if 0:#torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(args.seed)
else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

# Getting the number of users and movies
num_users, num_items, num_side_features, num_features,\
u_features, v_features, u_features_side, v_features_side, class_values = get_loader(args.data_type)

class_values = torch.from_numpy(class_values).to(device).float()
u_features = torch.from_numpy(u_features).to(device).float()
v_features = torch.from_numpy(v_features).to(device).float()
u_features_side = torch.from_numpy(u_features_side).to(device)
v_features_side = torch.from_numpy(v_features_side).to(device)
rating_train = torch.load(args.train_path).to(device)
rating_val = torch.load(args.val_path).to(device)
rating_test = torch.load(args.test_path).to(device)

# Creating the architecture of the Neural Network
model = AMG(num_users, num_items, num_side_features, args.nb,
            u_features, v_features, u_features_side, v_features_side,
            num_users+num_items, class_values, args.emb_dim, args.hidden, args.dropout).to(device)

"""Print out the network information."""
num_params = 0
for p in list(model.parameters()):
    num_params += p.numel()
print(args)
print(model)
print("The number of parameters: {}".format(num_params))

optimizer = optim.Adam(model.parameters(), lr = args.lr, betas=[args.beta1, args.beta2])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

best_epoch = 0
best_loss  = 9999.

def reset_grad():
    """Reset the gradient buffers."""
    optimizer.zero_grad()

def train():
    global best_loss, best_epoch
    if args.start_epoch:
        model.load_state_dict(torch.load(os.path.join(args.model_path,
                              'model-%d.pkl'%(args.start_epoch))).state_dict())

    # Training
    for epoch in range(args.start_epoch, args.num_epochs):
        scheduler.step()
        model.train()

        m_hat, loss_ce, loss_rmse = model(rating_train)

        loss = loss_rmse
        reset_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():

            m_hat, loss_ce, loss_rmse = model(rating_val)

        print('epoch: '+str(epoch+1)+'[val loss] : '+str(loss_ce.item())+
              ' [val rmse] : '+str(loss_rmse.item()))
        print('alpha/beta: ', model.gcl1.alpha.item(), model.gcl1.beta.item(), model.gcl1.gamma.item())
        if best_loss > loss_rmse.item():
            best_loss = loss_rmse.item()
            best_epoch= epoch+1
            torch.save(model.state_dict(), os.path.join(args.model_path, 'model-%d.pkl'%(best_epoch)))

def test():
    # Test
    model.load_state_dict(torch.load(os.path.join(args.model_path,
                          'model-%d.pkl'%(best_epoch))))
    model.eval()
    with torch.no_grad():

        m_hat, loss_ce, loss_rmse = model(rating_test)

    print('[test loss] : '+str(loss_ce.item()) +
          ' [test rmse] : '+str(loss_rmse.item()))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        best_epoch = args.test_epoch
    test()
