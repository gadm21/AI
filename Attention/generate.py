

from transformers import GTransformer
from utils import enwik









def train(config):

  torch.manual_seed(1)

  train_data, val_data, test_data = enwik('enwik9')

  model = GTransformer(config)

  opt = torch.optim.Adam(lr = config['lr'], params = model.parameters())
  #sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min())
  
  for i in tqdm.trange(config['batch_size']):
    opt.zero_grad() #sets the gradients of all tensors to zero

    start_inds = torch.randint(size = (config['batch_size'],), low = 0, high = train_data.size(0) - config['seq_length']-1)
    input_seqs = [ train_data[ind: ind+ config['seq_length']] for ind in start_inds]
    out_seqs = [ train_data[ind+1: ind+ config['seq_length']+1] for ind in start_inds]
    source = torch.cat([seq[None, :] for seq in input_seqs], dim = 0).to(torch.long)
    target = torch.cat([seq[None, :] for seq in out_seqs], dim = 0).to(torch.long)

    output = model(source)

    loss = F.nll_loss(output.transpose(2,1), target, reduction = 'mean')
    
    loss.backward()
    opt.step()





def main():
    config = {
        'num_tokens': 256,
        'depth': 3,
        'emb': 120,
        'seq_length': 200,
        'batch_size': 32,
        'lr': 0.0001,
        'wide': False,
        'heads': 4,
        'mask': True,
        'ff_hidden_mult': 3,
        'dropout': 0.1

    }
    train(config)


if __name__ == "__main__":
    main() 