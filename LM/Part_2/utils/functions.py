import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import math

# Device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def collate_fn(data, pad_token):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    # here return the sum of the loss, not the mean
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)
    
def store_result(task_name, epochs, train_loss_values, dev_loss_values, 
                         train_ppl_values, dev_ppl_values, best_ppl, final_ppl, 
                         opt_method, model, best_model, config):
    # Set up directory structure
    base_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_path, "experiment_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Generate unique folder name
    existing_experiments = len([x for x in os.listdir(results_dir) if x.startswith(task_name)])
    experiment_title = f"{task_name}_run_{existing_experiments + 1}_PPL_{int(final_ppl)}"
    experiment_path = os.path.join(results_dir, experiment_title)
    os.makedirs(experiment_path, exist_ok=True)

    # Plot and save loss curves
    plt.figure()
    plt.plot(epochs, train_loss_values, '-', label='Training')
    plt.plot(epochs, dev_loss_values, '-', label='Development')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(experiment_path, "Loss_Training_vs_Dev.pdf"))

    # Plot and save perplexity curves
    plt.figure()
    plt.plot(epochs, train_ppl_values, '-', label='Training')
    plt.plot(epochs, dev_ppl_values, '-', label='Development')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.savefig(os.path.join(experiment_path, "Perplexity_Training_vs_Dev.pdf"))

    # Save training details to text file
    output_file = os.path.join(experiment_path, "experiment_details.txt")
    with open(output_file, "w") as f:
        f.write(f"Experiment: {task_name}\n\n")
        for param_name, param_value in config.items():
            f.write(f"{param_name} = {param_value}\n")
        f.write(f"Optimal Dev Perplexity: {best_ppl}\n")
        f.write(f"Final Test Perplexity: {final_ppl}\n")
        f.write(f"Optimization Method: {opt_method}\n")
        f.write(f"Model Type: {model}\n")

    # Save the best model parameters
    torch.save(best_model.state_dict(), os.path.join(experiment_path, "trained_model.pt"))