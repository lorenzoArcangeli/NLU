from utils.classes import *
from models import *
from utils.functions import * 
import os
import copy
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader


# Main function
if __name__ == "__main__":
    
    # Device
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # HYPERPARAMETERS
    config = {
        "batch_size_train": 32,
        "batch_size_dev": 128,
        "batch_size_test": 128,
        "hid_size": 400,
        "emb_size": 500,
        "lr": 3, # 3 with SGD, 0.001 with adamw
        "clip": 5, # normalize the gradient if >5
        "n_epochs": 100,
        "patience": 3
    }
    
    # LOADING DATA
    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_raw = read_file(os.path.join(current_dir, "dataset/PTB/train.txt"))
    dev_raw = read_file(os.path.join(current_dir, "dataset/PTB/valid.txt"))
    test_raw = read_file(os.path.join(current_dir, "dataset/PTB/test.txt"))

    #get vocab and trasform in id
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    # Create datasets and dataloaders
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size_train"], 
                              collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config["batch_size_dev"], 
                            collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size_test"], 
                             collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    vocab_len = len(lang.word2id)

    # TRAINING
    model = LM_LSTM_DROP(config["emb_size"], config["hid_size"], vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    #model = LM_LSTM(config["emb_size"], config["hid_size"], vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    opt_method = optim.SGD(model.parameters(), lr=config["lr"])
    #opt_method = optim.AdamW(model.parameters(), lr=config["lr"])
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    patience = 3 # Early stopping
    train_loss_values = [] # Store the losses
    dev_loss_values = []
    dev_ppl_values = [] # Store the perplexity
    train_ppl_values = []
    epochs = [] # Store the epochs
    best_ppl = math.inf
    best_model = None   
    pbar = tqdm(range(1, config["n_epochs"] + 1))
    
    # Training loop
    for epoch in pbar:
        # Train
        loss_train = train_loop(train_loader, opt_method, criterion_train, model, config["clip"])
        train_ppl_values.append(loss_train)
        
        # Evaluate
        if epoch % 1 == 0:
            epochs.append(epoch)
            train_loss_values.append(np.asarray(loss_train).mean())
            
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            dev_ppl_values.append(ppl_dev)
            dev_loss_values.append(np.asarray(loss_dev).mean())
            
            pbar.set_description(f"PPL: {ppl_dev:.4f}")
            
            # Early stopping
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1

            if patience <= 0:
                break

    # Test
    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)

    # Save results
    task_name = "1_1"
    store_result(task_name, epochs, train_loss_values, dev_loss_values, 
                         train_ppl_values, dev_ppl_values, best_ppl, final_ppl, 
                         opt_method, model, best_model, config)