from utils import *
from models import *
from functions import * 
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
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA sid

    # Configuration/hyeperparameters
    config = {
        "batch_size_train": 32, #original 128
        "batch_size_dev": 128, #original 64
        "batch_size_test": 128, #original 64
        "hid_size": 500, #original 200
        "emb_size": 500, #original 300
        "lr": 0.0001, 
        "clip": 5, 
        "n_epochs": 200,
        "patience": 3
    }
    
    # load data
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tmp_train_raw = load_data(os.path.join(current_dir, 'dataset', 'train.json'))
    test_raw = load_data(os.path.join(current_dir, 'dataset', 'test.json'))

    train_raw, dev_raw = get_dev(tmp_train_raw)


    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute 
                                                            # the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    lang = Lang(words, intents, slots, cutoff=0)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size_train'], collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size_dev'], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], collate_fn=collate_fn)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS(config['hid_size'], out_slot, out_int, config['emb_size'], vocab_len, pad_index=PAD_TOKEN).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    patience = config['patience']
    for x in tqdm(range(1,config['n_epochs'])):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model, clip=config['clip'])
        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                        criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                # Here you should save the model
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                            criterion_intents, model, lang)    
    f1_test= results_test['total']['f']
    accuracy_test= results_test['accuracy']
    print('Slot F1: ',f1_test)
    print('Intent Accuracy:', accuracy_test)

    '''
    # Save results
    task_name = "1_1"
    store_result(task_name, epochs, losses_train, losses_dev,
                opt_method, model, best_model, config, f1_test, accuracy_test, lang)
    '''
