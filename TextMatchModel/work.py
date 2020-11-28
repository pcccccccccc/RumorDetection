
import torch
from sys import platform
from torch.utils.data import DataLoader
from data import LCQMC_Dataset, load_embeddings
from model import SiaGRU
from utils import get_score
import pandas as pd
import re

def init_csv(sequences, references, save_file):
    sequences = [seq.strip() for seq in sequences if len(seq.strip())!=0]
    references = [ref.strip() for ref in references if len(ref.strip())!=0]

    sentence1 = [seq for ref in references for seq in sequences]
    sentence2 = [ref for ref in references for seq in sequences]
    labels = [1 for ref in references for seq in sequences]
    df = pd.DataFrame({'sentence1': sentence1, 'sentence2': sentence2, 'label':labels})
    df.to_csv(save_file)


def main(inputs, vocab_file, embeddings_file, pretrained_file, max_length=50, gpu_index=0, batch_size=128):
    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
    # print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    embeddings = load_embeddings(embeddings_file)
    # print("\t* Loading test data...")
    # test_data = LCQMC_Dataset(test_file, vocab_file, max_length)
    # test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    # print("\t* Building model...")
    model = SiaGRU(embeddings, device=device).to(device)
    model.load_state_dict(checkpoint["model"])
    # print(20 * "=", " Testing SiaGRU model on device: {} ".format(device), 20 * "=")

    database = [line.strip() for line in open('./data/rumors.txt', 'r', encoding='utf-8') if len(line.strip())>0]

    # input("enter to continue")
    # inputs = [line for line in open('./data/input.txt', 'r', encoding='utf-8')]
    inputs = re.split(r"。|\.|？|\n", inputs)
    inputs = [seq.strip() for seq in inputs if len(seq.strip())!=0]
    init_csv(inputs, database, './data/work_data.csv')
    dataset = LCQMC_Dataset('./data/work_data.csv', vocab_file, max_length)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    prob = get_score(model, dataloader)
    ans=""
    for i, p in enumerate(prob):
        if p > 0.5:
            ans += inputs[i//len(database)]+"$"
            ans += database[i % len(database)]+"$"
            ans += str(p.item())+"@"
    return ans


def run(inputs):
    return main(inputs, "./data/vocab.txt", "./data/token_vec_300.bin", "./models/best.pth.tar")


# if __name__ == '__main__':
#     main("./data/vocab.txt", "./data/token_vec_300.bin", "./models/best.pth.tar")