import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
from googletrans import Translator
translator = Translator()
eval = load_dataset("mlfoundations/VisIT-Bench")
moondream = load_dataset("Q-bert/moondreamv2-out")
vllm = load_dataset("Q-bert/vllm-out")
z = load_dataset("Q-bert/vllm-out2")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").cuda()

gpt4 = eval["test"]["gpt4_prediction"]
answer = eval["test"]["instruction_conditioned_caption"]
mondrm = moondream["train"]['text']
vllm = [i.text for i in translator.translate(vllm["train"]['text'] + z["train"]['text'], dest='en')]
print(vllm)

def encode(lst):
    encoded_texts = tokenizer(lst, return_tensors='pt', padding=True, truncation=True).to('cuda')
    with torch.no_grad():
        outputs = model(**encoded_texts)
    return outputs.last_hidden_state[:, -1, :].detach().cpu().numpy()
n = len(vllm)
a = encode(gpt4[:n])
b = encode(answer[:n])
c = encode(mondrm[:n])
d = encode(vllm[:n])



def average_cosine_similarity(x, y):
    sim = 0
    for i in range(len(x)):
      sim += cosine_similarity(x, y)[i][i]
    return sim/len(x)

cosine_sim_matrix = np.zeros((4, 4))
datasets = [a, b, c, d]
for i in range(4):
    for j in range(4):
        cosine_sim_matrix[i, j] = average_cosine_similarity(datasets[i], datasets[j])

labels = ['GPT-4', 'VisIT-Bench', 'MoondreamV2', """Ours"""]
plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim_matrix, annot=True, cmap='viridis', xticklabels=labels, yticklabels=labels)
plt.title('VisIT-Bench Cosine Similarity Confusion Matrix')
plt.show()
