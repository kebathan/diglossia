from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from classify import load_data
from umap import UMAP
from torch import no_grad, device, cuda
from tqdm import tqdm
import pandas as pd
from plotnine import ggplot, aes, geom_point, facet_wrap, theme, element_blank
import argparse

@no_grad()
def visualise(batch_size=16):
    # load data
    datasets = {
        "irumozhi": load_data("regdata", "none", augment=False)[:2],
        "dakshina": load_data("regdata", "dakshina", augment=False)[2:],
        "cc100": load_data("regdata", "cc100", augment=False)[2:],
        "tamilmixsentiment": load_data("regdata", "tamilmixsentiment", augment=False)[2:],
        "offenseval": load_data("regdata", "offenseval", augment=False)[2:],
        "hope_edi": load_data("regdata", "hope_edi", augment=False)[2:],
    }

    all_data = []
    for model_name in ["aryaman/xlm-roberta-base-irumozhi", "xlm-roberta-base"]:
        print(model_name)
        # load model
        dev = device("cuda" if cuda.is_available() else "cpu")
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print(tokenizer.model_max_length)
        model = XLMRobertaForSequenceClassification.from_pretrained(model_name).to(dev)
        model.eval()

        # embed
        data = []
        for name in datasets:
            print(name)
            dataset = datasets[name][0]
            labels = datasets[name][1]
            for i in tqdm(range(0, len(dataset), batch_size)):
                sentences = dataset[i:i+batch_size]
                tokenized = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(dev)
                last = model.roberta(**tokenized).last_hidden_state
                for j in range(len(sentences)):
                    data.append({
                        "Sentence": sentences[j],
                        "Embedding": last[j, 0].detach().cpu().numpy(),
                        "Label": labels[i] if name in ["irumozhi", "dakshina"] else "unknown",
                        "Dataset": name,
                        "Model": model_name.split('/')[-1]
                    })
        
        # umap
        umap = UMAP(n_components=2)
        embeddings = umap.fit_transform([d["Embedding"] for d in data])
        for i, d in enumerate(data):
            del d["Embedding"]
            d["x"] = embeddings[i][0]
            d["y"] = embeddings[i][1]
        all_data.extend(data)

    # make df
    df = pd.DataFrame(all_data)
    print(len(df))
    plot = (ggplot(df, aes(x="x", y="y", fill="Label", color="Label", shape="Dataset"))
            + geom_point(alpha=0.5) + facet_wrap("Model", scales="free")
            + theme(axis_ticks_major_x=element_blank(),
                   axis_ticks_major_y=element_blank(),
                   axis_text_x=element_blank(),
                   axis_text_y=element_blank(),
                   axis_title_x=element_blank(),
                   axis_title_y=element_blank(),))
    plot.save(f"figures/umap.pdf")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    visualise(args.batch_size)

if __name__ == "__main__":
    main()
