# analyses

from plotnine import ggplot, ggtitle, aes, geom_histogram
from levenshteindist import lev2
import pandas as pd
import csv

from nltk.translate.bleu_score import sentence_bleu

# LEVENSHTEIN DISTANCE >>>

sents = []

ann1 = []
ann2 = []

dists_aa = []
normalized_dists_aa = []

dists_a1 = []
normalized_dists_a1 = []

dists_a2 = []
normalized_dists_a2 = []

with open("regdataset.csv") as f:

    reader = csv.DictReader(f)
    
    for row in reader:

        transliterated = row["transliterated"]
        annotator1 = row["colloquial: annotator 1"]
        annotator2 = row["colloquial: annotator 2"]

        ann1.append(annotator1)
        ann2.append(annotator2)

        # get regular and normalized levenshtein distances between each pair
        dist_aa = lev2(annotator1, annotator2)
        normalized_aa = dist_aa/max(len(annotator1), len(annotator2))

        dist_a1 = lev2(transliterated, annotator1)
        normalized_a1 = dist_a1/max(len(transliterated), len(annotator1))

        dist_a2 = lev2(transliterated, annotator2)
        normalized_a2 = dist_a2/max(len(transliterated), len(annotator2))


        # append distances to lists
        sents.append(transliterated)

        dists_aa.append(dist_aa)
        normalized_dists_aa.append(normalized_aa)

        dists_a1.append(dist_a1)
        normalized_dists_a1.append(normalized_a1)

        dists_a2.append(dist_a2)
        normalized_dists_a2.append(normalized_a2)
        

d_aa = {"original" : sents, "dists" : dists_aa, "normalized dists" : normalized_dists_aa}
df_aa = pd.DataFrame(data=d_aa)

d_a1 = {"original" : sents, "dists" : dists_a1, "normalized dists" : normalized_dists_a1}
df_a1 = pd.DataFrame(data=d_a1)

d_a2 = {"original" : sents, "dists" : dists_a2, "normalized dists" : normalized_dists_a2}
df_a2 = pd.DataFrame(data=d_a2)

data = {"original" : sents, "annotator 1" : ann1, "annotator 2" : ann2}
dataset = pd.DataFrame(data=data)

# print("max dist: " + str(max(dists_aa)))
# print(f"max normalized dist: {max(normalized_dists_aa)}")
# print(df_aa.loc[df_aa["normalized dists"].idxmax()])

print("\nbetween two annotators:\n-----------")
print(f"average dist {sum(dists_aa)/len(dists_aa)}")
print(f"average normalized dists {sum(normalized_dists_aa)/len(normalized_dists_aa)}\n\n")

print("between transliteration and annotator 1:\n-----------")
print(f"average dist {sum(dists_a1)/len(dists_a1)}")
print(f"average normalized dists {sum(normalized_dists_a1)/len(normalized_dists_a1)}\n\n")

print("between transliteration and annotator 2:\n-----------")
print(f"average dist {sum(dists_a2)/len(dists_a2)}")
print(f"average normalized dists {sum(normalized_dists_a2)/len(normalized_dists_a2)}\n\n")


plot = ggplot(df_aa, aes(x="normalized dists")) + geom_histogram(bins=100) + ggtitle("between two annotators")
print(plot)

plot1 = ggplot(df_a1, aes(x="normalized dists")) + geom_histogram(bins=100) + ggtitle("between transliteration + annotator 1")
print(plot1)

plot2 = ggplot(df_a2, aes(x="normalized dists")) + geom_histogram(bins=100) + ggtitle("between transliteration + annotator 2")
print(plot2)


# BLEU SCORE >>>

# test_og = dataset.iloc[0]["original"]
# test_ann1 = dataset.iloc[0]["annotator 1"]
# test_ann2 = dataset.iloc[0]["annotator 2"]

# tests = [test_og.split(), test_ann1.split(), test_ann2.split()]

# print('BLEU score -> {}'.format(sentence_bleu(tests, "seenivaasan thayaarichaaru.".split())))