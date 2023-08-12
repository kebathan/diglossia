# analyses

from plotnine import ggplot, aes, geom_histogram
from levenshteindist import lev2
import pandas as pd
import csv

# LEVENSHTEIN DISTANCE >>>

dists = []
normalized_dists = []
sents = []

with open("ipadataset.csv") as f:

    reader = csv.DictReader(f)
    
    for row in reader:

        sents.append(row["transliterated"])
        annotator1 = row["colloquial: annotator 1"]
        annotator2 = row["colloquial: annotator 2"]

        dist = lev2(annotator1, annotator2)
        normalized = dist/max(len(annotator1), len(annotator2))

        dists.append(dist)
        normalized_dists.append(normalized)


d = {"original" : sents, "dists" : dists, "normalized dists" : normalized_dists}
dist_dataframe = pd.DataFrame(data=d)


print("max dist: " + str(max(dists)))
print(f"average dist {sum(dists)/len(dists)}\n")

print(f"max normalized dist: {max(normalized_dists)}")
print(f"average normalized dists {sum(normalized_dists)/len(normalized_dists)}\n")

print(dist_dataframe.loc[dist_dataframe["normalized dists"].idxmax()])


plot = ggplot(dist_dataframe, aes(x="normalized dists")) + geom_histogram(bins=100)
print(plot)