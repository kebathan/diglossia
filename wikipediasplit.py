import csv
import random
from nltk import tokenize

def tamToEng(string):

    uyir = {"அ" : "a", "ஆ" : "ā",  "இ" : "i", "ஈ" : "ī", "உ" : "u", "ஊ" : "ū", "எ" : "ye", "ஏ" : "yē", "ஐ" : "ai", "ஒ" : "o", "ஓ" : "ō", "ஔ" : "au"}
    mei = {"க" : ["ga", "ka"], "ங" : ["ṅa"], "ச" : ["sa", "cha"], "ஞ" : ["ña"], "ட" : ["ḍa", "ṭa"], "ண" : ["ṇa"], "த" : ["dha", "tha"], "ந" : ["na"], "ப" : ["ba", "pa"], "ம" : ["ma"], "ய" : ["ya"], "ல" : ["la"], "ர" : ["ra"], "வ" : ["va"], "ழ" : ["zha"], "ள" : ["ḷa"], "ற" : ["ṟa", "ṯṟa"], "ன" : ["ṉa"], "ஷ" : ["sha"], "ஸ" : ["sa"], "ஹ" : ["ha"], "ஜ" : ["ja"]}
    marks = {"ா" : "ā", "ி" : "i", "ீ" : "ī", "ு" : "u", "ூ" : "ū", "ெ" : "e", "ே" : "ē", "ை" : "ai", "ொ" : "o", "ோ" : "ō", "ௌ" : "au"}
    pulli = "்"

    vallinam = ["க", "ச", "ட", "த", "ப", "ற"]

    processed = list(string.strip())
    final = []

    idx = 0
    pullied = False

    for i in range(len(processed)):

        letter = processed[i]
        
        # adds uyir ezhuthukkal
        if letter in uyir.keys():
            final.append(uyir[letter])
            idx += 1

        # adds mei ezhuthukkal
        elif letter in mei.keys():

            # checks for word-initial vallinam ezhuthukkal
            if letter in vallinam and (idx == 0 or processed[i-1] in ' !"#$%&\'()*+,-./0123456789:;<=>?[\\]^_`{|}~'):
                final.append(mei[letter][1])
                idx += 1

            # checks for rule changes
            elif pullied and letter in vallinam:

                # adding "nj"
                if processed[i-2] == "ஞ" and letter == "ச":
                    final[idx-1] = "n"; final.append("ja")
                    idx += 1

                # adding "ndr"
                elif processed[i-2] == "ன" and letter == "ற":
                    final[idx-1] = "ṉ"; final.append("ḏra")
                    idx += 1

                # adding "tr"
                elif processed[i-2] == "ற" and letter == "ற":
                    final[idx-1] = "ṯ"; final.append("ṟa")
                    idx += 1

                # geminated consonants
                elif processed[i-2] == letter:
                    if len(mei[letter][1]) == 2:
                        final.append(final[idx-1])
                        idx += 1
                    final[idx-1] += "a"
                    

                # double vallinam
                elif processed[i-2] in vallinam:
                    final.append(mei[letter][1])
                    idx += 1

                # regular consonant + vallinam  
                else:
                    final.append(mei[letter][0])
                    idx += 1

                pullied = False
                
            # adds regular mei ezhuthukkal
            else:
                final.append(mei[letter][0])
                idx += 1
        
        # adds diacritic vowel markings
        elif letter in marks.keys():

            # kutriyalugaram: shortening of the word-final "u"
            if letter == "ு" and (i == len(processed)-1 or processed[i+1] == " "):
                final[idx-1] = final[idx-1][:-1] + "ŭ"

            # regular vowels
            else:
                final[idx-1] = final[idx-1][:-1] + marks[letter]

        # removes inherent vowels
        elif letter == pulli:
            pullied = True
            if processed[i-1] not in vallinam or processed[i-1] == "ற":
                final[idx-1] = final[idx-1][:-1]
            else:
                final[idx-1] = mei[processed[i-1]][1][:-1]
            
        # adds any not-letters
        else:
            final.append(letter)
            idx += 1
            pullied = False

    # returns final string
    return "".join(final)

all_sentences = []

for j in range(100):

    k = str(j)
    if j < 10:
        k = "0" + k

    file = open("/Users/ashok/Documents/Kabi/Funsies/drive-download-20230705T013735Z-001/AA_wiki_" + k + ".txt", "r+")
    text = file.read()
    file.close()

    all_sentences.extend(list(tokenize.sent_tokenize(text)))


random.shuffle(all_sentences)

large = open("/Users/ashok/Documents/Kabi/Funsies/datacollection.csv", "w", encoding='UTF8', newline='')
small = open("/Users/ashok/Documents/Kabi/Funsies/datacollectionless.csv", "w", encoding='UTF8', newline='')

writer1 = csv.writer(large)
writer1.writerow(["tamil", "transliterated", "colloquial"])
writer2 = csv.writer(small)
writer2.writerow(["tamil", "transliterated", "colloquial"])

count = 0
for token in all_sentences:
    if "<" in token or ">" in token or "https" in token or "doc" in token or "url" in token or "wikipedia" in token or len(token) == 0 or "\n" in token:
        pass
    else:
        if count < 10000:
            writer2.writerow([token, tamToEng(token)])
        writer1.writerow([token, tamToEng(token)])

    count += 1

small.close()
large.close()
