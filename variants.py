# create colloquial variants

def plural(text):
    
    text = text.split()

    for i in range(len(text)):
        if text[i][-4:] in ["ngal", "rgal"]:
            text[i] = text[i][:-4] + "nga"
        elif "rgal" in text[i]:
            idx = text[i].find("rgal")
            text[i] = text[i][:idx] + "ngal" + text[i][idx+4:]

    return " ".join(text)


def locative(text): 

    text = text.split()

    for i in range(len(text)):
        if text[i][-3:] == "yil":
            text[i] = text[i][:-3] + 'le'
        elif text[i][-2:] == "il":
            text[i] = text[i][:-2] + "le"

    return " ".join(text)


def accusative(text):

    text = text.split()

    for i in range(len(text)):
        if text[i][-2:] == "ai":
            text[i] = text[i][:-2] + "e"

    return " ".join(text)


def gemination(text, geminate):
    
    for gem in geminate:
        while text.lower().find(gem) != -1:
            ungem = gem[:int(len(gem)/2)]
            text = text.replace(gem, ungem)

    return text


def vowel_orth(text, vowels):

    vows = {"ee" : "ii",
            "oo" : "uu",
            "aa" : "a",
            "ae" : "e"}
    
    for vow in vowels:
        while text.lower().find(vow) != -1:
            text = text.replace(vow, vows[vow])

    return text


def h_g(text):

    vowels = ("a", "e", "i", "o", "u")
    text = text.lower()
    idx = 0

    while text[idx+1:].find("h") != -1:

        idx = text[idx:].find("h")
        
        if idx > 0 and idx < len(text) - 1:
            if text[idx-1] in vowels and text[idx+1] in vowels:
                text = text.replace("h", "g")

    return text


def ch_s(text):

    text = text.lower().split()

    for i in range(len(text)):
        if text[i].find("ch") == 0:
            text[i] = text[i].replace("ch", "s")

    return " ".join(text)


def orth_replace(text, rep):

    while text.lower().find(rep[0]) != -1:
        text = text.replace(rep[0], rep[1])

    return text


test = "changar mattrum ivargal taj mahalil thamizh puththagangalai padippaargal"

print(test)
print(h_g(ch_s(orth_replace(orth_replace(gemination(accusative(locative(plural(test))), ["thth", "pp"]), ("zh", "l")), ("th", "t")))))
