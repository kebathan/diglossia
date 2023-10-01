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
        ungem = gem[:int(len(gem)/2)]
        text = text.replace(gem, ungem)

    return text


def vowel_orth(text, vowels):

    vows = {"ee" : "ii",
            "oo" : "uu",
            "aa" : "a",
            "ae" : "e"}
    
    for vow in vowels:
        text = text.replace(vow, vows[vow])

    return text


def h_g(text):

    vowels = ("a", "e", "i", "o", "u")
    idx = 0

    while text[idx+1:].find("h") != -1:

        idx += text[idx:].find("h")
        
        if idx > 0 and idx < len(text) - 1:
            if text[idx-1] in vowels and text[idx+1] in vowels:
                text = text[:idx] + "g" + text[idx+1:]
            else: 
                idx += 1
        else:
            idx += 1

    return text


def ch_s(text):
    if text[:2] == "ch":
        text = "s" + text[2:]
    return text.replace(" ch", " s")

def le_la(text):
    return text.replace("le ", "la ")

test = "changar mattrum ivargal taj mahalil thamizh puththagangalai padippaargal"

print(test)
#print(h_g(ch_s(orth_replace(orth_replace(gemination(accusative(locative(plural(test))), ["thth", "pp"]), ("zh", "l")), ("th", "t")))))

print(ch_s(test))
print(gemination(test, ["thth", "pp"]))
print(h_g(test))
print(locative(test))
print(le_la(locative(test)))