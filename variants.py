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


ex = "ivargal palliyil puthangangalai padippaargal"

print(ex)
print(accusative(locative(plural(ex))))
