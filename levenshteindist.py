# levenshtein distance 

def lev(a, b):

    if len(b) == 0:
        return abs(len(a))
    
    elif len(a) == 0:
        return abs(len(b))
    
    elif a[0] == b[0]:
        return lev(a[1:], b[1:])
    
    else:
        return 1 + min(lev(a[1:], b), lev(a, b[1:]), lev(a[1:], b[1:]))

def lev2(a, b):
    lev = {}

    for i in range(len(a) + 1): lev[i,0] = i 
    for j in range(len(b) + 1): lev[0,j] = j

    for r in range(1, len(a)+1):
        for c in range(1, len(b)+1):
            cost = 0 if a[r-1] == b[c-1] else 1
            lev[r, c] = min(lev[r, c-1] + 1, lev[r-1, c] + 1, lev[r-1, c-1] + cost)

    return lev[r, c]
