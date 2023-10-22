import gensim.downloader as g1
# See Documentation for other models that can be downloaded
# model = g1.load("glove-wiki-gigaword-100")
# Notice that the model is of class genism.models.keyedvectors, see the
# API for that model to see what methods you can use
# Below is just a small sample of it (and the name are self-#explanatory)
# x = model.most_similar('board')
# print(x)
# vec = model.get_vector("king")
# vec2 = model.get_vector("man")
# vec3 = model.get_vector("woman")

# print(vec)
# print(model.most_similar())
# print(vec2)



def simValues(model,key,countList):
    return [model.most_similar(key,topn=c)[-1][1] if c >0 and isinstance(c,int) else -10000.0 for c in countList]
def simValuesPct(model,key,countPctList):
    rtn = []
    for pct in countPctList:
        wordz = model.most_similar(key,topn=None)
        idx = int(pct * len(wordz)/100)
        if idx < 0 or idx >= len(wordz):
            rtn.append(-10000.0)
        else:
            rtn.append(wordz[idx])
        print(idx)
    return rtn

def getDistribution(model,nthWord=0,maxItems=1000):
    return [simValues(model,t,[nthWord]) for t in model.index_to_key[:maxItems]]

if __name__ == "__main__":
    model = g1.load("glove-wiki-gigaword-100")

    # l1 = simValues(model, "board", [1, 5, 50, 10, -5])
    # print(l1)
    print(f'Num Words: {type(model.index_to_key)}')
    l1 = simValuesPct(model, "board", [0, 0.5, 0.75, 0.1, -5,99.9])
    print(l1)
    # print(getDistribution(model,1))
    