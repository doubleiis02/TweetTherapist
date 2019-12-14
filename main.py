import numpy as np
import matplotlib.pyplot as plt
from data import x_train, y_train, x_test, y_test, x_train_str, x_test_str, stops, glove
from model import *
from noggin import create_plot
from mynn.optimizers.adam import Adam
from data import strip_punc
#from model import *




        

def get_embedding(text):
    if text in glove:
        return glove.get_vector(text)[:, np.newaxis]


def vectorize(x, maxdim1=0):
    posts = []
    #x_wordlist = []
    z = np.zeros((50, 1))
    if maxdim1 > 0:
        for post in x:
            wordlist = post.split()
            m = len(wordlist)
            for word in wordlist:
                if word in stops:
                    if m != 1:
                        wordlist.pop(wordlist.index(word))
            
            if len(wordlist) < maxdim1:
                diff = maxdim1 - len(wordlist)
                if diff % 2 == 0:
                        wordlist = np.pad(wordlist,(diff//2, diff//2), 'constant')
                        wordlist = list(wordlist)
                else:
                    wordlist = np.pad(wordlist, (diff//2, diff//2 + 1), 'constant')
                    wordlist = list(wordlist)
            
            """
            if m > 2:
                wordlist = [get_embedding(i) if i in glove else z for i in wordlist], axis = 1]
            if m == 2:
                wordlist = [z, get_embedding(wordlist[0]), get_embedding(wordlist[1]), z]
            else:
                wordlist = [z, get_embedding(wordlist[0]), z]
            if len(wordlist) < maxdim1:
                diff = maxdim1 - len(wordlist)
                if diff % 2 == 0:
                        wordlist = np.pad(wordlist,(diff//2, diff//2), 'constant')
                        wordlist = list(wordlist)
                else:
                    wordlist = np.pad(wordlist, (diff//2, diff//2 + 1), 'constant')
                    wordlist = list(wordlist)
            """
            
            wordlist = np.concatenate([get_embedding(i) if i in glove else z for i in wordlist], axis = 1)   
            posts.append(wordlist)
            if wordlist.shape[1] != maxdim1:
                print(wordlist.shape)
          
    
        posts = tuple(posts)
        overall = np.concatenate(posts, axis=0).reshape(-1, 50, maxdim1)
        overall = overall.astype(np.float32)
        for post in overall:
            post /= np.linalg.norm(post)
    if maxdim1 == 0:
        wordlist = x.split()
        m = len(wordlist)
        for word in wordlist:
            if word in stops:
                if m != 1:
                    wordlist.pop(wordlist.index(word))
        m = len(wordlist)
        
        if m > 1:
            wordlist = np.concatenate([get_embedding(i) if i in glove else z for i in wordlist], axis = 1)
        elif m == 1:
            wordlist = get_embedding(wordlist[0])
        if m < 3:
            wordlist = np.hstack((z, wordlist, z))
        m = wordlist.shape[1]
        wordlist = wordlist.reshape(-1, 50, m)
        overall = wordlist.astype(np.float32)
        overall /= np.linalg.norm(overall)
    return overall

    
def train(x_tr, y_tr, x_te, y_te):
    x_tr = np.array(x_tr)
    y_tr = np.array(y_tr)
    x_train_vec = vectorize(x_tr, 92)
    x_te = np.array(x_te)
    y_te = np.array(y_te)
    x_test_vec = vectorize(x_te, 58)
    model = Model()
    optim = Adam(model.parameters, learning_rate = 1e-4)
    plotter, fig, ax = create_plot(metrics=["loss", "accuracy"])
    
    
    batch_size = 50
    
    for epoch_cnt in range(7):
        idxs = np.arange(len(x_tr))
        np.random.shuffle(idxs)
           
        for batch_cnt in range(len(x_tr)//batch_size):
            # make slice object so indices can be referenced later
            batch_indices = idxs[slice(batch_cnt * batch_size, (batch_cnt + 1) * batch_size)]
            #batch = x_train[batch_indices]  # random batch of our training data
            
            # retrieve glove embeddings for batch
            # initialize every value as small number which will be the placeholder for not found embeddings
            arr = x_train_vec[batch_indices]
            """
            arr = np.ones((len(batch), 200, max(train_max, test_max))) / 1000000
            for i, sent in enumerate(batch):
                for j , word in enumerate(sent):   
                    # retrieve glove embedding for every word in sentence
                    try:
                        arr[i,:,j] = glove_model.get_vector(word.lower())
                    
                    # continue if glove embedding not found
                    except Exception as e:
                        continue
            """
            
            
            # pass model through batch and perform gradient descent
            pred = model(arr)
            truth = y_tr[batch_indices]
            
            loss = binary_cross_entropy(pred[:,0], truth)
            loss.backward()
    
            optim.step()
            loss.null_gradients()
            
            acc = accuracy(pred[:,0], truth)
            
            # pass loss and accuracy to noggin for plotting
            plotter.set_train_batch({"loss" : loss.item(),
                                     "accuracy" : acc},
                                     batch_size=batch_size)
            """
                return model
            
            
            
            def test(x_te, y_te, model):     
                # compute test statistics
                #idxs = np.arange(len(x_test))
                
                x_te = np.array(x_te)
                y_te = np.array(y_te)
                x_test_vec = vectorize(x_te, 58)
                logger = LiveLogger()
                idxs = np.arange(len(x_te))
                logger.set_train_batch(dict(metric_a=50., metric_b=5.), batch_size=50)
                batch_size = 50
                
                for epoch_cnt in range(2):
                    idxs = np.arange(len(x_te))
                    np.random.shuffle(idxs)
                """
        idxs = np.arange(len(x_te))
        for batch_cnt in range(0, len(x_te) // batch_size):
            batch_indices = idxs[slice(batch_cnt * batch_size, (batch_cnt + 1) * batch_size)]
            #batch = x_test[batch_indices]
            
            # again, find embeddings for batch
            arr = x_test_vec[batch_indices]
            """
            arr = np.ones((len(batch), 200, max(train_max, test_max))) / 1000000
            for i, sent in enumerate(batch):
                for j , word in enumerate(sent):   
                    try:
                        arr[i,:,j] = glove_model.get_vector(word.lower())
                    
                    except Exception as e:
                        continue
            """
            
            # perform forward pass and find accuracy but DO NOT backprop
            pred = model(arr)
            truth = y_te[batch_indices]
            acc = accuracy(pred[:,0], truth)

           
            # log the test-accuracy in noggin
            plotter.set_test_batch({"accuracy" : acc},
                                     batch_size=batch_size)
       
        # plot the epoch-level train/test statistics
        plotter.set_train_epoch()
        plotter.set_test_epoch()
    return model    
"""
    batch_array, epoch_array = logger.to_xarray("test")
    print(batch_array)
    print(epoch_array)
    
 """   
    
 
 
#model = train(x_train, y_train, x_test, y_test)
 
def main_func(model):
    inp = input("Enter text to analyze, or press q to quit\n")
    while inp!="q":
        if inp == None or inp.replace(" ", "") == "":
            print("Please try again")
        else:
            inp = strip_punc(inp).lower()
            wl = inp.split()
            for w in wl:
                if w not in glove:
                    wl.pop(wl.index(w))
            if wl == []:
                print("The word(s) in this text could not be analyzed :( \nPlease try again\n")
            else:
                vector = vectorize(inp)
                pred = model(vector)
                pred = pred[:,0]
                pred = np.round(pred.data)
                if pred == 0:
                    print("You seem perfectly fine!\n")
                if pred == 1:
                    print("Oof, seems like you may have depression. If you ever start having dark thoughts, I suggest you seek help and/or call 1-800-273-8255.\n")
        inp = input("Enter text to analyze, or press q to quit\n")