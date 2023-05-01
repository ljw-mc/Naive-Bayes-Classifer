import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """

    a = 1 # my current choice for alpha - the smoothing parameter

    # dictionary of words : spam_count[word] = count(word)
    spam_count = util.get_word_freq(file_lists_by_category[0])
    ham_count = util.get_word_freq(file_lists_by_category[1])

    # a list of keys from spam and ham ie. list of words
    spam_keys = spam_count.keys()
    ham_keys = ham_count.keys()


    # Obtain size of Vocab {V}
    V = 0
    for key in spam_keys:
        if spam_count[key] > 0:
            V+=1
    for key in ham_keys:
        if key not in spam_keys and ham_count[key] > 0:
            V+=1
    

    # Total Number of Words in All Spam & Ham Emails
    N_spam = 0
    N_ham = 0
    for key in spam_keys:
        N_spam+= spam_count[key]
    for key in ham_keys:
        N_ham+= ham_count[key]
    


    spam_prob = dict()
    ham_prob = dict()
    # p_d and q_d
    list_of_words = list()
    for word in spam_keys:
        if word not in spam_prob:
            spam_prob[word] = (spam_count[word] + 1) / (N_spam + V)#spam_count[word] / N_spam #(spam_count[word] + a) / (N_spam + a * V)
        list_of_words.append(word)
    for word in ham_keys:
        if word not in ham_prob:
            ham_prob[word] = (ham_count[word] + 1) / (N_ham + V)#ham_count[word] / N_ham #(ham_count[word] + a) / (N_ham + a * V)
        list_of_words.append(word)
    
    for word in list_of_words:
        if word not in spam_prob:
            spam_prob[word] = (spam_count[word] + 1) / (N_spam + V)
        if word not in ham_prob:
            ham_prob[word] = (ham_count[word] + 1) / (N_ham + V)
    
    
    spam_prob["THIS WORD IS NOT IN TRAINING"] = 1 / (N_spam + V)
    ham_prob["THIS WORD IS NOT IN TRAINING"] = 1 / (N_ham + V)

    return (spam_prob, ham_prob)


def myLogFactorial(x):
    logSum = 0
    for i in range(1, x+1):
        logSum += np.log(i)

    return logSum


def classify_new_email(filename,probabilities_by_category,prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    # Unpack Spam & Ham Probs
    spam_prob = probabilities_by_category[0]
    ham_prob = probabilities_by_category[1]

    # Record --- which word appeared & how many times
    word_freq = util.get_word_freq([filename])
    words = word_freq.keys()

    # log [ (x1 + x2 ... xD)! ] 
    email = util.get_words_in_file(filename)
    C = myLogFactorial(len(email))

    for word in words:
        C -= myLogFactorial(word_freq[word])


    logp = C
    logq = C
    for word in words:
        if word in spam_prob.keys():        
            logp += word_freq[word] * np.log(spam_prob[word])
        if word in ham_prob.keys():
            logq += word_freq[word] * np.log(ham_prob[word])
        # elif word not in spam_prob.keys():
        #     logp += word_freq[word] * np.log(spam_prob["THIS WORD IS NOT IN TRAINING"])
        # elif word not in ham_prob.keys():
        #     logq += word_freq[word] * np.log(ham_prob["THIS WORD IS NOT IN TRAINING"])
    
    b = [logp + np.log(prior_by_category[0]), logq + np.log(prior_by_category[1])]
    B = max(b)

    log_p_spam = logp + np.log(prior_by_category[0]) - (B + np.log(np.exp(b[0] - B) + np.exp(b[1] - B)))
    log_p_ham = logq + np.log(prior_by_category[1]) - (B + np.log(np.exp(b[0] - B) + np.exp(b[1] - B)))# - np.log(np.exp(logp)*0.5 + np.exp(logq)*0.5)

    # if log_p_spam > log_p_ham:
    if log_p_spam > log_p_ham :
        return ("spam", [log_p_spam, log_p_ham])
    
    else:
        return ("ham", [log_p_spam, log_p_ham])

    #'data/testing/5037.2001-11-02.farmer.ham.txt'

def alpha_controlled_classify_new_email(t, filename,probabilities_by_category,prior_by_category):
    spam_prob = probabilities_by_category[0]
    ham_prob = probabilities_by_category[1]

    # Record --- which word appeared & how many times
    word_freq = util.get_word_freq([filename])
    words = word_freq.keys()

    # log [ (x1 + x2 ... xD)! ] 
    email = util.get_words_in_file(filename)
    C = myLogFactorial(len(email))

    for word in words:
        C -= myLogFactorial(word_freq[word])


    logp = C
    logq = C
    for word in words:
        if word in spam_prob.keys():        
            logp += word_freq[word] * np.log(spam_prob[word])
        if word in ham_prob.keys():
            logq += word_freq[word] * np.log(ham_prob[word])
    
    b = [logp + np.log(prior_by_category[0]), logq + np.log(prior_by_category[1])]
    B = max(b)

    log_p_spam = logp + np.log(prior_by_category[0]) - (B + np.log(np.exp(b[0] - B) + np.exp(b[1] - B)))
    log_p_ham = logq + np.log(prior_by_category[1]) - (B + np.log(np.exp(b[0] - B) + np.exp(b[1] - B)))# - np.log(np.exp(logp)*0.5 + np.exp(logq)*0.5)

    # if log_p_spam > log_p_ham:
    if log_p_spam - log_p_ham > t:
        return ("spam", [log_p_spam, log_p_ham])
    
    else:
        return ("ham", [log_p_spam, log_p_ham])

    #'data/testing/5037.2001-11-02.farmer.ham.txt'

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    

    performance_measures = np.zeros([2,2])


    ## TODO: Write your code here to modify the decision rule such that
    ## Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    errors = list()
    threshold = np.linspace(-257, 66, num=30)
    for t in threshold:
        performance_measures = np.zeros([2,2])
        for filename in (util.get_files_in_folder(test_folder)):
            label,log_posterior = alpha_controlled_classify_new_email(t, filename,
                                                    probabilities_by_category,
                                                    priors_by_category)
            
            # Measure performance (the filename indicates the true label)
            
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        #print(template % (correct[0],totals[0],correct[1],totals[1]))
        errors.append((totals[0] - correct[0], totals[1] - correct[1] ))

    plt.title("Q1(c)")
    plt.xlabel("Number of Type 1 Errors")
    plt.ylabel("Number of Type 2 Errors")
    plt.scatter(*zip(*errors)) 
    plt.show()




   

 