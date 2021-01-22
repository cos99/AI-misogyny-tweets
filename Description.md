## Short description
   This project is for educational purpose. The main goal of the project is to determine whether a tweet is misogynistic or not. For this to be achieved, I used machine learnig 
algorithms from the *sklearn* library, trained my model and then checked the accuracy of my model, before testing it with the test samples.

## How does it work?
   Firstly, we need to bring the tweets in computer-readable form. To do that, we will use the *__bag of words__* model:
   - clean the text (eliminate symbols: * , @ # etc), normalize the text (all words should be with lower letters), keep important words (stemmatize and lemmatize if necessary).
   - make a dictionary with all the words that are present in all the tweets.
   - create the bag of words by linking the most important __N__ words (where N can be any positive integer, it depends from case to case) with the number of times 
they appear in the respective tweet.
   A bag of words example: 
           @  che  .   ,   di  e
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2

   Basically, the bag of words moddel will be a matrix with __N__ columns and __n__ rows (where n is the number of tweets in the testing/training file).
   
