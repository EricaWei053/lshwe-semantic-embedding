# lshwe-semantic-embedding
This is a project for the course COMS 4995 Semantic Representation for NLP at Columbia University, Spring 2021. 

Teammate: Erica Wei, Chuhui Chen, Gary Liu, Samuel Weissmann. 


## Project description 


## Data 



## Code 
We mainly used colab(or jupyter notebook) semantic.ipynb to genreate LSHWE embeddings from data we used. The expected runtime is around 1-2 hours.

After getting genreated lshwe adn w2v embedding vecotrs, we postprocessed them to be average or sum vector for a whole sentence. 

classifier.py: this file does analysis for two representatin embeddings we compared, including distance caclulation, PCA visuallization and classification. 

## How to run: 
You need to run colab first to get lshwe and w2v embeddings. Put those embedding files into ./genreated_embeds to run next step. 
Then run 
```
python classifier.py
```

Note: You can directly check out ./resuls folder if you don't want to run it by yourself. 


## Future work: 
1. Adding subword technique on LSHWE model.
2. Try multiple classifers to compare results. 
