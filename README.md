# lshwe-semantic-embedding
This is a project for the course COMS 4995 Semantic Representation for NLP at Columbia University, Spring 2021. 
Prof: Daniel Bauer
Teammate: Erica Wei, Chuhui Chen, Gary Liu, Samuel Weissmann. 


## Project description 
As the proliferation of online hate speech continues to grow, many people turn to natural language processing to help curb it’s spread. However, online communities that routinely engage in hate speech often employ simple word substitutions with fictitious, rare, or out-of-context words to avoid detection. Our project aims to solve this issue by using LSHWE. LSHWE uses a Nearest Neighbor (NN) Search to identify words that share high amounts of contextual similarities, while an autoencoder helps learn representations for rare or obfuscated words that share contexts with known words. We used LSHWE embedding and word2vec embedding to compare and analysis how much improvement LSHWE can make on this issue. Details in ./results 

## Data 

We used dataset from: ETHOS: an Online Hate Speech Detection Dataset, which is a textual dataset with two variants: binary and multi-label, called 'ETHOS', based on YouTube and Reddit comments validated through figure-eight crowdsourcing platform. (https://arxiv.org/abs/2006.08328)


## Code 
We mainly useds ```semantic_embeds.ipynb``` on Colab(via GPU to speed up) to genreate LSHWE embeddings from data we used. The expected runtime is around 1-2 hours. We directly used LSHWE model from paper: 

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


## Reference 
```
Ioannis Mollas and Zoe Chrysopoulou and Stamatis Karlos and Grigorios Tsoumakas, "ETHOS: an Online Hate Speech Detection Dataset", 2020, arXiv:2006.08328. 

Z. Zhao, M. Gao, F. Luo, Y. Zhang and Q. Xiong, "LSHWE: Improving Similarity-Based Word Embedding with Locality Sensitive Hashing for Cyberbullying Detection," 2020 International Joint Conference on Neural Networks (IJCNN), 2020, pp. 1-8, doi: 10.1109/IJCNN48605.2020.9207640.
```
