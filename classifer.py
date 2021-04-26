import numpy as np 
from sklearn.decomposition import PCA
import csv
import pandas as pd 
import matplotlib.pyplot as plt

def load_data():
	"""
	Load embedding information and corresponding words from generated LSHWE embedding. 
	"""
	word_dict = {}
	words = [] 
	x = [] 
	with open("./generated_embs/lshwe_embedding.csv") as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			#print(row['vector'])
			remove_idx = [] 
			vector = row['vector'].split()
			new_v = [] 
			for i in range(len(vector)):
				vector[i] = vector[i].replace("[", '')
				vector[i] = vector[i].replace("\n", '')
				vector[i] = vector[i].replace("]", '')
				if len(vector[i]) > 0:
					vector[i] = float(vector[i])
					new_v.append(vector[i])

			word_dict[row['word']] = new_v
			x.append(new_v )
			words.append(row['word'])
	return x, words, word_dict

def dist_plot():
	"""
	This function calculates the euclidean dsitance between normal word and dirty word by using LSHWE embedding.
	The goal is to see if it can detect unseen dirty word by distance. 
	Second part will do a PCA plot on two classes: dirty words and normal words. 
	"""

	print("Calcuate near words...")
	x, words, word_dict = load_data()
	 
	f = open("./generated_embs/en_dirty_word.txt", "r")
	dirty_words = []
	for word in f.readlines():
		dirty_words.append(word.replace("\n", ""))
	other_word = []
	label_word = [] 
	targets = [] 
	for word in words:
		if word in dirty_words:
			label_word.append(word)
			targets.append(1)
		else:
			other_word.append(word)
			targets.append(0)
	
	fout = open("./results/near_words.txt", "w")
	for dw in label_word:
		fout.write(f"\ndirty word: {dw} \n\n")
		dist = {}
		for other in other_word:
			dist[other] = np.linalg.norm(np.array(word_dict[other]) 
								- np.array(word_dict[dw]))

		sort_d = sorted(dist.items(), key=lambda x:-x[1], reverse=True)[:10]
		fout.write("Top 10 near words: \n\n")
		for tt in sort_d: 
			fout.write("{0}: {1} \n".format(*tt))

	fout.close() 
	print("Near words done.")

	print("PCA plot...")
	subset_x = [] 
	subset_target = [] 
	for i, t in enumerate(targets):
		if t == 1: 
			subset_x.append(x[i])
			subset_target.append(t) 
		elif len(subset_x) < 500:
			subset_x.append(x[i])
			subset_target.append(t)


	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(subset_x)
	principalDf = pd.DataFrame(data = principalComponents
		, columns = ['principal component 1', 'principal component 2'])

	principalDf['target'] = subset_target
	
	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Principal Component 1', fontsize = 15)
	ax.set_ylabel('Principal Component 2', fontsize = 15)
	ax.set_title('2 component PCA', fontsize = 20)
	colors = ['r','b']
	labels = [1, 0]
	for target, color in zip(labels, colors):
		indicesToKeep = principalDf['target'] == target
		ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1'], 
				principalDf.loc[indicesToKeep, 'principal component 2'],
				c = color, s = 5)
	ax.legend(["dirty word", "normal"])
	ax.grid()
	plt.savefig("./results/PCA_vsi.png")
	print("PCA plot done.")
	plt.close()

def classification():
	"""
	This function does classification using SVM on both embedding vecors (LSHWE and W2V) 
	Print out accuracy and runtime for 5 iterations. 
	"""
	from sklearn import svm
	from sklearn.preprocessing import StandardScaler
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	import time


	print("SVM classification with LSHWE...")

	X = np.load('./generated_embs/lshwe_sum.npy')
	Y = np.load("./generated_embs/label.npy")
	acc_list = [] 
	fout = open("./results/classify_result.txt", "w")
	
	for i in range(5):
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
		
		start = time.time() 
		clf = svm.SVC(kernel='poly', degree=10)
		clf.fit(X_train, y_train)

		y_pred = clf.predict(X_test)
		#print("predict :")
		end = time.time() 
		acc = accuracy_score(y_test, y_pred)
		fout.write(f"accuracy: {acc}\n", )
		fout.write(f"run time: {end-start} \n")
		acc_list.append(acc)

	fout.write(f"\nlshwe avg accuracy: {np.mean(acc_list)}\n\n")

	print("SVM classification with LSHWE done.")
	
	print("SVM classification with W2V...")
	X = np.load('./generated_embs/w2v_sum.npy')
	acc_list = [] 
	for i in range(5):
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
		start = time.time() 
		clf = svm.SVC(kernel='poly', degree=10)
		clf.fit(X_train, y_train)

		y_pred = clf.predict(X_test)
		#print("predict :")
		end = time.time()  
		acc = accuracy_score(y_test, y_pred) 
		fout.write(f"accuracy: {acc} \n")
		fout.write(f"run time: {end-start} \n")

		acc_list.append(accuracy_score(y_test, y_pred))
	
	fout.write(f"\n w2v avg accuracy: {np.mean(acc_list)}\n\n")
	print("SVM classification with W2V done.")

dist_plot()
classification()



