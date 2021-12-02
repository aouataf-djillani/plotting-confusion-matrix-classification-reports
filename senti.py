import nltk
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,ConfusionMatrixDisplay
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia= SentimentIntensityAnalyzer()

df= pd.read_csv("amazonreviews.tsv", sep= '\t')
df.dropna(inplace=True)
spaces=[]
for index, label, review in df.itertuples():
	if type(review)==str:
		if review.isspace():
			spaces.append(index)

df.drop(spaces, inplace=True)
df['scores']= df['review'].apply(lambda review: sia.polarity_scores(review))
df['compound']= df['scores'].apply(lambda d:d['compound'])
df['polarity']= df['compound'].apply(lambda compound: 'pos' if compound>=0 else 'neg')
print(df.head())

accuracy_score(df['label'], df['polarity'])
cm=confusion_matrix(df['label'], df['polarity'])
#simple plot using confusionMatrixDisplay
cmd_obj = ConfusionMatrixDisplay(cm, display_labels=['neg', 'pos'])
cmd_obj.plot()

#using Seaborn and Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Purples');  
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['neg', 'pos']); ax.yaxis.set_ticklabels(['neg', 'pos']);
#Visualizing Classification Report 
report = classification_report(df['label'], df['polarity'], output_dict=True)
df_report = pd.DataFrame(report).transpose().round(2)
df_report.style.background_gradient(cmap='Purples').set_precision(2)