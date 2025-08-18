# CSE151A
ML_project

## Dataset
Dataset is stored on Google Drive:  (Since data is bigger than 100MB so I have to do this)

https://drive.google.com/drive/folders/1-FgLo4-60RSu90v8pcpO4wi5B3rW-qAM?dmr=1&ec=wgc-drive-globalnav-goto

Here is the oringal source :
https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam?resource=download&select=recommendations.csv

## Environment (Google Colab)

1. Open the notebook in Google Colab.
2. Mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')



For questions 6 and 4, I first checked for duplicate records and missing values. Since the data was clean, I then manually inspected the features to identify any irrelevant ones, such as steam_deck, which I decided to drop.
Finally, I normalized some features to ensure they are on a common scale. I also dropped features such as review_id, since it is randomly generated and does not carry any meaningful information for the analysis.
In this case, I applied a log transform because some of the numeric features were highly skewed.


You can run the project notebook directly in Google Colab here:  
[Open Notebook in Colab](https://colab.research.google.com/drive/1oD1BPJjESRtBzKWiDDiOkHiTpC-dRAig?authuser=1#scrollTo=0G3Zu6dNsOuR)
