# Superspreader Detection - Welcome
Investigating superspreader detection methods
---
This repo contains code and experiments about the results presented in [this paper](https://arxiv.org/abs/2207.09524).
For this experiments a twitter dataset has been used about the spreading of COVID-19 in italy (Vaccinitaly).
To label the tweets creating a ground truth, [newsguard](https://www.newsguardtech.com/) scores has been exploited.

- `fibi.py` code implementing the FIB-index evaluation (h-index variant);
- `False_Information_Broadcaster_index_Analysis_PANDAS.ipynb` Example of evaluated FIB-indexes;
- `Splitter.ipynb` The procedure used to split and select features from the main dataset.
- `newsguard_scores.csv` a subset of scores retrieved from Newsguard
