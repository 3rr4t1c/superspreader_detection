# Retweet Network on vaccinitaly dataset

To build a retweet network three steps are followed:
1. `Network-data-preprocessing.ipynb` -> Read the raw data and select features, filter rows etc.
2. `Network-edgelist-builder.ipynb` -> Build a list of edges from preprocessed tweets 
3. `Network-component-extraction.ipynb` -> Extract the giant component from list of edges

The extracted component is in the file:
* `edgelist_component.csv`

Other intermediate files are not included due to the big size.