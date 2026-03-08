from MILdata.PRM800K.dataset import load_dataset as load_prm800k_dataset  # pylint: disable=wrong-import-position
from MILdata.sentiment.dataset import load_dataset as load_sentiment_dataset  # pylint: disable=wrong-import-position

ds = load_prm800k_dataset("train", hf_dataset="MIL/MILdata/PRM800K/data/data_processed")

print(ds[0])