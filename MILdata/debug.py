from transformers import AutoTokenizer
from data.loader import TokenizedDocumentDataset, load_dataset, create_mil_data_collator
from transformers import Trainer
from model.simple_mil_model import SimpleMILModel

samples = load_dataset("imdb_edus")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = TokenizedDocumentDataset(samples, tokenizer)
collator = create_mil_data_collator(tokenizer)
samples = load_dataset("imdb_sent")
model = SimpleMILModel(backbone_name="bert-base-uncased", decision_threshold=0.5)

trainer = Trainer(model=model, train_dataset=dataset, data_collator=collator)
