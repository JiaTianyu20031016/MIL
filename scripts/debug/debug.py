from MILdata.shepherd.dataset import (
    TokenizedDocumentDataset,
    create_mil_data_collator,
    load_dataset as load_mil_dataset,
)


samples = load_mil_dataset(hf_dataset="peiyi9979/Math-Shepherd", split="math")[:2000]
train_dataset = TokenizedDocumentDataset(samples, tokenizer=None)
