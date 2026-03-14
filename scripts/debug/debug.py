from MILdata.ProcessBench.dataset import (
    TokenizedDocumentDataset,
    create_mil_data_collator,
    load_dataset as load_mil_dataset,
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/data2/Common_LLM_Base/Qwen/Qwen3-4B/")

samples = load_mil_dataset(split="math")[:2000]
train_dataset = TokenizedDocumentDataset(samples, tokenizer=tokenizer)
print(train_dataset[0])