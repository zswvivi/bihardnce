from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese')

model_save_path = 'models/bert'

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
