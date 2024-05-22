from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Define the model name or path to the checkpoint
model_name_or_path = "out/cls-models/bert-large-uncased-jpwahle/etpc-paraphrase-detection/checkpoint-3045"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

# Now you can use the tokenizer and model for inference or further training
print("loaded model succesfully")

