from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import HfApi, HfFolder, Repository
from dotenv import load_dotenv
import os


load_dotenv()
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# Model and tokenizer paths
MODEL_DIR = './trained_model'
REPO_NAME = 'poudelsanjayp1/datamining_project'
REPO_URL = f"https://huggingface.co/{REPO_NAME}"

# Login to Hugging Face
HfFolder.save_token(HUGGINGFACE_TOKEN)
api = HfApi()
api.upload_folder(
    folder_path=MODEL_DIR,
    repo_id=REPO_NAME,
    repo_type="model",

)
print(f"Model uploaded to {REPO_URL}")
