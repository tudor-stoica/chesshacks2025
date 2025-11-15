from huggingface_hub import HfApi
import os
import dotenv

dotenv.load_dotenv(dotenv_path=".env.local")

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/model",
    repo_id="wilsonfwang/ChessHacks",
    repo_type="model",
)
