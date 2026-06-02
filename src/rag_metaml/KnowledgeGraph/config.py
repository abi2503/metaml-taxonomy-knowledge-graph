from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load variables from .env into process env
load_dotenv()

@dataclass
class Settings:
    # MySQL
    mysql_host: str = os.getenv("MYSQL_HOSTNAME", "127.0.0.1")
    mysql_port: int = int(os.getenv("MYSQL_PORT", 3306))
    mysql_db: str = os.getenv("MYSQL_DB_NAME", "Meta_ML")
    mysql_user: str = os.getenv("MYSQL_USER", "")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")

    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_dir: str = os.getenv("CHROMA_DIR", ".chroma")
    taxonomy_collection: str = os.getenv("TAXONOMY_COLLECTION", "taxonomy_nodes")

  
    domains_table: str = os.getenv("DOMAINS_TABLE", "domains")
    dimensions_table: str = os.getenv("DIMENSIONS_TABLE", "dimensions")
    areas_table: str = os.getenv("AREAS_TABLE", "areas")
    collections_table: str = os.getenv("COLLECTIONS_TABLE", "cm_collections")

settings = Settings()
