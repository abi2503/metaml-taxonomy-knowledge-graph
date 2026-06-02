import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag_metaml.logging_setup import logger
from src.rag_metaml.config import settings

logger.info(f"MySQL_Host:{settings.mysql_host}")
logger.info(f"MySQL_Port:{settings.mysql_port}")
logger.info(f"MySQL_DB:{settings.mysql_db}")
logger.info(f"MySQL_User:{settings.mysql_user}")
logger.info(f"MySQL_Password:{settings.mysql_password}")
logger.info(f"Embed_Model:{settings.embed_model}")
logger.info(f"Chroma_Dir:{settings.chroma_dir}")
logger.info(f"Taxonomy_Collection:{settings.taxonomy_collection}")
logger.info(f"Domains_Table:{settings.domains_table}")
logger.info(f"Dimensions_Table:{settings.dimensions_table}")
logger.info(f"Areas_Table:{settings.areas_table}")
logger.info(f"Collections_Table:{settings.collections_table}")
logger.info("Smoke test completed successfully")









