from typing import Dict, List, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from .config import settings
from .logging_setup import logger


def get_engine() -> Engine:
    url=f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_db}"
    engine=create_engine(url,pool_pre_ping=True,pool_recycle=3600)
    logger.info(f"Connected to MySQL database: {settings.mysql_db}")
    return engine


def fetch_domains(engine: Engine)->List[Dict[str, Any]]:
    sql_query=text("""
    SELECT DOMAIN_ID as domain_id, DOMAIN_NAME as domain_name
    FROM domains
    """)
    with engine.connect() as conn:
        result=conn.execute(sql_query).mappings().all()
    logger.info(f"Fetched {len(result)} domains")
    return [dict(r) for r in result]


def fetch_dimensions(engine: Engine)->List[Dict[str,Any]]:
    """
    Fetch dimensions and the domain info they belong to.
    Returns: [{dimension_id, dimension_name, domain_id, domain_name}, ...]
    Robust to schemas where DIMENSIONS.DOMAIN_ID might be missing or named oddly.
    """
    sql = text("""
        SELECT
            d.DIMENSION_ID   AS dimension_id,
            d.DIMENSION_NAME AS dimension_name,
            d.DIMENSION_ATTRIBUTE AS dimension_attribute,
            m.DOMAIN_ID      AS domain_id
        FROM dimensions d
        LEFT JOIN domains m
          ON m.DOMAIN_ID = d.DOMAIN_ID
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()

    if rows:
        sample_keys = list(rows[0].keys())
        logger.info(f"Fetched {len(rows)} dimensions. Sample keys: {sample_keys}")
    else:
        logger.info("Fetched 0 dimensions.")

    return [dict(r) for r in rows]


def fetch_areas(engine: Engine) -> List[Dict[str, Any]]:
    """
    Fetch all areas with parent pointer + owning dimension.
    Returns: [{area_id, area_parent_id, area_depth_level, area_name, dimension_id}, ...]
    """
    sql = text("""
        SELECT AREA_ID          AS area_id,
               AREA_PARENT_ID   AS area_parent_id,
               AREA_DEPTH_LEVEL AS area_depth_level,
               AREA_NAME        AS area_name,
               DIMENSION_ID     AS dimension_id
        FROM areas
    """)
    with engine.connect() as conn:
        result = conn.execute(sql).mappings().all()
    logger.info(f"Fetched {len(result)} areas.")
    return [dict(r) for r in result]




