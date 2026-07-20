from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.models import (
    InspectionInferenceResult,
    InspectionInferenceResultFinding,
    InspectionPreprocessingResult,
    InspectionPreprocessingResultFinding,
    InspectionProcessingRun,
)

VERSION = 3
IDENTIFIER = "003_processing_results"
TABLES = (
    InspectionProcessingRun.__table__,
    InspectionPreprocessingResult.__table__,
    InspectionPreprocessingResultFinding.__table__,
    InspectionInferenceResult.__table__,
    InspectionInferenceResultFinding.__table__,
)
REQUIRED_TABLE_NAMES = frozenset(table.name for table in TABLES)


async def upgrade(connection: AsyncConnection) -> None:
    await connection.run_sync(
        lambda sync_connection: InspectionProcessingRun.metadata.create_all(
            sync_connection,
            tables=list(TABLES),
            checkfirst=True,
        )
    )
