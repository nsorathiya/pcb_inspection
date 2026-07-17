from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.models import InspectionValidation, InspectionValidationFinding

VERSION = 2
IDENTIFIER = "002_validation_results"
TABLES = (
    InspectionValidation.__table__,
    InspectionValidationFinding.__table__,
)
REQUIRED_TABLE_NAMES = frozenset(table.name for table in TABLES)


async def upgrade(connection: AsyncConnection) -> None:
    await connection.run_sync(
        lambda sync_connection: InspectionValidation.metadata.create_all(
            sync_connection,
            tables=list(TABLES),
            checkfirst=True,
        )
    )
