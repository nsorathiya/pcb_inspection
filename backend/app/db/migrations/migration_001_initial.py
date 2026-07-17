from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.models import (
    AuditEvent,
    Inspection,
    InspectionArtifact,
    ModelVersion,
    Recipe,
    SchemaVersion,
)

VERSION = 1
IDENTIFIER = "001_initial"
TABLES = (
    SchemaVersion.__table__,
    Inspection.__table__,
    InspectionArtifact.__table__,
    Recipe.__table__,
    ModelVersion.__table__,
    AuditEvent.__table__,
)
REQUIRED_TABLE_NAMES = frozenset(table.name for table in TABLES)


async def upgrade(connection: AsyncConnection) -> None:
    await connection.run_sync(
        lambda sync_connection: SchemaVersion.metadata.create_all(
            sync_connection,
            tables=list(TABLES),
            checkfirst=True,
        )
    )
