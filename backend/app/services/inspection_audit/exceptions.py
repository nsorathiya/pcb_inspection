class AuditError(Exception):
    pass


class AuditCursorError(AuditError):
    pass


class AuditCursorVersionError(AuditCursorError):
    pass


class AuditCursorInspectionMismatchError(AuditCursorError):
    pass


class AuditProjectionError(AuditError):
    pass


class AuditRetrievalError(AuditError):
    pass
