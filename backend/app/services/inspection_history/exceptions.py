class InspectionHistoryError(Exception):
    """Base class for safe inspection-history failures."""


class HistoryFilterError(InspectionHistoryError):
    pass


class HistoryCursorError(InspectionHistoryError):
    pass


class HistoryCursorVersionError(HistoryCursorError):
    pass


class HistoryCursorFilterMismatchError(HistoryCursorError):
    pass


class HistoryConsistencyError(InspectionHistoryError):
    pass


class HistoryRetrievalError(InspectionHistoryError):
    pass
