class DevelopmentReportError(Exception):
    pass


class DevelopmentReportRetrievalError(DevelopmentReportError):
    pass


class DevelopmentReportConsistencyError(DevelopmentReportError):
    pass


class DevelopmentReportCanonicalError(DevelopmentReportError):
    pass
