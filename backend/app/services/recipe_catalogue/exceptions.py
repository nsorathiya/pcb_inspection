class RecipeCatalogueError(Exception):
    """Base class for safe recipe-catalogue failures."""


class RecipeCatalogueFilterError(RecipeCatalogueError):
    pass


class RecipeCatalogueCursorError(RecipeCatalogueError):
    pass


class RecipeCatalogueCursorVersionError(RecipeCatalogueCursorError):
    pass


class RecipeCatalogueCursorFilterMismatchError(RecipeCatalogueCursorError):
    pass


class RecipeCatalogueConsistencyError(RecipeCatalogueError):
    pass


class RecipeCatalogueRetrievalError(RecipeCatalogueError):
    pass
