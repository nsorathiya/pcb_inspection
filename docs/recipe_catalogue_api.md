# Read-Only Recipe Catalogue API

Date: 2026-07-20

## Route and purpose

`GET /api/v1/recipes` returns persisted recipe identities and display metadata
for future operator inspection-intake selection. A client can display `name`,
retain `status` as lifecycle context, and submit the returned `recipe_id` and
`recipe_version` unchanged to `POST /api/v1/inspections`.

The route is read-only. It does not create, edit, delete, activate, retire,
clone, import, or export recipes. It does not append audit events, read files,
or execute validation, preprocessing, or inference.

## Actual schema-version 3 fields

The persisted `Recipe` row contains:

| Field | Catalogue behavior |
| --- | --- |
| Internal UUID `id` | Used only as the deterministic cursor tie-breaker; not returned. |
| `recipe_id` | Returned unchanged; part of the unique intake identity. |
| `recipe_version` | Returned unchanged; part of the unique intake identity. |
| `name` | Returned as persisted display metadata. |
| `configuration_json` | Deliberately neither selected nor returned. |
| `status` | Returned as `DRAFT`, `ACTIVE`, or `RETIRED`. |
| `created_at` | Returned as UTC and used as the primary ordering field. |
| `updated_at` | Returned as UTC display metadata; not used to reorder catalogue history. |

`(recipe_id, recipe_version)` is unique. Multiple versions of the same logical
recipe remain separate selectable items. Version strings are opaque identities:
the API does not parse semantic versions, group versions, identify a “latest”
textual version, or invent a current/preferred version.

## Schema limitations and safety meaning

Schema version 3 has no recipe description, station, board revision,
calibration identity, approval record, production-readiness flag, model link,
preferred-version field, or dedicated recipe-configuration contract. The
`status` enum is returned literally; `ACTIVE` is not interpreted as proof of
model compatibility, calibration validity, production approval, or suitability
for a particular board.

Inspections store `recipe_id` and `recipe_version` as intake metadata but have no
foreign key to `recipes`. Existing intake validates and stores the supplied text
identity; it does not resolve the catalogue, check recipe status, or load
`configuration_json`. Catalogue selection therefore improves identity accuracy
for clients but is not an authorization or compatibility gate.

Persisted catalogue identities must already be trimmed and acceptable to the
existing intake normalization contract. Unsafe or noncanonical stored display
metadata fails the catalogue request safely instead of being silently changed.

## Response

```json
{
  "items": [
    {
      "recipe_id": "PCB-A",
      "recipe_version": "2.0",
      "name": "PCB A operator recipe",
      "status": "ACTIVE",
      "created_at": "2026-07-20T10:00:00Z",
      "updated_at": "2026-07-20T10:15:00Z"
    }
  ],
  "page": {
    "limit": 25,
    "has_more": false,
    "next_cursor": null
  },
  "applied_filters": {},
  "request_id": "recipe-catalogue-request-123"
}
```

The response omits configuration JSON, internal row IDs, paths, model data,
secrets, audit internals, SQL, and any total count.

## Ordering and cursor pagination

Rows use stable newest-first ordering:

1. `recipes.created_at DESC`
2. internal canonical recipe UUID `DESC`

`limit` defaults to 25 and accepts 1 through 100. The query fetches `limit + 1`
projected rows to determine `has_more`; no `total_count` is calculated.

`next_cursor` is canonical compact JSON encoded with URL-safe base64. It
contains cursor contract `pcb-aoi-recipe-catalogue-cursor/1.0`, the last UTC
`created_at`, the internal canonical row UUID, and a SHA-256 digest of normalized
filters. It contains no SQL offset, path, or secret and is not an authentication
token. Malformed, incomplete, noncanonical, unsupported, or filter-mismatched
cursors return HTTP 400. Changing only `limit` remains valid.

## Filters

The following exact-match filters are supported and combined with AND:

- `recipe_id`
- `recipe_version`
- `name`
- `status` (`DRAFT`, `ACTIVE`, or `RETIRED`)

Identity filters accept at most 128 characters and `name` accepts at most 256.
Blank, control-character, and overlong values are rejected. There is no
wildcard or full-text search. Unknown exact identities return a successful
empty page. `applied_filters` contains the normalized safe filter values.

## Query and read-only guarantees

Every page, including an empty one, uses one bounded column-projected SELECT.
The query does not load `configuration_json` and performs no inspection,
artifact, validation, processing, model, or audit lookup. It executes no
INSERT, UPDATE, or DELETE and changes no runtime file.

## Request IDs and errors

The current request ID is returned both in `X-Request-ID` and the response body.
An incoming ID is preserved; otherwise existing middleware generates a UUID.
Catalogue viewing creates no audit event and exposes no historical request ID.

Errors use `{code, message, request_id}`:

- HTTP 400: invalid filter, malformed/unsupported cursor, or cursor/filter mismatch
- HTTP 422: invalid query shape such as `limit=0` or `limit=101`
- HTTP 500: safe retrieval or persisted-metadata consistency failure

Responses never include SQL, database paths, tracebacks, or exception text.

## PowerShell examples

```powershell
$headers = @{ "X-Request-ID" = "recipe-catalogue-request-123" }
$page = Invoke-RestMethod `
  -Headers $headers `
  -Uri "http://127.0.0.1:8000/api/v1/recipes?limit=25&status=ACTIVE"

$cursor = [System.Uri]::EscapeDataString($page.page.next_cursor)
Invoke-RestMethod `
  -Headers $headers `
  -Uri "http://127.0.0.1:8000/api/v1/recipes?limit=50&status=ACTIVE&cursor=$cursor"
```

Example selection for intake:

```powershell
$selected = $page.items[0]
$form = @{
  board_id = "PCB-A"
  recipe_id = $selected.recipe_id
  recipe_version = $selected.recipe_version
  lot_id = "LOT-1"
  operator_id = "operator-1"
  station_id = "station-1"
  rgb_image = Get-Item "C:\intake\rgb.png"
  height_map = Get-Item "C:\intake\height.npy"
}
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/inspections" `
  -Form $form
```

Listing or selecting a recipe does not prove model compatibility, calibration,
production approval, or PCB acceptance.

## Focused tests

```powershell
.\.venv\Scripts\python.exe -m pytest .\backend\tests\test_recipe_catalogue_api.py -q
```
