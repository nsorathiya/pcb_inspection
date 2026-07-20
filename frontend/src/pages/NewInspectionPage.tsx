import { type FormEvent, useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { createInspection } from '../api/inspections'
import { getRecipes } from '../api/recipes'
import { toApiClientError, type ApiClientError } from '../api/errors'
import type { RecipeCatalogueItem, RecipeCatalogueResponse, RecipeFilters } from '../api/types'
import { ErrorPanel } from '../components/ErrorPanel'
import { FileSelection } from '../components/FileSelection'
import { StatusBadge } from '../components/StatusBadge'

interface FormErrors {
  boardId?: string
  recipe?: string
  rgbImage?: string
  heightMap?: string
}

const EMPTY_RECIPE_FILTERS: RecipeFilters = {
  recipe_id: '',
  recipe_version: '',
  name: '',
  status: '',
}

export function NewInspectionPage() {
  const navigate = useNavigate()
  const [recipes, setRecipes] = useState<RecipeCatalogueResponse | null>(null)
  const [recipeFilters, setRecipeFilters] = useState<RecipeFilters>(EMPTY_RECIPE_FILTERS)
  const [appliedRecipeFilters, setAppliedRecipeFilters] = useState<RecipeFilters>(EMPTY_RECIPE_FILTERS)
  const [recipeCursors, setRecipeCursors] = useState<Array<string | undefined>>([undefined])
  const [selectedRecipe, setSelectedRecipe] = useState<RecipeCatalogueItem | null>(null)
  const [boardId, setBoardId] = useState('')
  const [lotId, setLotId] = useState('')
  const [operatorId, setOperatorId] = useState('')
  const [stationId, setStationId] = useState('')
  const [rgbImage, setRgbImage] = useState<File | null>(null)
  const [heightMap, setHeightMap] = useState<File | null>(null)
  const [errors, setErrors] = useState<FormErrors>({})
  const [apiError, setApiError] = useState<ApiClientError | null>(null)
  const [recipeError, setRecipeError] = useState<ApiClientError | null>(null)
  const [loadingRecipes, setLoadingRecipes] = useState(true)
  const [submitting, setSubmitting] = useState(false)

  const currentRecipeCursor = recipeCursors.at(-1)
  const loadRecipes = useCallback(async (signal?: AbortSignal) => {
    setLoadingRecipes(true)
    setRecipeError(null)
    try {
      const response = await getRecipes(appliedRecipeFilters, currentRecipeCursor, 10, signal)
      setRecipes(response.data)
    } catch (caught) {
      const mapped = toApiClientError(caught)
      if (mapped.code !== 'REQUEST_ABORTED') setRecipeError(mapped)
    } finally {
      setLoadingRecipes(false)
    }
  }, [appliedRecipeFilters, currentRecipeCursor])

  useEffect(() => {
    const controller = new AbortController()
    void loadRecipes(controller.signal)
    return () => controller.abort()
  }, [loadRecipes])

  const validate = (): FormErrors => {
    const next: FormErrors = {}
    if (!boardId.trim()) next.boardId = 'Board ID is required.'
    if (!selectedRecipe) next.recipe = 'Select one recipe identity and version.'
    if (!rgbImage) next.rgbImage = 'Select an RGB image.'
    if (!heightMap) next.heightMap = 'Select a height/depth file.'
    return next
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (submitting) return
    const nextErrors = validate()
    setErrors(nextErrors)
    setApiError(null)
    if (Object.keys(nextErrors).length || !selectedRecipe || !rgbImage || !heightMap) return

    setSubmitting(true)
    try {
      const response = await createInspection({
        boardId,
        recipe: selectedRecipe,
        rgbImage,
        heightMap,
        lotId,
        operatorId,
        stationId,
      })
      navigate(`/inspections/${response.data.inspection_id}`, {
        state: {
          intakeCreated: true,
          status: response.data.status,
          inspectionId: response.data.inspection_id,
        },
      })
    } catch (caught) {
      setApiError(toApiClientError(caught))
    } finally {
      setSubmitting(false)
    }
  }

  const applyRecipeFilters = (event: FormEvent) => {
    event.preventDefault()
    setRecipeCursors([undefined])
    setAppliedRecipeFilters(recipeFilters)
  }

  return (
    <section aria-labelledby="new-inspection-title">
      <div className="page-heading">
        <div><p className="eyebrow">Paired file intake</p><h2 id="new-inspection-title">New Inspection</h2><p>Register one RGB image and one native height/depth file for the same physical inspection.</p></div>
      </div>

      {apiError && <ErrorPanel error={apiError} title="Inspection was not created" />}
      {Object.keys(errors).length > 0 && (
        <div className="form-error-summary" role="alert" aria-labelledby="form-errors-title">
          <h3 id="form-errors-title">Complete the required fields</h3>
          <ul>{errors.boardId && <li><a href="#board-id">{errors.boardId}</a></li>}{errors.recipe && <li><a href="#recipe-selection">{errors.recipe}</a></li>}{errors.rgbImage && <li><a href="#rgb_image">{errors.rgbImage}</a></li>}{errors.heightMap && <li><a href="#height_map">{errors.heightMap}</a></li>}</ul>
        </div>
      )}

      <form className="inspection-form" onSubmit={submit} noValidate>
        <section className="panel" aria-labelledby="identity-heading">
          <div className="panel-heading"><div><p className="step-number">01</p><h3 id="identity-heading">Inspection identity</h3></div><p>Required identifiers are normalized by the backend.</p></div>
          <div className="form-grid">
            <label htmlFor="board-id">Board ID <span aria-hidden="true">*</span><input id="board-id" value={boardId} onChange={(event) => setBoardId(event.target.value)} aria-invalid={Boolean(errors.boardId)} aria-describedby={errors.boardId ? 'board-id-error' : undefined} /></label>
            {errors.boardId && <p className="field-error" id="board-id-error">{errors.boardId}</p>}
            <label htmlFor="lot-id">Lot ID <span className="optional-label">Optional</span><input id="lot-id" value={lotId} onChange={(event) => setLotId(event.target.value)} /></label>
            <label htmlFor="operator-id">Operator ID <span className="optional-label">Optional</span><input id="operator-id" value={operatorId} onChange={(event) => setOperatorId(event.target.value)} /></label>
            <label htmlFor="station-id">Station ID <span className="optional-label">Optional · audit only</span><input id="station-id" value={stationId} onChange={(event) => setStationId(event.target.value)} /></label>
          </div>
        </section>

        <section className="panel" id="recipe-selection" aria-labelledby="recipe-heading">
          <div className="panel-heading"><div><p className="step-number">02</p><h3 id="recipe-heading">Recipe selection</h3></div><p>Choose an exact catalogue identity. ACTIVE does not mean production approved.</p></div>
          <div className="recipe-filter-block">
            <div className="compact-filter-grid">
              <label>Recipe ID<input value={recipeFilters.recipe_id} onChange={(e) => setRecipeFilters((value) => ({ ...value, recipe_id: e.target.value }))} /></label>
              <label>Version<input value={recipeFilters.recipe_version} onChange={(e) => setRecipeFilters((value) => ({ ...value, recipe_version: e.target.value }))} /></label>
              <label>Name<input value={recipeFilters.name} onChange={(e) => setRecipeFilters((value) => ({ ...value, name: e.target.value }))} /></label>
              <label>Status<select value={recipeFilters.status} onChange={(e) => setRecipeFilters((value) => ({ ...value, status: e.target.value as RecipeFilters['status'] }))}><option value="">Any status</option><option>DRAFT</option><option>ACTIVE</option><option>RETIRED</option></select></label>
              <button className="button secondary" type="button" onClick={applyRecipeFilters} disabled={loadingRecipes}>Filter recipes</button>
            </div>
          </div>
          <div aria-live="polite">
            {loadingRecipes && <p className="loading-state">Loading recipe catalogue…</p>}
            {recipeError && <ErrorPanel error={recipeError} onRetry={() => void loadRecipes()} title="Recipe catalogue unavailable" />}
          </div>
          {!loadingRecipes && !recipeError && recipes?.items.length === 0 && <div className="empty-state compact"><h4>No recipes available</h4><p>The read-only catalogue returned no matching recipe identities.</p></div>}
          {!loadingRecipes && !recipeError && recipes && recipes.items.length > 0 && (
            <fieldset className={`recipe-list ${errors.recipe ? 'field-invalid' : ''}`}>
              <legend>Select one recipe and version</legend>
              {recipes.items.map((recipe) => {
                const key = `${recipe.recipe_id}\u0000${recipe.recipe_version}`
                const checked = selectedRecipe?.recipe_id === recipe.recipe_id && selectedRecipe.recipe_version === recipe.recipe_version
                return <label className={`recipe-option ${checked ? 'selected' : ''}`} key={key}><input type="radio" name="recipe" checked={checked} onChange={() => setSelectedRecipe(recipe)} /><span><strong>{recipe.name}</strong><span className="mono">{recipe.recipe_id} · {recipe.recipe_version}</span></span><StatusBadge value={recipe.status} /></label>
              })}
              <div className="pagination compact"><button type="button" className="text-button" disabled={recipeCursors.length === 1} onClick={() => setRecipeCursors((value) => value.slice(0, -1))}>Previous recipes</button><span>Catalogue page {recipeCursors.length}</span><button type="button" className="text-button" disabled={!recipes.page.has_more || !recipes.page.next_cursor} onClick={() => setRecipeCursors((value) => [...value, recipes.page.next_cursor ?? undefined])}>Next recipes</button></div>
            </fieldset>
          )}
          {errors.recipe && <p className="field-error">{errors.recipe}</p>}
        </section>

        <section className="panel" aria-labelledby="files-heading">
          <div className="panel-heading"><div><p className="step-number">03</p><h3 id="files-heading">Paired source files</h3></div><p>Selection guidance only; backend technical validation remains authoritative.</p></div>
          <div className="file-grid">
            <FileSelection id="rgb_image" label="RGB image" hint="Accepted intake extensions include PNG, JPEG, BMP, and TIFF." accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,image/png,image/jpeg,image/bmp,image/tiff" file={rgbImage} onChange={setRgbImage} error={errors.rgbImage} />
            <FileSelection id="height_map" label="Height / depth map" hint="Accepted intake extensions include PNG, TIFF, and NPY. Native depth is preserved." accept=".png,.tif,.tiff,.npy,image/png,image/tiff,application/octet-stream" file={heightMap} onChange={setHeightMap} error={errors.heightMap} />
          </div>
        </section>

        <div className="submit-bar">
          <div><strong>Intake only</strong><span>Validation will run only when you choose it on the detail page.</span></div>
          <button className="button primary" type="submit" disabled={submitting}>{submitting ? 'Creating inspection…' : 'Create inspection'}</button>
        </div>
        <p className="sr-only" aria-live="polite">{submitting ? 'Inspection upload is in progress.' : ''}</p>
      </form>
    </section>
  )
}
