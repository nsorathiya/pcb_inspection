import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { HistoryPage } from './pages/HistoryPage'
import { InspectionDetailPage } from './pages/InspectionDetailPage'
import { NewInspectionPage } from './pages/NewInspectionPage'
import { NotFoundPage } from './pages/NotFoundPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<HistoryPage />} />
          <Route path="inspections/new" element={<NewInspectionPage />} />
          <Route path="inspections/:inspectionId" element={<InspectionDetailPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
