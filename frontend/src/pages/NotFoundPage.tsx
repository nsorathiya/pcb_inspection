import { Link } from 'react-router-dom'

export function NotFoundPage() {
  return <section className="not-found"><p className="eyebrow">404</p><h2>Operator route not found</h2><p>This console provides inspection history, paired intake, and inspection workflow routes only.</p><Link className="button secondary" to="/">Return to inspection history</Link></section>
}
