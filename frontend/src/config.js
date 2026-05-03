// API base URL — configured via environment variable for deployment flexibility.
// In Docker: nginx proxies /api/* to the backend, so we use "" (relative URLs).
// In local dev: defaults to http://127.0.0.1:8000 (Django dev server).
const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default API_BASE;
