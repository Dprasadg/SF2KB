import axios from "axios";

const BASE_URL =
  import.meta.env.VITE_API_URL ||
  import.meta.env.REACT_APP_API_URL ||
  "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 300000,
});

export default api;
