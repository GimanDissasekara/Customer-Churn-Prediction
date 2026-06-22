import axios from "axios";

const api = axios.create({
  baseURL: "/api",
});

export const getHealth = () => api.get("/health");

export const getModelMetadata = () => api.get("/model/metadata");

export const trainModel = (file) => {
  const form = new FormData();
  if (file) form.append("file", file);
  return api.post("/train", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300000,
  });
};

export const predictChurn = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/predict", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300000,
  });
};

export default api;
