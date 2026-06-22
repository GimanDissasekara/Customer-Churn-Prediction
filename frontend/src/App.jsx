import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Train from "./pages/Train";
import Predict from "./pages/Predict";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/train" element={<Train />} />
          <Route path="/predict" element={<Predict />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
