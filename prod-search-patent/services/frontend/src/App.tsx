// frontend/src/App.tsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import PatentInput from './components/PatentInput';
import BatchResults from './components/BatchResults';

function App() {
  return (
    <Router>
      <Routes>
        {/* 新しいルート - 複数特許入力がメイン */}
        <Route path="/" element={<PatentInput />} />
        <Route path="/results" element={<BatchResults />} />
      </Routes>
    </Router>
  );
}

export default App;