import React, { useState, useEffect } from 'react';
import { process } from './api';
import './App.css';

function App() {
  const [pngUrl, setPngUrl] = useState<string | null>(null);
  const [svgUrl, setSvgUrl] = useState<string | null>(null);
  const [showPng, setShowPng] = useState<boolean>(true);
  const [showSvg, setShowSvg] = useState<boolean>(true);

  const handleDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
      const { pngUrl, svgUrl } = await process(file);
      setPngUrl(pngUrl);
      setSvgUrl(svgUrl);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  useEffect(() => {
    document.addEventListener('drop', handleDrop);
    document.addEventListener('dragover', handleDragOver);

    return () => {
      document.removeEventListener('drop', handleDrop);
      document.removeEventListener('dragover', handleDragOver);
    };
  }, []);

  return (
    <div>
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '150px', backgroundColor: '#f9f9f9', zIndex: 1 }}>
        <div 
          onDrop={handleDrop} 
          onDragOver={handleDragOver} 
          style={{ border: '2px dashed #ccc', padding: '20px', textAlign: 'center' }}
        >
          Drop your PDF anywhere
        </div>
        <div style={{ marginTop: '20px', textAlign: 'center' }}>
          <button onClick={() => setShowPng(!showPng)}>
            Turn PNG layer {showPng ? 'off' : 'on'}
          </button>
          <button onClick={() => setShowSvg(!showSvg)} style={{ marginLeft: '10px' }}>
            Turn SVG layer {showSvg ? 'off' : 'on'}
          </button>
        </div>
      </div>
      <div style={{ marginTop: '220px', textAlign: 'center' }}>
        {pngUrl && svgUrl && (
          <div style={{ position: 'relative', display: 'inline-block' }}>
            {showPng && <img src={pngUrl} alt="Processed PNG" style={{ display: 'block', position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }} />}
            {showSvg && <img src={svgUrl} alt="Processed SVG" style={{ display: 'block', position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', backgroundColor: 'transparent' }} />}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
