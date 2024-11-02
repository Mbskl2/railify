import React, { useState, useEffect } from 'react';
import Draggable, { DraggableData, DraggableEvent } from 'react-draggable';
import { preprocess, process } from './api';
import './App.css';

const App = () => {
  const [pngUrl, setPngUrl] = useState<string | null>('path/to/your/image.png');
  const [svgUrl, setSvgUrl] = useState<string | null>(null);
  const [showPng, setShowPng] = useState<boolean>(true);
  const [showSvg, setShowSvg] = useState<boolean>(true);
  const [showBorderCreation, setShowBorderCreation] = useState<boolean>(false);
  const [showComparison, setShowComparison] = useState<boolean>(false);
  const [svgPosition, setSvgPosition] = useState({ x: 0, y: 0 });
  const [border, setBorder] = useState({ top: 0, left: 0, width: 0, height: 0 });
  const [isDrawing, setIsDrawing] = useState(false);

  const handleDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
      const pngUrl = await preprocess(file);
      setPngUrl(pngUrl);
      setShowBorderCreation(true);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleDrag = (e: DraggableEvent, data: DraggableData) => {
    setSvgPosition({ x: data.x, y: data.y });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setBorder({
      top: e.clientY - rect.top,
      left: e.clientX - rect.left,
      width: 0,
      height: 0,
    });
    setIsDrawing(true);
    e.preventDefault();
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing) return;
    const rect = e.currentTarget.getBoundingClientRect();
    setBorder((prevBorder) => ({
      ...prevBorder,
      width: e.clientX - rect.left - prevBorder.left,
      height: e.clientY - rect.top - prevBorder.top,
    }));
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    setIsDrawing(false);
    // Save the border data here
    console.log('Border data:', border);
    e.stopPropagation();
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
        {showComparison && <div style={{ marginTop: '20px', textAlign: 'center' }}>
          <button onClick={() => setShowPng(!showPng)}>
            Turn PNG layer {showPng ? 'off' : 'on'}
          </button>
          <button onClick={() => setShowSvg(!showSvg)} style={{ marginLeft: '10px' }}>
            Turn SVG layer {showSvg ? 'off' : 'on'}
          </button>
        </div>}
      </div>
      {showBorderCreation && (
        <div style={{ marginTop: '220px', textAlign: 'center' }}>
          {pngUrl && (
            <div
              style={{
                position: 'relative',
                display: 'inline-block',
              }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
            >
              <img
                src={pngUrl}
                alt="Processed PNG"
                style={{
                  display: 'block',
                  position: 'relative',
                }}
              />
              <div
                style={{
                  position: 'absolute',
                  top: border.top,
                  left: border.left,
                  width: border.width,
                  height: border.height,
                  border: '2px solid red',
                  pointerEvents: 'none',
                }}
              />
            </div>
          )}
        </div>
      )}
      {showComparison && (<div style={{ marginTop: '220px', textAlign: 'center' }}>
        {pngUrl && svgUrl && (
          <div style={{ position: 'relative', display: 'inline-block'}}>
            {showPng && <img src={pngUrl} alt="Processed PNG" style={{ display: 'block', position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }} />}
            {showSvg && (
              <Draggable position={svgPosition} onDrag={handleDrag}>
                <div>
                  <img src={svgUrl} alt="Processed SVG" style={{ display: 'block', position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', backgroundColor: 'transparent', filter: 'invert(29%) sepia(100%) saturate(7483%) hue-rotate(180deg) brightness(97%) contrast(104%)' }} />
                </div>
              </Draggable>
            )}
          </div>
        )}
      </div>)}
    </div>
  );
}

export default App;
