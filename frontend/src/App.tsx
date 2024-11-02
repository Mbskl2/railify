import './App.css'
import { useEffect } from 'react'
import { process } from './api'

function App() {
  const handleDrop = async (event: DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer?.files[0];
    console.log(file?.name);
    if (file && file.type === 'application/pdf') {
      await process(file);
    }
  };

  const handleDragOver = (event: DragEvent) => {
    console.log('Drag over');
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
    <>
      <div 
        style={{ border: '2px dashed #ccc', padding: '20px', textAlign: 'center' }}
      >
        Drop your PDF anywhere on the page
      </div>
    </>
  )
}

export default App
