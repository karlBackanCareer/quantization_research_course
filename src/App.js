import React, { useState, useEffect } from 'react';
import './App.css';

const loadImage = src => new Promise((resolve, reject) => {
  const img = new Image();
  img.onload = () => resolve(img);
  img.onerror = reject;
  img.src = src;
});

const getFileSize = async path => {
  const response = await fetch(path);
  const blob = await response.blob();
  return blob.size;
};

const getImageType = (imageName) => {
  const types = {
    'ppt3': 'text', 'comic': 'synthetic', 'baboon': 'natural_texture',
    'lenna': 'portrait', 'bridge': 'natural_scene', 'barbara': 'portrait',
    'coastguard': 'natural_scene', 'face': 'portrait', 'flowers': 'natural_texture',
    'foreman': 'portrait', 'man': 'portrait', 'monarch': 'natural_texture',
    'pepper': 'natural_texture', 'zebra': 'natural_texture'
  };
  return types[imageName] || 'unknown';
};

const getContentType = (imageName) => {
  const content = {
    'ppt3': 'text_heavy', 'comic': 'line_art', 'baboon': 'high_frequency',
    'lenna': 'mixed_frequency', 'bridge': 'structured', 'barbara': 'texture_detail',
    'coastguard': 'motion_blur', 'face': 'smooth_gradient', 'flowers': 'natural_pattern',
    'foreman': 'facial_features', 'man': 'portrait_detail', 'monarch': 'fine_detail',
    'pepper': 'smooth_surface', 'zebra': 'repeated_pattern'
  };
  return content[imageName] || 'unknown';
};

const App = () => {
  const scales = [0.85, 0.60, 0.45, 0.30];
  const baseImages = ['baboon', 'barbara', 'bridge', 'coastguard', 'comic', 'face', 'flowers', 'foreman', 'lenna', 'man', 'monarch', 'pepper', 'ppt3', 'zebra'];

  const [tests, setTests] = useState([]);
  const [currentTest, setCurrentTest] = useState(0);
  const [testStarted, setTestStarted] = useState(false);
  const [timeLeft, setTimeLeft] = useState(10);
  const [testData, setTestData] = useState({
    id: Date.now().toString(),
    startTime: new Date().toISOString(),
    comparisonResults: [],
    qualityResults: []
  });

  useEffect(() => {
    const generatedTests = [...baseImages.flatMap(img => scales.map(scale => ({
      type: 'comparison', image: img, scale, showProcessedFirst: Math.random() < 0.5,
    }))), ...baseImages.flatMap(img => scales.map(scale => ({
      type: 'quality', image: img, scale,
    })))].sort(() => Math.random() - 0.5);
    setTests(generatedTests);
  }, []);

  useEffect(() => {
    let timer;
    if (testStarted && timeLeft > 0) {
      timer = setInterval(() => {
        setTimeLeft(prev => prev - 1);
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [testStarted, timeLeft]);

  const handleComparison = async (selectedWorse) => {
    const startTime = Date.now();
    const currentImageTest = tests[currentTest];
    const originalPath = `./processed_images/${currentImageTest.image}_original.jpg`;
    const processedPath = `./processed_images/${currentImageTest.image}_processed_${currentImageTest.scale}.jpg`;

    const [origSize, procSize] = await Promise.all([
      getFileSize(originalPath),
      getFileSize(processedPath)
    ]);

    setTestData(prev => ({
      ...prev,
      comparisonResults: [...prev.comparisonResults, {
        image: currentImageTest.image,
        scale: currentImageTest.scale,
        correctlyIdentified: currentImageTest.showProcessedFirst ?
          (selectedWorse === 'first') : (selectedWorse === 'second'),
        processedShownFirst: currentImageTest.showProcessedFirst,
        file_size_original: origSize,
        file_size_processed: procSize,
        compression_ratio: procSize / origSize,
        image_type: getImageType(currentImageTest.image),
        dominant_content: getContentType(currentImageTest.image),
        response_time: Date.now() - startTime
      }]
    }));
    setTimeLeft(10);
    nextTest();
  };

  const handleRating = async (rating) => {
    const startTime = Date.now();
    const currentImageTest = tests[currentTest];
    const originalPath = `./processed_images/${currentImageTest.image}_original.jpg`;
    const processedPath = `./processed_images/${currentImageTest.image}_processed_${currentImageTest.scale}.jpg`;

    const [origSize, procSize] = await Promise.all([
      getFileSize(originalPath),
      getFileSize(processedPath)
    ]);

    setTestData(prev => ({
      ...prev,
      qualityResults: [...prev.qualityResults, {
        image: currentImageTest.image,
        scale: currentImageTest.scale,
        rating,
        file_size_original: origSize,
        file_size_processed: procSize,
        compression_ratio: procSize / origSize,
        image_type: getImageType(currentImageTest.image),
        dominant_content: getContentType(currentImageTest.image),
        response_time: Date.now() - startTime
      }]
    }));
    setTimeLeft(10);
    nextTest();
  };

  const nextTest = () => {
    if (currentTest === tests.length - 1) {
      finishTest();
    } else {
      setCurrentTest(prev => prev + 1);
    }
  };

  const startTest = () => {
    setTestStarted(true);
    setTimeLeft(10);
    setTestData({
      ...testData,
      startTime: new Date().toISOString()
    });
  };

  const finishTest = () => {
    const finalResults = {
      ...testData,
      endTime: new Date().toISOString(),
      summary: {
        totalTests: tests.length,
        comparisonAccuracy: testData.comparisonResults.filter(r => r.correctlyIdentified).length /
                           testData.comparisonResults.length,
        averageRatingsByScale: scales.reduce((acc, scale) => ({
          ...acc,
          [scale]: {
            average: testData.qualityResults
              .filter(r => r.scale === scale)
              .reduce((sum, r) => sum + r.rating, 0) /
              testData.qualityResults.filter(r => r.scale === scale).length
          }
        }), {})
      }
    };
    const blob = new Blob([JSON.stringify(finalResults, null, 2)]);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quality_test_${finalResults.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!tests.length) {
    return <div className="app-container">Loading...</div>;
  }

  if (!testStarted) {
    return (
      <div className="app-container">
        <button
          className="start-button"
          onClick={startTest}
          style={{
            padding: '1rem 2rem',
            fontSize: '1.2rem',
            fontWeight: 'bold',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          Start Test
        </button>
      </div>
    );
  }

  if (currentTest >= tests.length) {
    return <div className="app-container">Test Complete</div>;
  }

  const currentTestData = tests[currentTest];
  const imagePaths = currentTestData.type === 'comparison' ? {
    first: `./processed_images/${currentTestData.image}_${currentTestData.showProcessedFirst ? 'processed_' + currentTestData.scale : 'original'}.jpg`,
    second: `./processed_images/${currentTestData.image}_${!currentTestData.showProcessedFirst ? 'processed_' + currentTestData.scale : 'original'}.jpg`
  } : {
    image: `./processed_images/${currentTestData.image}_processed_${currentTestData.scale}.jpg`
  };

  return (
    <div className="app-container">
      <div className="main-card">
        <div className="header">
          <h2 className="title">
            {currentTestData.type === 'comparison' ? 'Quality Comparison' : 'Quality Rating'}
          </h2>
          <div className="progress">
            Test {currentTest + 1} of {tests.length}
            <div style={{
              marginLeft: '1rem',
              color: timeLeft === 0 ? 'red' : 'inherit'
            }}>
              Time: {timeLeft}s
            </div>
          </div>
        </div>

        {currentTestData.type === 'comparison' ? (
          <div className="image-section">
            <div className="image-wrapper">
              <img className="test-image" src={imagePaths.first} alt="First" loading="eager" />
              <div className="image-label">1</div>
            </div>
            <div className="image-wrapper">
              <img className="test-image" src={imagePaths.second} alt="Second" loading="eager" />
              <div className="image-label">2</div>
            </div>
          </div>
        ) : (
          <div className="image-section">
            <div className="image-wrapper">
              <img className="test-image" src={imagePaths.image} alt="Rate" loading="eager" />
            </div>
          </div>
        )}

        <div className="control-panel">
          <div className="control-panel-inner">
            <div className="question">
              {currentTestData.type === 'comparison'
                ? 'Select the lower quality image:'
                : 'Rate image quality:'}
            </div>

            {currentTestData.type === 'comparison' ? (
              <div className="button-container">
                <button
                  className="choice-button"
                  onClick={() => handleComparison('first')}
                >
                  Image 1
                </button>
                <button
                  className="choice-button"
                  onClick={() => handleComparison('second')}
                >
                  Image 2
                </button>
              </div>
            ) : (
              <>
                <div className="rating-container">
                  {[1, 2, 3, 4, 5].map(rating => (
                    <button
                      key={rating}
                      className="rating-button"
                      onClick={() => handleRating(rating)}
                    >
                      {rating}
                    </button>
                  ))}
                </div>
                <div className="rating-labels">
                  <span>Poor</span>
                  <span>Excellent</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;