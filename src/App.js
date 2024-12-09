import React, { useState, useEffect } from 'react';
import './App.css';

const App = () => {
  const scales = [0.85, 0.66, 0.50, 0.33];
  const methods = ['cubic', 'lanczos'];
  const baseImages = ['baboon', 'bridge', 'flowers', 'man', 'monarch', 'pepper', 'zebra'];

  const [tests, setTests] = useState([]);
  const [currentTest, setCurrentTest] = useState(0);
  const [testStarted, setTestStarted] = useState(false);
  const [timeLeft, setTimeLeft] = useState(10);
  const [questionStartTime, setQuestionStartTime] = useState(null);
  const [testData, setTestData] = useState({
    id: Date.now().toString(),
    startTime: new Date().toISOString(),
    results: []
  });

  useEffect(() => {
    const processedTests = baseImages.flatMap(img =>
      scales.flatMap(scale => methods.map(method => ({
        image: img,
        scale,
        method,
        isOriginal: false
      })))
    );

    const originalTests = baseImages.map(img => ({
      image: img,
      isOriginal: true
    }));

    const singleRoundTests = [...processedTests, ...originalTests];
    const allTests = [...singleRoundTests, ...singleRoundTests].sort(() => Math.random() - 0.5);
    setTests(allTests);
  }, []);

  useEffect(() => {
    let timer;
    if (testStarted && timeLeft > 0) {
      timer = setInterval(() => setTimeLeft(prev => prev - 1), 1000);
    }
    return () => clearInterval(timer);
  }, [testStarted, timeLeft]);

  useEffect(() => {
    if (testStarted) {
      setQuestionStartTime(Date.now());
    }
  }, [currentTest, testStarted]);

  const handleRating = rating => {
    const endTime = Date.now();
    const responseTime = (endTime - questionStartTime) / 1000;

    setTestData(prev => ({
      ...prev,
      results: [...prev.results, {
        ...tests[currentTest],
        rating,
        responseTimeSeconds: responseTime,
        startTime: questionStartTime,
        endTime: endTime
      }]
    }));

    setTimeLeft(10);
    if (currentTest >= tests.length - 1) {
      finishTest();
    } else {
      setCurrentTest(prev => prev + 1);
    }
  };

  const startTest = () => {
    setTestStarted(true);
    setTimeLeft(10);
    setQuestionStartTime(Date.now());
  };

  const finishTest = () => {
    const results = {
      ...testData,
      endTime: new Date().toISOString(),
      summary: {
        totalTests: tests.length,
        averageResponseTime: testData.results.reduce((sum, r) => sum + r.responseTimeSeconds, 0) / testData.results.length,
        averageByScale: scales.reduce((acc, scale) => ({
          ...acc,
          [scale]: testData.results
            .filter(r => !r.isOriginal && r.scale === scale)
            .reduce((sum, r) => sum + r.rating, 0) /
            testData.results.filter(r => !r.isOriginal && r.scale === scale).length || 0
        }), {}),
        averageOriginal: testData.results
          .filter(r => r.isOriginal)
          .reduce((sum, r) => sum + r.rating, 0) /
          testData.results.filter(r => r.isOriginal).length || 0
      }
    };

    const blob = new Blob([JSON.stringify(results, null, 2)]);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quality_test_${results.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!tests.length) return <div className="app-container">Loading...</div>;

  if (!testStarted) {
    return (
      <div className="app-container">
        <div className="welcome-screen">
          <div className="title-section">
            <h1 className="main-title">Image Quality Assessment</h1>
            <p className="subtitle">Rate the quality of processed images in my study</p>
            <div className="stats">
              <div className="stat-item">
                <span className="stat-value">{tests.length}</span>
                <span className="stat-label">Total Images</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">10s</span>
                <span className="stat-label">Per Rating</span>
              </div>
            </div>
          </div>

          <div className="training-samples">
            <div className="sample-card">
              <div className="quality-badge high">Original</div>
              <img src="processed_images/baboon_original.jpg" alt="Original" />
            </div>
            <div className="sample-card">
              <div className="quality-badge low">Low Quality</div>
              <img src="processed_images/baboon_0.33_cubic.jpg" alt="Poor Quality" />
            </div>
          </div>

          <button className="start-button" onClick={startTest}>Begin Assessment</button>
        </div>
      </div>
    );
  }

  if (currentTest >= tests.length) return <div className="app-container">Test Complete</div>;

  const currentTestData = tests[currentTest];
  const imagePath = currentTestData.isOriginal
    ? `processed_images/${currentTestData.image}_original.jpg`
    : `processed_images/${currentTestData.image}_${currentTestData.scale}_${currentTestData.method}.jpg`;

  return (
    <div className="app-container">
      <div className="main-card">
        <div className="header">
          <h2 className="title">Quality Rating</h2>
          <div className="progress">
            Test {currentTest + 1} of {tests.length}
            <div style={{color: timeLeft === 0 ? 'red' : 'inherit'}}>Time: {timeLeft}s</div>
          </div>
        </div>

        <div className="image-section">
          <img
            className="test-image"
            src={imagePath}
            alt="Rate"
            loading="eager"
          />
        </div>

        <div className="control-panel">
          <div className="rating-container">
            {[1,2,3,4,5].map(rating => (
              <button key={rating} className="rating-button" onClick={() => handleRating(rating)}>
                {rating}
              </button>
            ))}
          </div>
          <div className="rating-labels">
            <span>Poor</span>
            <span>Excellent</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;