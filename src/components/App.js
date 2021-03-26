import * as tf from "@tensorflow/tfjs";
import { useEffect, createRef } from "react";

import "./styles/App.css";
import { buildGenerator } from "../models/generator";

/**
 * Generate image using generator model
 */
const generateImage = async () => {
  // generate a noise vector (random)
  const noise = tf.randomNormal([1, 100], null, 100, "float32");

  // build the generator
  const generator = buildGenerator();

  // generate image based on a noise
  let generatedImg = generator.predict(noise).add(1).div(2); // get values between 0 and 1

  // reduce the rank to 3 aka [28,28,1]
  generatedImg = tf.unstack(generatedImg)[0];

  // add the image to the canvas
  await tf.browser.toPixels(generatedImg, genImageCanvas.current);
};

let genImageCanvas = createRef();

/**
 * Main application component
 */
const App = () => {

  useEffect(() => {
    // wrapper function to run async code
    const init = async () => {
      await generateImage(genImageCanvas);
    };

    init();
  });

  return (
    <div className="App">
      <div className="ig-generated-image-container">
        <h1>Generated Image</h1>
        <canvas className="ig-generated-image" ref={genImageCanvas} />
      </div>
    </div>
  );
};

export default App;
