import * as tf from "@tensorflow/tfjs";
import { useEffect, createRef, useState, useRef } from "react";

import "./styles/App.scss";
import { buildGenerator } from "../models/generator";
import { buildDiscriminator } from "../models/discriminator";

/**
 * display tensor image
 * @param {*} imageTensor
 * @param {*} canvas
 */
const showImage = async (imageTensor, canvas) => {
  // if the rank is 4 unstack the tensor
  if (imageTensor.rank === 4) {
    // reduce the rank to 3 aka [28,28,1]
    imageTensor = tf.unstack(imageTensor)[0];
  }

  // draw on the canvas
  await tf.browser.toPixels(imageTensor, canvas);
};
/**
 * Generate image using generator model
 */
const generateImage = (generator) => {
  // generate a noise vector (random)
  const noise = tf.randomNormal([1, 100], null, 100, "float32");

  // generate image based on a noise
  return generator.predict(noise).add(1).div(2); // get values between 0 and 1
};

let genImageCanvas = createRef();

/**
 * Main application component
 */
const App = () => {
  const [decision, setDecision] = useState(0);

  const generator = useRef(buildGenerator());
  const discriminator = useRef(buildDiscriminator());

  useEffect(() => {
    // generate an image
    const generatedImage = generateImage(generator.current);

    // discriminate
    const result = discriminator.current.predict(generatedImage);

    result.data().then((d) => {
      setDecision(d?.[0]);
    });
    // display the image
    showImage(generatedImage, genImageCanvas.current);
  }, []);

  return (
    <div className="App">
      <div className="ig-generated-image-container">
        <h1>Generated Image</h1>
        <canvas className="ig-generated-image" ref={genImageCanvas} />
      </div>
      <div className="ig-decision">
        <label className="ig-decision-label">Decision:</label>
        <p className="ig-decision-value">{decision}</p>
      </div>
    </div>
  );
};

export default App;
