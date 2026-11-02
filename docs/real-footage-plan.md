# From made-up video to real broadcasts

This prototype runs on rendered matches so that every frame has exact ground
truth. Moving to real eBasketball and eSoccer streams would follow these steps.

## 1. Scoreboard reading

- **Templates.** Each broadcast layout gets a small config file with the
  position of every digit slot. A new layout can be set up in minutes from one
  screenshot.
- **Labelled crops.** Label a few thousand real digit crops per layout
  (semi-automatically: run the current reader, check only low-confidence
  cases), then fine-tune the CNN. The temporal checks stay the same.
- **Cross-checks.** Compare the scoreboard with the game's own data feed where
  one exists, and alert on any disagreement.

## 2. Players, ball and gamers

- **Detection.** Replace colour-based detection with a trained detector (for
  example a small YOLO model) fine-tuned on labelled frames, because real
  players overlap, have animations and share colours with the background.
- **Gamers on camera.** For streams that show the gamers, add pose
  estimation (for example MediaPipe or RTMPose) to track hand and body
  movement over time. This can support integrity checks, such as a player
  stopping input during key moments.
- **Temporal models.** Detect events from short sequences of frames (for
  example a small 3D CNN or a transformer over tracked positions) instead of
  single-frame rules, once enough labelled events exist.

## 3. Speed

- Target: every frame processed in under 40 ms (25 fps) with a stable p99.
- Run the CNN in batches on a GPU or with ONNX Runtime, decode video on the
  GPU, and process only the scoreboard region on every frame and the full
  frame at a lower rate if needed.
- Measure end-to-end delay from the live stream to the price update, not just
  model time.

## 4. Deployment on Azure

- Package the pipeline as a container; run it on Azure Kubernetes Service or
  Azure Container Apps, one worker per live match.
- Stream events to the existing Java and .NET systems through a message bus
  (for example Azure Event Hubs or Service Bus), with the same JSON format the
  Java service in this repo already reads.
- Store events and model versions so every price can be traced back to the
  frames and model that produced it.

## 5. Prices and integrity

- Retrain the in-play model on real match histories per game title, and check
  calibration every week.
- For integrity, combine score-based signals with richer ones that only real
  data has: player inputs, in-game telemetry, betting market movements.
  The prototype shows why: short spells of poor play cannot be told apart
  from bad luck using the score alone.
- Every integrity flag goes to a person (and, where relevant, to the Esports
  Integrity Commission process), never to an automatic action.
