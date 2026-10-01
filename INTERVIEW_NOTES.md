# Interview Notes

## Dataset choice

The assessment did not provide the promised client dataset, so I used a public
Normal-vs-Defective manufacturing-style dataset as a reproducible substitute.
I chose the AC Piston Checks dataset because it already contains the two labels
required by the assessment. I documented the source instead of presenting it
as client data. If the interviewer provides the real dataset, I would rerun
the exact same pipeline and replace the reported metrics.


These are short reminders for explaining the project naturally. I would explain the reasoning rather than memorizing the wording.

### Why classification?
The requested output is simply Normal or Defective. Classification is enough for that requirement. If the operator later needs the exact defect location, I would change the problem to detection or segmentation.

### Why EfficientNet-B0?
The dataset is small, so transfer learning makes more sense than training from scratch. B0 is also relatively lightweight, which matters when the model has to run as an API.

### Why not accuracy only?
A defect dataset can be imbalanced. A model could get good accuracy while still missing too many defective products. That is why I report precision, recall and F1 as well.

### What is the important error?
A false negative means a defective item was predicted as normal. That is the error I would pay particular attention to in a manufacturing setting.

### How did you handle imbalance?
I measured the training class distribution and used weighted sampling when needed. I did not artificially rebalance validation or test data.

### Why augmentation?
The model should see realistic variations such as small rotations and lighting changes. I avoided transformations that could change the meaning of the defect.

### How did you avoid leakage?
I removed exact duplicates. More importantly, if the dataset contains multiple images from the same product or production batch, I would use a group-based split instead of a simple random split.

### What did you do for error analysis?
The evaluation script writes false positives and false negatives to separate folders. I would open those images and look for recurring visual patterns before deciding what to change.

### How would you improve the model?
I would improve the data first: labels, coverage of defect types, hard examples and production variation. I would only move to a larger model if the error analysis showed that model capacity was the problem.

### How would you reduce latency?
First measure it on the target hardware. Then consider a smaller input size, ONNX, quantization or hardware acceleration if the measured latency requires it.

### What would you monitor after deployment?
API errors, latency, prediction distribution, confidence distribution and performance on periodically labeled production samples.

### What should I not claim?
I should not claim a performance number or an observed failure pattern until I have actually run the supplied dataset and reviewed the resulting files.
