# R2D2: Repeatable and Reliable Detector and Descriptor

## Core Contribution

R2D2 challenges the traditional detect-and-describe paradigm by jointly learning keypoint detection and secription while making a critical distinction between repeatability (can we detect this point consistently?) and reliability (can we match this descriptor accurately).

## Key Insights

Repeatable != Reliable: Salient regions are not necessarily discriminative. The paper's tou example shows this perfectly.

Black triangle: Only repeatable location, but all patches containing it are equally reliable.
Checkerboard: All corners are highly repeatable (salient), but none are discriminative due to self-similarity, you cant match them.

This same problem appears in real scenes: tree leaves, skyscraper windows, sea waves, all salient but hard to match.

## Architecture

Fully convolutional L2-Net that outputs 3 components per image:

1. Dense descriptors (HxWx128): One descriptor per pixel
2. Repeatability map (HxWx1): Probability that the descriptor is repeatable across viewpoints
3. Reliability map (HxWx1): Probability that the descriptor can be accurately matched (discriminative)

Final keypoints: Maximize both confidence maps (element-wise product)

## Training Methodology (Self-Supervised)

Novel unsupervised detector loss:
Encourages repeatability across viewpoint changes
Enforces sparsity (not too many keypoints)
Ensures uniform spatial coverage

Descriptor learning:
AP-based ranking loss (not triplet/contrastive): Optimizes average precision metric directly
Considers multiple descriptors per batch (listwise ranking)
More robust than pairwise losses

Reliability estimation:
Trained to predict which pixels will have high AP descriptors
Jointly learned with descriptor - single network predicts both descriptor and its expected matching quality

Key results:
State-of-the-art on HPatches dataset
Record performance on Aachen Day-Night localization benchmark
Ablation shows both repeatability AND reliability are essential, removing either significantly drops performance

## How R2D2 Relates to my Thesis

### Direct Relevance, Core Architecture Inspiration

R2D2 is essentially your blueprint. Your three learned heads map almost directly to R2D2's architecture.

Keypoint Selector Head -> Saliency heatmap = Repeatability map
Descriptor Refiner Head -> 128-dim descriptors = Dense Descriptor Network
Uncertainty Estimator Head -> Confidence Scores = Reliability map

R2D2's repeatability vs reliability distinction is exactly what ill implement with seperate selector and uncertainty heads.

### What You Can Adopt from R2D2

Architecture ideas:

✅ Separate confidence maps for selection vs matching quality (they validate this works!)
✅ Element-wise product of repeatability × reliability for final keypoint selection
✅ AP-based descriptor loss instead of triplet loss (proven more effective)

Training strategies:

Consider their sparsity + uniform coverage loss for your keypoint selector
Their reliability calibration approach (predicting expected AP) could inform your uncertainty estimator
Ablation methodology: They show removing either map hurts performance — you should do the same

What to avoid:

❌ Don't train from scratch like R2D2 — your DINOv2 backbone is your advantage
❌ Their homography-based training might not be optimal for SLAM (you want temporal consistency, not just viewpoint invariance)