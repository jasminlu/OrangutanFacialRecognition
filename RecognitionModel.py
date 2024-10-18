from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern
import albumentations as A
import shutil

model = YOLO('/Users/jasminlu/Documents/2024/Semester2/Topics/Orangutan/runs/detect/test6/weights/best.pt')

def checkImageQuality(img):
    if img.shape[0] < 50 or img.shape[1] < 50:
        return False
    return cv2.Laplacian(img, cv2.CV_64F).var() > 100

def deleteLowQualityImages(dir):
    deletedImages = 0
    for root, _, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                imagePath = os.path.join(root, file)
                img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                if img is not None and not checkImageQuality(img):
                    os.remove(imagePath)
                    deletedImages += 1
    return deletedImages

def extractFeatures(img):
    img = cv2.resize(img, (128, 128))
    hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
    hogFeatures = hog.compute(img).flatten()
    lbp = local_binary_pattern(img, 8, 1, method="uniform")
    lbpHist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    return np.concatenate([hogFeatures, lbpHist])

def augmentImage(img):
    augmentation = A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    ])
    augmented = augmentation(image=img)
    return augmented['image']

def loadKnownOrangutans(dir):
    orangutans = {}
    for orangutan in os.listdir(dir):
        orangutanDir = os.path.join(dir, orangutan)
        if not os.path.isdir(orangutanDir):
            continue
        orangutans[orangutan] = []
        for file in os.listdir(orangutanDir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                imagePath = os.path.join(orangutanDir, file)
                img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Original image
                    features = extractFeatures(img)
                    orangutans[orangutan].append(features)
                    
                    # Augmented images
                    for _ in range(5):  # Create 5 augmented versions of each image
                        augmented_img = augmentImage(img)
                        augmented_features = extractFeatures(augmented_img)
                        orangutans[orangutan].append(augmented_features)
    return orangutans

def trainModel(knownOrangutans):
    allFeatures = np.vstack([np.array(features) for features in knownOrangutans.values()])
    scaler = StandardScaler()
    scaledFeatures = scaler.fit_transform(allFeatures)
    return {name: np.mean(scaler.transform(features), axis=0) for name, features in knownOrangutans.items()}, scaler

def predictOrangutan(img, scaledFeatures, scaler, threshold=0.1):
    testFeatures = extractFeatures(img)
    testScaled = scaler.transform(testFeatures.reshape(1, -1))
    similarities = {name: cosine_similarity(testScaled, features.reshape(1, -1))[0][0] 
                    for name, features in scaledFeatures.items()}
    bestName = max(similarities, key=similarities.get)
    bestSimilarity = similarities[bestName]
    return ("Unknown", bestSimilarity) if bestSimilarity < threshold else (bestName, bestSimilarity)

def evaluateModel(directory, scaledFeatures, scaler):
    results = {name: {"true_positives": 0, "false_positives": 0, "false_negatives": 0, "total": 0} 
               for name in scaledFeatures.keys()}
    
    misclassified_dir = os.path.join(directory, "Misclassified")
    if not os.path.exists(misclassified_dir):
        os.makedirs(misclassified_dir)

    for root, _, files in os.walk(directory):
        # Skip the Misclassified directory and its subdirectories
        if "Misclassified" in root.split(os.path.sep):
            continue

        true_orangutan = os.path.basename(root)
        
        # Skip if the current directory is not an orangutan name
        if true_orangutan not in results:
            continue

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                imagePath = os.path.join(root, file)
                img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                if img is not None and checkImageQuality(img):
                    predictedName, _ = predictOrangutan(img, scaledFeatures, scaler)
                    
                    results[true_orangutan]["total"] += 1
                    
                    if predictedName == true_orangutan:
                        results[true_orangutan]["true_positives"] += 1
                    else:
                        results[true_orangutan]["false_negatives"] += 1
                        if predictedName != "Unknown":
                            results[predictedName]["false_positives"] += 1
                        
                        # Move misclassified image
                        misclassified_subdir = os.path.join(misclassified_dir, f"{true_orangutan}_as_{predictedName}")
                        if not os.path.exists(misclassified_subdir):
                            os.makedirs(misclassified_subdir)
                        shutil.copy2(imagePath, os.path.join(misclassified_subdir, file))

    evaluation_results = {}
    for name, result in results.items():
        tp = result["true_positives"]
        fp = result["false_positives"]
        fn = result["false_negatives"]
        total = result["total"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = tp / total if total > 0 else 0
        
        evaluation_results[name] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_predictions": total,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy
        }
    
    return evaluation_results

def processVideo(videoPath, outputPath, yoloModel, scaledFeatures, scaler):
    cap = cv2.VideoCapture(videoPath)
    out = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'mp4v'), 
                          cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = yoloModel(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if float(box.conf) > 0.4:
                    faceImg = frame[y1:y2, x1:x2]
                    if faceImg.size > 0:
                        grayFace = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)
                        orangutanId, recognitionConfidence = predictOrangutan(grayFace, scaledFeatures, scaler)
                        color = (0, 255, 0) if orangutanId != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{orangutanId} ({recognitionConfidence:.2f})", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        out.write(frame)
        cv2.imshow('Orangutan Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Main execution
baseDir = '/Users/jasminlu/Documents/2024/Semester2/Topics/Orangutan/test'
trainDir = os.path.join(baseDir, 'Training')
testDir = os.path.join(baseDir, 'Testing')

# Delete low quality images
deletedCount = deleteLowQualityImages(trainDir)
print(f"Deleted {deletedCount} low quality images from the training set.")

# Load and train the model
knownOrangutans = loadKnownOrangutans(trainDir)
scaledFeatures, scaler = trainModel(knownOrangutans)

# Evaluate the model
evaluationResults = evaluateModel(testDir, scaledFeatures, scaler)

for orangutan, result in evaluationResults.items():
    print(f"Evaluation results for {orangutan}:")
    print(f"  True Positives: {result['true_positives']}")
    print(f"  False Positives: {result['false_positives']}")
    print(f"  False Negatives: {result['false_negatives']}")
    print(f"  Total predictions: {result['total_predictions']}")
    print(f"  Precision: {result['precision']:.2%}")
    print(f"  Recall: {result['recall']:.2%}")
    print(f"  F1 Score: {result['f1_score']:.2%}")
    print(f"  Accuracy: {result['accuracy']:.2%}")

videoPath = '/Users/jasminlu/Documents/2024/Semester2/Topics/Orangutan/Videos/new/IMG_5532.MOV'
outputPath = '/Users/jasminlu/Documents/2024/Semester2/Topics/Orangutan/output_video.mp4'
processVideo(videoPath, outputPath, model, scaledFeatures, scaler)