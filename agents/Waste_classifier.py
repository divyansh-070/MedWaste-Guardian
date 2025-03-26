from crewai import Agent
from ultralytics import YOLO

class WasteClassificationAgent(Agent):
    def __init__(self, model_path="/Users/devayushrout/Desktop/MedWaste Guardian/resultsyolov8/yolov8_medical_waste/weights/best.pt"):
        self.yolo_model = YOLO(model_path)
        super().__init__(
            name="Waste Classification Agent",
            role="Biomedical Waste Classifier",
            goal="Classify biomedical waste and send the category to the compliance agent."
        )
        
    def run(self, image_path):
        results = self.yolo_model(image_path)
        if results[0].boxes:
            waste_category = results[0].names[int(results[0].boxes.cls[0].item())]
            return waste_category
        else:
            return "Unknown Waste Type"