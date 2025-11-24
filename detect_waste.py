from ultralytics import YOLO
import cv2

# LOAD YOUR TRAINED MODEL
model = YOLO('runs/detect/airline_waste_model/weights/best.pt')

# ANALYZE AN IMAGE
image_path = 'test_tray.jpg'  # Replace with a path to a new photo
results = model(image_path)

waste_count = 0
total_count = 0
waste_report = []

print("\n--- TRAY ANALYSIS ---")

for result in results:
    for box in result.boxes:
        # Get Class ID and Name
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        
        total_count += 1
        
        # LOGIC: If the class name contains 'untouched' or 'full', it is waste.
        if 'untouched' in class_name or 'full' in class_name:
            waste_count += 1
            waste_report.append(f"WASTE DETECTED: {class_name}")
        else:
            waste_report.append(f"CONSUMED: {class_name}")

# CALCULATION
if total_count > 0:
    waste_percentage = (waste_count / total_count) * 100
else:
    waste_percentage = 0

print(f"\nTotal Items: {total_count}")
print(f"Wasted Items: {waste_count}")
print(f"Waste Percentage: {waste_percentage:.2f}%")
print("\nDetails:")
for item in waste_report:
    print(item)

# SHOW THE IMAGE WITH BOXES
res_plotted = results[0].plot()
cv2.imshow("Waste Detection", res_plotted)
cv2.waitKey(0)
cv2.destroyAllWindows()
