import os

image_path = "C:/Users/naray/Desktop/fire_gan/archive/forest_fire/Testing/abc169.jpg"

if os.path.exists(image_path):
    print("✅ File exists:", image_path)
else:
    print("❌ File NOT found! Check the file path.")
    print("Available files:", os.listdir("C:/Users/naray/Desktop/fire_gan/archive/forest_fire/Testing/"))
