import cv2

# Đường dẫn ảnh gốc
img_path = r"F:\PyCharmProjects\homography-computation\PhysicalAI-SmartSpaces_py\MTMC_Tracking_2025\train\Warehouse_012\map.png"

# Đọc ảnh
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Không thể mở file: {img_path}")

# Xoay ảnh 180 độ
rotated = cv2.rotate(img, cv2.ROTATE_180)

# Lật ảnh (flip theo trục dọc: 1 = mirror trái-phải, 0 = lật trên-dưới, -1 = cả hai)
flipped = cv2.flip(rotated, 1)

# Lưu ảnh kết quả
output_path = img_path.replace(".png", "_rotated_flipped.png")
cv2.imwrite(output_path, flipped)

print(f"Ảnh đã được lưu tại: {output_path}")
